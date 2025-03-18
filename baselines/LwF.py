import torch.nn
from models import *
from . import multihead_model
from clnetworks import CLnetwork
from metric import evaluate_tasks_multihead, ConfusionMatrix


class LwFnetwork(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(LwFnetwork, self).__init__(args, fold_num, logs)
        self.net = multihead_model.MultiHeadSleepNet(self.num_channels, args.dropout, args.task_num, self.args.enable_multihead)
        self.net.apply(init_weight)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.net.to(self.device)
        self.kldloss = nn.KLDivLoss(reduction='none')
        '''distill settings'''
        self.teacher_model = multihead_model.MultiHeadSleepNet(self.num_channels, args.dropout, args.task_num, self.args.enable_multihead)
        self.teacher_model.to(self.device)

    def start_task(self):
        super(LwFnetwork, self).start_task()
        '''load teacher models'''
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')

    def start_epoch(self):
        super(LwFnetwork, self).start_epoch()
        self.teacher_model.eval()

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if self.task > 0:
            self.net.freeze_parameters()
        y_hat = self.net(X, self.task)
        L_current = self.loss(y_hat, y.view(-1))
        L = torch.mean(L_current)
        if self.task > 0:
            '''perform knowledge distillation'''
            y_distill = self.teacher_model(X, self.task - 1).detach() / self.args.tau
            L_distill = torch.sum(self.kldloss(nn.functional.log_softmax(y_hat / self.args.tau, dim=1), y_distill.softmax(dim=1)), dim=1)
            L = L + torch.mean(L_distill)
        L.backward()
        self.optimizer.step()
        self.train_loss += L.item()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
        train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
        print(f'epoch: {self.epoch}, train loss: {self.train_loss / self.cnt:.3f}, train accuracy: {train_acc:.3f}, '
              f"macro F1: {train_mf1:.3f}, 1000 lr: {learning_rate * 1000:.3f}")
        self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'epoch:{self.epoch}'], {
            'train loss': self.train_loss / self.cnt,
            'train accuracy': train_acc,
            'train mF1': train_mf1,
            '1000 lr': learning_rate * 1000
        })
        if (self.epoch + 1) % self.args.valid_epoch == 0:
            print(f'validating on the datasets...')
            valid_confusion = ConfusionMatrix(1)
            valid_confusion = evaluate_tasks_multihead(self.net, [valid_dataset], valid_confusion,
                                                       self.device, self.args.valid_batch, self.task)
            valid_acc, valid_mf1 = valid_confusion.accuracy(), valid_confusion.macro_f1()
            print(f'valid accuracy: {valid_acc:.3f}, valid macro F1: {valid_mf1:.3f}')
            self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'valid epoch:{self.epoch}'], {
                'valid accuracy': valid_acc,
                'valid mF1': valid_mf1
            })
            if valid_acc + valid_mf1 > self.best_valid_acc and self.epoch + 1 >= self.args.min_epoch:
                self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'valid epoch:{self.epoch}', 'saved'], True)
                self.best_train_loss = self.train_loss / self.cnt
                self.best_train_acc = train_acc
                self.best_valid_acc = valid_acc + valid_mf1
                self.best_net = './modelsaved/' + str(self.args.replay_mode) + '_task' + str(self.task) + '_fold' + str(self.fold_num) + '.pth'
                print(f'model saved: {self.best_net}')
                torch.save(self.net.state_dict(), self.best_net)
        self.epoch += 1
        self.scheduler.step()

    def end_task(self, dataset=None):
        super(LwFnetwork, self).end_task()


if __name__ == '__main__':
    pass
