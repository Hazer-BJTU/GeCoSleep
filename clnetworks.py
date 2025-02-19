import random
from models import *
from metric import *


class CLnetwork:
    def __init__(self, args, fold_num, logs):
        self.args = args
        self.fold_num = fold_num
        self.logs = logs
        self.num_channels = len(args.isruc1)
        self.net = SleepNet(self.num_channels, args.dropout)
        self.net.apply(init_weight)
        self.scheduler = None
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0.0, 0.0, 0.0
        self.train_loss, self.confusion_matrix, self.cnt = 0.0, ConfusionMatrix(1), 0
        self.best_net = None
        self.best_net_memory = []
        self.device = torch.device(f'cuda:{args.cuda_idx}')
        self.net.to(self.device)
        self.epoch = 0
        self.task = 0
        self.label_cnt = None

    def start_task(self):
        self.epoch = 0
        if self.task > 0:
            self.net.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
        self.best_net = None
        self.label_cnt = torch.zeros(5, dtype=torch.float32, device=self.device, requires_grad=False)
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0.0, 0.0, 0.0
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.args.num_epochs // 6, 1), 0.6)

    def start_epoch(self):
        self.train_loss, self.cnt = 0.0, 0
        self.confusion_matrix.clear()
        self.net.train()

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        L_current = self.loss(y_hat, y.view(-1))
        L = torch.mean(L_current)
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
            valid_confusion = evaluate_tasks(self.net, [valid_dataset], valid_confusion,
                                             self.device, self.args.valid_batch)
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

    def end_task(self):
        self.task += 1
        self.best_net_memory.append(self.best_net)


class FineTuning(CLnetwork):
    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        if self.task > 0:
            '''freeze feature extractor'''
            self.net.freeze_parameters()
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        L_current = self.loss(y_hat, y.view(-1))
        L = torch.mean(L_current)
        L.backward()
        self.optimizer.step()
        self.train_loss += L.item()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(y_hat, y, 0)


class Independent(CLnetwork):
    def start_task(self):
        self.epoch = 0
        if self.task > 0:
            self.net.apply(init_weight)
        self.best_net = None
        self.label_cnt = torch.zeros(5, dtype=torch.float32, device=self.device, requires_grad=False)
        self.best_train_loss, self.best_train_acc, self.best_valid_acc = 0.0, 0.0, 0.0
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.args.num_epochs // 6, 1), 0.6)


class ExperienceReplay(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(ExperienceReplay, self).__init__(args, fold_num, logs)
        self.replay_buffer = 0
        self.sample_buffer = None
        self.label_buffer = None

    def start_task(self):
        super(ExperienceReplay, self).start_task()
        self.replay_buffer += self.args.replay_buffer

    def update_buffer(self, X, y):
        if self.sample_buffer is None or self.label_buffer is None:
            if X.shape[0] >= self.replay_buffer:
                self.sample_buffer = X[:self.replay_buffer, :, :, :]
                self.label_buffer = y[:self.replay_buffer, :]
            else:
                self.sample_buffer = X
                self.label_buffer = y
        else:
            remaining = self.replay_buffer - self.sample_buffer.shape[0]
            if remaining == 0:
                return
            if X.shape[0] >= remaining:
                self.sample_buffer = torch.cat((self.sample_buffer, X[:remaining, :, :, :]), dim=0)
                self.label_buffer = torch.cat((self.label_buffer, y[:remaining, :]), dim=0)
            else:
                self.sample_buffer = torch.cat((self.sample_buffer, X), dim=0)
                self.label_buffer = torch.cat((self.label_buffer, y), dim=0)

    def observe(self, X, y, first_time=False):
        if first_time:
            self.update_buffer(X, y)
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if self.task > 0:
            self.net.freeze_parameters()
        y_hat = self.net(X)
        L_current = self.loss(y_hat, y.view(-1))
        L = torch.mean(L_current)
        if self.task > 0:
            selected = random.sample(list(range(self.sample_buffer.shape[0])), self.args.batch_size)
            Xr, yr = self.sample_buffer[selected], self.label_buffer[selected]
            Xr, yr = Xr.to(self.device), yr.to(self.device)
            yr_hat = self.net(Xr)
            L_replay = self.loss(yr_hat, yr.view(-1))
            L = L + torch.mean(L_replay)
        L.backward()
        self.optimizer.step()
        self.train_loss += L.item()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        super(ExperienceReplay, self).end_epoch(valid_dataset)
        self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'epoch:{self.epoch - 1}',
                          'replay buffer size'], self.sample_buffer.shape[0])


if __name__ == '__main__':
    pass
