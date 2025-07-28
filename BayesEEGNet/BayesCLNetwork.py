import torch
import torch.nn as nn
from .BayesEEGNet import BayesEEGNet
from clnetworks import CLnetwork, linear_warmup_cosine_annealing
from metric import ConfusionMatrix, evaluate_tasks_bayes


def get_bayes_eeg_optimizer(net, params):
    if params['optimizer'] == 'SGD':
        return torch.optim.SGD(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adam':
        return torch.optim.Adam(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'Adamax':
        return torch.optim.Adamax(net.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])


class BayesCLNetwork(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(BayesCLNetwork, self).__init__(args, fold_num, logs)
        params = self.params = self.args.bayes_eeg_params
        self.net = BayesEEGNet(
            hidden_size=params['hiddenDim'],
            output_size=params['targetDim'],
            graph_node_dim=params['graph_dim'],
            num_nodes=params['num_nodes'],
            last_dense=params['dense']
        )
        self.scheduler = None
        self.optimizer = get_bayes_eeg_optimizer(self.net, params)
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(params['loss_score']).cuda())
        self.net.to(self.device)
        self.train_loss_kl1, self.train_loss_kl2 = 0, 0

    def start_task(self):
        super(BayesCLNetwork, self).start_task()
        self.optimizer = get_bayes_eeg_optimizer(self.net, self.params)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=linear_warmup_cosine_annealing(self.args.num_epochs, self.args.min_epoch)
        )

    def start_epoch(self):
        super(BayesCLNetwork, self).start_epoch()
        self.train_loss_kl1, self.train_loss_kl2 = 0, 0

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        batch_size, window_size, num_channels, series = X.shape
        X = X.view(batch_size * window_size, num_channels, series)
        y = y.view(-1)
        self.optimizer.zero_grad()
        outputs = self.net(X)
        loss_kl1 = self.params['lamada1'] * outputs['kl_g']
        loss_kl2 = self.params['lamada2'] * outputs['kl_b']
        L = self.loss(outputs['y_hat'], y) + loss_kl1 + loss_kl2
        L.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 1.0)
        self.optimizer.step()
        self.train_loss += L.item()
        self.train_loss_kl1 += loss_kl1.item()
        self.train_loss_kl2 += loss_kl2.item()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(outputs['y_hat'], y, 0)

    def end_epoch(self, valid_dataset):
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
        train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
        print(f'epoch: {self.epoch}, train loss: {self.train_loss / self.cnt:.3f}, '
              f'1e4 loss kl1: {self.train_loss_kl1 / self.cnt * 1e4:.3f}, '
              f'1e8 loss kl2: {self.train_loss_kl2 / self.cnt * 1e8:.3f}, '
              f'train accuracy: {train_acc:.3f}, '
              f"macro F1: {train_mf1:.3f}, 1000 lr: {learning_rate * 1000:.3f}")
        self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'epoch:{self.epoch}'], {
            'train loss': self.train_loss / self.cnt,
            'loss kl1': self.train_loss_kl1 / self.cnt,
            'loss kl2': self.train_loss_kl2 / self.cnt,
            'train accuracy': train_acc,
            'train mF1': train_mf1,
            '1000 lr': learning_rate * 1000
        })
        if (self.epoch + 1) % self.args.valid_epoch == 0:
            print(f'validating on the datasets...')
            valid_confusion = ConfusionMatrix(1)
            valid_confusion = evaluate_tasks_bayes(self.net, [valid_dataset], valid_confusion,
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

    def end_task(self, dataset=None):
        super(BayesCLNetwork, self).end_task(dataset)


if __name__ == '__main__':
    pass
