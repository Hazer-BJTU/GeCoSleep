import torch.nn
from models import *
from clnetworks import CLnetwork
from torch.utils.data import DataLoader


class EWCnetwork(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(EWCnetwork, self).__init__(args, fold_num, logs)
        '''initialize FIM'''
        self.fim = {}
        for name, param in self.net.named_parameters():
            self.fim[name] = torch.zeros_like(param.data, dtype=torch.float32, requires_grad=False, device=self.device)
        '''ewc settings'''
        self.teacher_model = SleepNet(self.num_channels, args.dropout)
        self.teacher_model.to(self.device)
        self.reg_value = 0

    def estimate_fisher(self, dataset):
        self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
        print(f'best model loaded: {self.best_net_memory[-1]}')
        new_fim = {}
        for name, param in self.teacher_model.named_parameters():
            new_fim[name] = torch.zeros_like(param.data, dtype=torch.float32, requires_grad=False, device=self.device)
        loader = DataLoader(dataset, self.args.batch_size, True)
        cnt, num_smaples = 0, 0
        self.teacher_model.eval()
        print(f'start calculating FIM on {self.args.ewc_batches} batches...')
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.teacher_model.zero_grad()
            y_hat = self.teacher_model(X)
            L = torch.mean(self.loss(y_hat, y.view(-1)))
            L.backward()
            for name, param in self.teacher_model.named_parameters():
                if param.grad is not None:
                    new_fim[name] += param.grad.data.pow(2) * X.shape[0]
                    num_smaples += X.shape[0]
            cnt += 1
            if cnt == self.args.ewc_batches:
                break
        for name, value in new_fim.items():
            if name in self.fim:
                value /= num_smaples
                if self.task == 0:
                    self.fim[name] = value
                else:
                    self.fim[name] = self.fim[name] * (1 - self.args.ewc_gamma) + value * self.args.ewc_gamma
        print('FIM updated')

    def start_task(self):
        super(EWCnetwork, self).start_task()
        self.reg_value = 0
        '''load teacher model'''
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')

    def start_epoch(self):
        super(EWCnetwork, self).start_epoch()

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if self.task > 0:
            self.net.freeze_parameters()
        y_hat = self.net(X)
        L_current = self.loss(y_hat, y.view(-1))
        L = torch.mean(L_current)
        if self.task > 0:
            '''start EWC regularization'''
            reg = 0
            self.teacher_model.eval()
            for (name_new, param_new), (name_old, param_old) in zip(self.net.named_parameters(), self.teacher_model.named_parameters()):
                if name_new in self.fim:
                    reg = reg + torch.sum(self.fim[name_new] * (param_new - param_old.detach()).pow(2))
            L = L + self.args.ewc_lambda * reg
            self.reg_value += self.args.ewc_lambda * reg.item()
        L.backward()
        self.optimizer.step()
        self.train_loss += L.item()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        super(EWCnetwork, self).end_epoch(valid_dataset)
        self.logs.append(
            ['train_info', f'task{self.task}_fold{self.fold_num}', f'epoch:{self.epoch - 1}', 'ewc reg'],
            self.reg_value / self.cnt
        )

    def end_task(self, dataset=None):
        super(EWCnetwork, self).end_task()
        '''update FIM'''
        self.estimate_fisher(dataset)


if __name__ == '__main__':
    '''
    exp_log = LogDocument(args)
    net = EWCnetwork(args, 0, exp_log)
    print(net.fim)
    '''
    pass
