import torch
import random
from models import *
from clnetworks import ExperienceReplay


class DERnetwork(ExperienceReplay):
    def __init__(self, args, fold_num, logs):
        super(DERnetwork, self).__init__(args, fold_num, logs)
        self.kldloss = nn.KLDivLoss(reduction='none')
        self.mseloss = nn.MSELoss()
        '''replay settings'''
        self.teacher_model = SleepNet(self.num_channels, args.dropout)
        self.teacher_model.to(self.device)

    def start_task(self):
        super(DERnetwork, self).start_task()
        '''load teacher model'''
        if self.task > 0:
            self.teacher_model.load_state_dict(
                torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')

    def start_epoch(self):
        super(DERnetwork, self).start_epoch()
        self.teacher_model.eval()

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
            yr_distill = self.teacher_model(Xr).detach()
            yr_hat = self.net(Xr)
            L_kl = torch.sum(self.kldloss(nn.functional.log_softmax(y_hat, dim=1), yr_distill.softmax(dim=1)), dim=1)
            L_ed = self.mseloss(yr_hat, yr_distill)
            L = L + self.args.der_alpha * L_kl + self.args.der_beta * L_ed
        L.backward()
        self.optimizer.step()
        self.train_loss += L.item()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(y_hat, y, 0)


if __name__ == '__main__':
    pass
