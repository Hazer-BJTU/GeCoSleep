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
        self.dark_experience = None
        self.teacher_model = SleepNet(self.num_channels, args.dropout)
        self.teacher_model.to(self.device)

    def start_task(self):
        super(DERnetwork, self).start_task()
        '''load teacher model
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')'''

    def start_epoch(self):
        super(DERnetwork, self).start_epoch()
        self.teacher_model.eval()

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        if self.task > 0:
            self.net.freeze_parameters()
        y_hat = self.net(X)
        L_current = self.loss(y_hat, y.view(-1))
        L = torch.mean(L_current)
        if self.task > 0:
            selected = random.sample(list(range(self.replay_buffer_size)), self.args.batch_size)
            Xr, dr, yr = self.sample_buffer[selected], self.dark_experience[selected], self.label_buffer[selected]
            Xr, dr, yr = Xr.to(self.device), dr.to(self.device), yr.to(self.device)
            yr_hat = self.net(Xr)
            L_cl = self.loss(yr_hat, yr.view(-1))
            L_ed = self.mseloss(yr_hat, dr.view(-1, 5))
            L = L + self.args.der_alpha * torch.mean(L_cl) + self.args.der_beta * L_ed
        L.backward()
        self.optimizer.step()
        self.train_loss += L.item()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_task(self, dataset=None):
        super(DERnetwork, self).end_task(dataset)
        batches = torch.split(self.sample_buffer, self.args.batch_size, dim=0)
        self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
        print(f'teacher model loaded: {self.best_net_memory[-1]}')
        self.teacher_model.eval()
        print(f'start calculating dark experience...')
        with torch.no_grad():
            if self.dark_experience is None:
                pred_list = []
                for batch in batches:
                    batch = batch.to(self.device)
                    pred = self.teacher_model(batch)
                    pred_list.append(pred.view(self.args.batch_size, self.args.window_size, 5).to('cpu'))
                self.dark_experience = torch.cat(pred_list, dim=0)
                print(f'calculation complete')
            else:
                cnt = 0
                for idx in range(self.task_tag.shape[0]):
                    if self.task_tag[idx] == self.task:
                        sample = self.sample_buffer[idx].unsqueeze(0).to(self.device)
                        pred = self.teacher_model(sample)
                        '''print(self.dark_experience[idx].shape, pred.shape)'''
                        self.dark_experience[idx].copy_(pred.to('cpu'))
                        cnt += 1
                print(f'{cnt} samples updated')


if __name__ == '__main__':
    pass
