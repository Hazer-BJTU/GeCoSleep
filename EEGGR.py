import torch
import torch.nn
from clnetworks import CLnetwork
from models import *
from generator import *


class EEGGRnetwork(CLnetwork):
    def __init__(self, args):
        super(EEGGRnetwork, self).__init__(args)
        self.num_epochs_solver = self.args.num_epochs - self.args.num_epochs_generator
        self.generator = EEGVAE(2, args.window_size)
        self.generator.apply(init_weight)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=args.lr_generator)
        self.rec_loss, self.kl_loss = 0, 0
        self.generator.to(self.device)
        self.mseloss = nn.MSELoss()

    def start_task(self):
        super(EEGGRnetwork, self).start_task()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.num_epochs_solver // 6, 1), 0.6)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_generator)

    def start_epoch(self):
        super(EEGGRnetwork, self).start_epoch()
        self.rec_loss, self.kl_loss = 0, 0
        self.generator.train()

    def observe(self, X, y, first_time=False):
        if self.epoch < self.num_epochs_solver:
            super(EEGGRnetwork, self).observe(X, y, first_time)
        else:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizerG.zero_grad()
            X_hat, L_kl = self.generator(X)
            L_rec = self.mseloss(X_hat, X)
            (L_rec + L_kl).backward()
            self.optimizerG.step()
            self.rec_loss += L_rec.item()
            self.kl_loss += L_kl.item()
            self.cnt += 1

    def end_epoch(self, valid_dataset):
        if self.epoch < self.num_epochs_solver:
            super(EEGGRnetwork, self).end_epoch(valid_dataset)
        else:
            print(f'epoch: {self.epoch}, reconstruction loss: {self.rec_loss / self.cnt:.3f}, '
                  f"kl loss {self.kl_loss / self.cnt:.3f}, 1000 lr: {self.optimizerG.state_dict()['param_groups'][0]['lr'] * 1000:.3f}")
            self.epoch += 1

    def end_task(self):
        super(EEGGRnetwork, self).end_task()

        