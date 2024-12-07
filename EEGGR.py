import torch
import torch.nn
from clnetworks import CLnetwork
from models import *
from generator import *


class EEGGRnetwork(CLnetwork):
    def __init__(self, args, fold_num):
        super(EEGGRnetwork, self).__init__(args, fold_num)
        '''generator settings'''
        self.start_training_generator = False
        self.num_epochs_solver = self.args.num_epochs - self.args.num_epochs_generator
        self.generator = EEGVAE(2)
        self.generator.apply(init_weight)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=args.lr_generator)
        self.schedulerG = None
        self.rec_loss, self.kl_loss, self.feautre_loss = 0, 0, 0
        self.generator.to(self.device)
        self.mseloss = nn.MSELoss()

    def start_task(self):
        super(EEGGRnetwork, self).start_task()
        '''generator settings'''
        self.start_training_generator = False
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.num_epochs_solver // 6, 1), 0.6)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_generator)
        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, max(self.args.num_epochs_generator // 6, 1), 0.6)

    def start_epoch(self):
        super(EEGGRnetwork, self).start_epoch()
        '''generator settings'''
        self.rec_loss, self.kl_loss, self.feautre_loss = 0, 0, 0
        self.generator.train()

    def observe(self, X, y, first_time=False):
        if self.epoch < self.num_epochs_solver:
            super(EEGGRnetwork, self).observe(X, y, first_time)
        else:
            if not self.start_training_generator:
                print('start training generator...')
                self.net.load_state_dict(torch.load(self.best_net, map_location=self.device, weights_only=True))
                print(f'best solver model loaded: {self.best_net}')
                self.start_training_generator = True
            '''freeze BN and dropout'''
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Dropout):
                    m.eval()
            X, y = X.to(self.device), y.to(self.device)
            self.optimizerG.zero_grad()
            self.optimizer.zero_grad()
            X_hat, L_kl = self.generator(X)
            L_rec = self.mseloss(X_hat, X)
            F = self.net.feaure_map(X)
            F_hat = self.net.feaure_map(X_hat)
            L_feature = self.mseloss(F_hat, F.detach())
            (L_rec + L_feature + self.args.beta * L_kl).backward()
            self.optimizerG.step()
            self.rec_loss += L_rec.item()
            self.kl_loss += L_kl.item()
            self.feautre_loss += L_feature.item()
            self.cnt += 1

    def end_epoch(self, valid_dataset):
        if self.epoch < self.num_epochs_solver:
            super(EEGGRnetwork, self).end_epoch(valid_dataset)
        else:
            print(f'epoch: {self.epoch}, reconstruction loss: {self.rec_loss / self.cnt:.3f}, '
                  f'feature loss: {self.feautre_loss / self.cnt:.3f}, kl loss: {self.kl_loss / self.cnt:.3f}, '
                  f"1000 lr: {self.optimizerG.state_dict()['param_groups'][0]['lr'] * 1000:.3f}")
            self.epoch += 1
            self.schedulerG.step()
            if self.epoch + 1 == self.args.num_epochs:
                generator_path = './modelsaved/generator_task' + str(self.task) + '_fold' + str(self.flod_num) + '.pth'
                torch.save(self.generator.state_dict(), generator_path)

    def end_task(self):
        super(EEGGRnetwork, self).end_task()

        