import copy

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
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=args.lr_generator)
        self.schedulerG = None
        self.rec_loss, self.kl_loss, self.task_loss = 0, 0, 0
        self.generator.to(self.device)
        self.mseloss = nn.MSELoss()
        '''replay settings'''
        self.teacher_model = SleepNet(2, args.dropout)
        self.teacher_model.to(self.device)
        self.generators = []
        self.replay_buffer = None
        self.replay_coef = None

    def generate_replay_buffer(self):
        self.replay_buffer, self.replay_coef = None, None
        self.teacher_model.eval()
        print('start generating replay samples, teachers loading: ')
        for teacher, decoder in zip(self.best_net_memory, self.generators):
            decoder.eval()
            self.teacher_model.load_state_dict(torch.load(teacher, map_location=self.device, weights_only=True))
            print(f'teacher model: {teacher}')
            self.teacher_model.eval()
            X_generated = decoder.generate(self.args.replay_buffer, self.device).detach()
            y_generated = self.teacher_model(X_generated).detach()
            temp = torch.ones(self.args.replay_buffer * self.args.window_size, dtype=torch.float32, device=self.device, requires_grad=False)
            if self.replay_buffer is None:
                self.replay_buffer = [X_generated, y_generated]
                self.replay_coef = temp
            else:
                self.replay_buffer[0] = torch.cat((self.replay_buffer[0], X_generated), dim=0)
                self.replay_buffer[1] = torch.cat((self.replay_buffer[1], y_generated), dim=0)
                self.replay_coef = torch.cat((2 * self.replay_coef, temp), dim=0)
        print(f'replay buffer {self.replay_buffer[0].shape[0]} samples generated')

    def start_task(self):
        super(EEGGRnetwork, self).start_task()
        '''generator settings'''
        self.start_training_generator = False
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.num_epochs_solver // 6, 1), 0.6)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_generator)
        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, max(self.args.num_epochs_generator // 6, 1), 0.6)
        '''replay settings
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')'''

    def start_epoch(self):
        super(EEGGRnetwork, self).start_epoch()
        '''generator settings'''
        self.rec_loss, self.kl_loss, self.task_loss = 0, 0, 0
        self.generator.train()
        '''generate replay buffer'''
        if self.task > 0 and not self.start_training_generator and self.epoch % self.args.generate_epoch == 0:
            self.generate_replay_buffer()

    def observe(self, X, y, first_time=False):
        if self.epoch < self.num_epochs_solver:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if self.task > 0:
                self.net.freeze_parameters()
            y_hat = self.net(X)
            L_current = self.loss(y_hat, y.view(-1))
            L = torch.mean(L_current)
            if self.task > 0:
                '''perform generative replay'''
                '''print(f'start generative replay on {len(self.generators)} tasks:')'''
                X_replay, y_replay = self.replay_buffer[0], self.replay_buffer[1]
                y_replay_hat = self.net(X_replay)
                L_replay = torch.mean(self.loss(y_replay_hat, y_replay.softmax(dim=1)) * self.replay_coef)
                L = L + L_replay
            L.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
            self.optimizer.step()
            self.train_loss += L.item()
            self.cnt += 1
            self.confusion_matrix.count_task_separated(y_hat, y, 0)
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
            pred_true = self.net(X).detach()
            pred_fake = self.net(X_hat)
            L_task = torch.mean(self.loss(pred_fake, pred_true.softmax(dim=1)))
            (L_rec + L_task + self.args.beta * L_kl).backward()
            nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=20, norm_type=2)
            self.optimizerG.step()
            self.rec_loss += L_rec.item()
            self.kl_loss += L_kl.item()
            self.task_loss += L_task.item()
            self.cnt += 1

    def end_epoch(self, valid_dataset):
        if self.epoch < self.num_epochs_solver:
            super(EEGGRnetwork, self).end_epoch(valid_dataset)
        else:
            print(f'epoch: {self.epoch}, reconstruction loss: {self.rec_loss / self.cnt:.3f}, '
                  f'task loss: {self.task_loss / self.cnt:.3f}, kl loss: {self.kl_loss / self.cnt:.3f}, '
                  f"1000 lr: {self.optimizerG.state_dict()['param_groups'][0]['lr'] * 1000:.3f}")
            self.epoch += 1
            self.schedulerG.step()

    def end_task(self):
        super(EEGGRnetwork, self).end_task()
        generator_path = './modelsaved/generator_task' + str(self.task - 1) + '_fold' + str(self.flod_num) + '.pth'
        torch.save(self.generator.state_dict(), generator_path)
        self.generators.append(copy.deepcopy(self.generator.decoder))
        