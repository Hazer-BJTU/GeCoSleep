import torch.nn
from models import *
from . import generator
from clnetworks import CLnetwork


class EEGGRnetwork(CLnetwork):
    def __init__(self, args, fold_num):
        super(EEGGRnetwork, self).__init__(args, fold_num)
        '''generator settings'''
        self.start_training_generator = False
        self.num_epochs_solver = self.args.num_epochs - self.args.num_epochs_generator
        self.sample_gen = generator.SampleVAE(self.num_channels)
        self.seq_gen = generator.SequentialVAE(512, 0)
        self.sample_gen.apply(init_weight)
        self.seq_gen.apply(init_weight)
        self.optim_sample_gen = torch.optim.Adam(self.sample_gen.parameters(), lr=args.lr_generator)
        self.optim_seq_gen = torch.optim.Adam(self.seq_gen.parameters(), lr=args.lr_generator)
        self.sched_sample_gen = None
        self.sched_seq_gen = None
        self.rec_loss, self.kl_loss, self.task_loss = [0, 0], [0, 0], [0, 0]
        self.sample_gen.to(self.device)
        self.seq_gen.to(self.device)
        self.mseloss = nn.MSELoss()
        self.kldloss = nn.KLDivLoss(reduction='batchmean')
        '''replay settings'''
        self.teacher_model = SleepNet(self.num_channels, self.args.dropout)
        self.teacher_sample_gen = generator.SampleVAE(self.num_channels)
        self.teacher_seq_gen = generator.SequentialVAE(512, 0)
        self.teacher_model.to(self.device)
        self.teacher_sample_gen.to(self.device)
        self.teacher_seq_gen.to(self.device)
        self.sample_gen_memory = []
        self.seq_gen_memory = []

    def start_task(self):
        super(EEGGRnetwork, self).start_task()
        '''generator settings'''
        self.start_training_generator = False
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.num_epochs_solver // 6, 1), 0.6)
        self.optim_sample_gen = torch.optim.Adam(self.sample_gen.parameters(), lr=self.args.lr_generator)
        self.sched_sample_gen = torch.optim.lr_scheduler.StepLR(self.optim_sample_gen,
                                                                max(self.args.num_epochs_generator // 6, 1), 0.6)
        self.optim_seq_gen = torch.optim.Adam(self.seq_gen.parameters(), lr=self.args.lr_generator)
        self.sched_seq_gen = torch.optim.lr_scheduler.StepLR(self.optim_seq_gen,
                                                             max(self.args.num_epochs_generator // 6, 1), 0.6)
        '''replay settings'''
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1],
                                                          map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')
            self.teacher_sample_gen.load_state_dict(torch.load(self.sample_gen_memory[-1],
                                                               map_location=self.device, weights_only=True))
            self.teacher_seq_gen.load_state_dict(torch.load(self.seq_gen_memory[-1],
                                                            map_location=self.device, weights_only=True))
            print(f'generators loaded: {self.sample_gen_memory[-1]}, {self.seq_gen_memory[-1]}')

    def start_epoch(self):
        super(EEGGRnetwork, self).start_epoch()
        '''generator settings'''
        self.rec_loss, self.kl_loss, self.task_loss = [0, 0], [0, 0], [0, 0]
        self.sample_gen.train()
        self.seq_gen.train()
        self.teacher_model.eval()
        self.teacher_sample_gen.eval()
        self.teacher_seq_gen.eval()

    def observe(self, X, y, first_time=False):
        if self.epoch < self.num_epochs_solver:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.net(X)
            L_current = self.loss(y_hat, y.view(-1))
            L = torch.mean(L_current)
            if self.task > 0:
                '''perform generative replay'''
                X_fake = self.teacher_sample_gen.decoder.generate(y).detach()
                f_fake = self.teacher_model.features(X_fake).detach()
                F_fake = self.teacher_seq_gen.decoder.generate(y).detach()
                y_fake = self.teacher_model.classify(F_fake).detach()
                f_pred = self.net.features(X_fake)
                y_pred = self.net.classify(F_fake)
                L_replay_f = self.mseloss(f_pred, f_fake)
                L_replay_y = self.kldloss(nn.functional.log_softmax(y_pred, dim=1), y_fake.softmax(dim=1))
                L = L + L_replay_f + L_replay_y
            L.backward()
            self.optimizer.step()
            self.train_loss += L.item()
            self.confusion_matrix.count_task_separated(y_hat, y, 0)
        else:
            if not self.start_training_generator:
                print('start training generators...')
                self.net.load_state_dict(torch.load(self.best_net, map_location=self.device, weights_only=True))
                print(f'best solver model loaded: {self.best_net}')
                self.start_training_generator = True
            '''freeze BN and dropout'''
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Dropout):
                    m.eval()
            X, y = X.to(self.device), y.to(self.device)
            '''training sample generator'''
            self.optim_sample_gen.zero_grad()
            self.optimizer.zero_grad()
            if self.task > 0:
                '''perform generative replay for sample generator'''
                X_fake = self.teacher_sample_gen.decoder.generate(y).detach()
                X_prime = torch.cat((X, X_fake), dim=0)
                y_prime = torch.cat((y, y), dim=0)
            else:
                X_prime = X
                y_prime = y
            X_hat, L_kl = self.sample_gen(X_prime, y_prime)
            L_rec = self.mseloss(X_hat, X_prime)
            pred_true = self.net.features(X_prime).detach()
            pred_fake = self.net.features(X_hat)
            L_task = self.mseloss(pred_fake, pred_true)
            (L_rec + self.args.alpha * L_task + self.args.beta * L_kl).backward()
            self.optim_sample_gen.step()
            self.rec_loss[0] += L_rec.item()
            self.task_loss[0] += L_task.item()
            self.kl_loss[0] += L_kl.item()
            '''training sequential generator'''
            self.optim_seq_gen.zero_grad()
            self.optimizer.zero_grad()
            F = self.net.features(X).detach()
            if self.task > 0:
                '''perform generative replay for sequential generator'''
                F_fake = self.teacher_seq_gen.decoder.generate(y).detach()
                F_prime = torch.cat((F, F_fake), dim=0)
                y_prime = torch.cat((y, y), dim=0)
            else:
                F_prime = F
                y_prime = y
            F_hat, L_kl = self.seq_gen(F_prime, y_prime)
            L_rec = self.mseloss(F_hat, F_prime)
            pred_true = self.net.classify(F_prime).detach()
            pred_fake = self.net.classify(F_hat)
            L_task = self.kldloss(nn.functional.log_softmax(pred_fake, dim=1), pred_true.softmax(dim=1))
            (L_rec + self.args.alpha * L_task + self.args.beta * L_kl).backward()
            self.optim_seq_gen.step()
            self.rec_loss[1] += L_rec.item()
            self.task_loss[1] += L_task.item()
            self.kl_loss[1] += L_kl.item()
        self.cnt += 1

    def end_epoch(self, valid_dataset):
        if self.epoch < self.num_epochs_solver:
            super(EEGGRnetwork, self).end_epoch(valid_dataset)
        else:
            lr_sample_gen = self.optim_sample_gen.state_dict()['param_groups'][0]['lr']
            lr_seq_gen = self.optim_seq_gen.state_dict()['param_groups'][0]['lr']
            print(f'epoch: {self.epoch}, '
                  f'reconstruction loss: ({self.rec_loss[0] / self.cnt:.3f}, {self.rec_loss[1] / self.cnt:.3f}), '
                  f'task loss: ({self.task_loss[0] / self.cnt:.3f}, {self.task_loss[1] / self.cnt:.3f}), '
                  f'kl loss: ({self.kl_loss[0] / self.cnt:.3f}, {self.kl_loss[1] / self.cnt:.3f}) '
                  f"1000 lr: ({lr_sample_gen * 1000:.3f}, {lr_seq_gen * 1000:.3f})")
            self.epoch += 1
            self.sched_sample_gen.step()
            self.sched_seq_gen.step()

    def end_task(self):
        super(EEGGRnetwork, self).end_task()
        sample_gen_path = './modelsaved/sample_gen_task' + str(self.task - 1) + '_fold' + str(self.flod_num) + '.pth'
        torch.save(self.sample_gen.state_dict(), sample_gen_path)
        self.sample_gen_memory.append(sample_gen_path)
        seq_gen_path = './modelsaved/seq_gen_task' + str(self.task - 1) + '_fold' + str(self.flod_num) + '.pth'
        torch.save(self.seq_gen.state_dict(), seq_gen_path)
        self.seq_gen_memory.append(seq_gen_path)
