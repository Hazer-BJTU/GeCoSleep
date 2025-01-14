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
        self.generator = generator.VAE(512, 0)
        self.generator.apply(init_weight)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=args.lr_generator)
        self.schedulerG = None
        self.rec_loss, self.kl_loss, self.task_loss = 0, 0, 0
        self.generator.to(self.device)
        self.mseloss = nn.MSELoss()
        self.distloss = nn.KLDivLoss(reduction='batchmean')
        '''replay settings'''
        self.teacher_model = SleepNet(2, self.args.dropout)
        self.teacher_generator = generator.VAE(512, 0)
        self.teacher_model.to(self.device)
        self.teacher_generator.to(self.device)
        self.best_generator_memory = []

    def start_task(self):
        super(EEGGRnetwork, self).start_task()
        '''generator settings'''
        self.start_training_generator = False
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.num_epochs_solver // 6, 1), 0.6)
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_generator)
        self.schedulerG = torch.optim.lr_scheduler.StepLR(self.optimizerG, max(self.args.num_epochs_generator // 6, 1),0.6)
        '''replay settings'''
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1],
                                                          map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')
            self.teacher_generator.load_state_dict(torch.load(self.best_generator_memory[-1],
                                                              map_location=self.device, weights_only=True))
            print(f'generator loaded: {self.best_generator_memory[-1]}')

    def start_epoch(self):
        super(EEGGRnetwork, self).start_epoch()
        '''generator settings'''
        self.rec_loss, self.kl_loss, self.task_loss = 0, 0, 0
        self.generator.train()
        self.teacher_model.eval()
        self.teacher_generator.eval()

    def observe(self, X, y, first_time=False):
        if self.epoch < self.num_epochs_solver:
            if self.task > 0:
                '''freeze the feature extraction module'''
                self.net.freeze_parameters()
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.net(X)
            L_current = self.loss(y_hat, y.view(-1))
            L = torch.mean(L_current)
            if self.task > 0:
                '''perform generative replay'''
                F_fake = self.teacher_generator.decoder.generate(y).detach()
                y_fake = self.teacher_model.classify(F_fake).detach()
                y_pred = self.net.classify(F_fake)
                y_pred = nn.functional.log_softmax(y_pred, dim=1)
                L_replay = self.distloss(y_pred, y_fake.softmax(dim=1))
                L = L + L_replay
            L.backward()
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
            F = self.net.features(X).detach()
            if self.task > 0:
                '''perform generative replay for generator'''
                F_fake = self.teacher_generator.decoder.generate(y).detach()
                F = torch.cat((F, F_fake), dim=0)
                y_prime = torch.cat((y, y), dim=0)
            else:
                y_prime = y
            F_hat, L_kl = self.generator(F, y_prime)
            L_rec = self.mseloss(F_hat, F)
            pred_true = self.net.classify(F).detach()
            pred_fake = self.net.classify(F_hat)
            pred_fake = nn.functional.log_softmax(pred_fake, dim=1)
            L_task = self.distloss(pred_fake, pred_true.softmax(dim=1))
            (L_rec + self.args.alpha * L_task + self.args.beta * L_kl).backward()
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
        self.best_generator_memory.append(generator_path)
