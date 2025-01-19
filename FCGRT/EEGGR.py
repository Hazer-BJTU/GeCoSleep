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
        self.seq_gen = generator.SequentialVAE(512, 0)
        self.seq_gen.apply(init_weight)
        self.optim_seq_gen = torch.optim.Adam(self.seq_gen.parameters(), lr=args.lr_seq_gen)
        self.sched_seq_gen = None
        self.rec_loss, self.kl_loss, self.task_loss = 0, 0, 0
        self.seq_gen.to(self.device)
        self.mseloss = nn.MSELoss()
        self.kldloss = nn.KLDivLoss(reduction='none')
        '''replay settings'''
        self.teacher_model = SleepNet(self.num_channels, self.args.dropout)
        self.teacher_seq_gen = generator.SequentialVAE(512, 0)
        self.teacher_model.to(self.device)
        self.teacher_seq_gen.to(self.device)
        self.seq_gen_memory = []
        '''statistics settings'''
        self.running_task_loss = None

    def start_task(self):
        super(EEGGRnetwork, self).start_task()
        '''generator settings'''
        self.start_training_generator = False
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, max(self.num_epochs_solver // 6, 1), 0.6)
        self.optim_seq_gen = torch.optim.Adam(self.seq_gen.parameters(), lr=self.args.lr_seq_gen)
        self.sched_seq_gen = torch.optim.lr_scheduler.StepLR(self.optim_seq_gen, max(self.args.num_epochs_generator // 6, 1), 0.6)
        '''replay settings'''
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')
            self.teacher_seq_gen.load_state_dict(torch.load(self.seq_gen_memory[-1], map_location=self.device, weights_only=True))
            print(f'generator loaded: {self.seq_gen_memory[-1]}')
        '''statistics settings'''
        if self.task > 0:
            self.running_task_loss = torch.zeros(self.task, dtype=torch.float32, requires_grad=False, device=self.device)

    def start_epoch(self):
        super(EEGGRnetwork, self).start_epoch()
        '''generator settings'''
        self.rec_loss, self.kl_loss, self.task_loss = 0, 0, 0
        self.seq_gen.train()
        self.teacher_model.eval()
        self.teacher_seq_gen.eval()

    def observe(self, X, y, first_time=False):
        if self.epoch < self.num_epochs_solver:
            X, y = X.to(self.device), y.to(self.device)
            if self.task > 0:
                '''freeze feature extractor'''
                self.net.freeze_parameters()
            self.optimizer.zero_grad()
            y_hat = self.net(X)
            L_current = self.loss(y_hat, y.view(-1))
            L = torch.mean(L_current)
            if self.task > 0:
                '''perform generative replay'''
                weights = self.running_task_loss.softmax(dim=0)
                t = torch.multinomial(weights, y.shape[0], replacement=True)
                F_fake = self.teacher_seq_gen.decoder.generate(y, t).detach()
                y_fake = self.teacher_model.classify(F_fake).detach()
                y_pred = self.net.classify(F_fake)
                L_replay = torch.sum(self.kldloss(nn.functional.log_softmax(y_pred, dim=1), y_fake.softmax(dim=1)), dim=1)
                L = L + torch.mean(L_replay)
                '''update running task loss'''
                delta = torch.zeros_like(self.running_task_loss)
                cnt = torch.zeros_like(self.running_task_loss)
                for i in range(y.shape[0]):
                    delta[t[i]] += L_replay[i].item()
                    cnt[t[i]] += 1
                delta = delta / torch.clamp(cnt, min=1.0)
                self.running_task_loss = (1 - self.args.gamma) * self.running_task_loss + self.args.gamma * delta
            L.backward()
            self.optimizer.step()
            self.train_loss += L.item()
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
            '''training sequential generator'''
            self.optim_seq_gen.zero_grad()
            self.optimizer.zero_grad()
            F = self.net.features(X).detach()
            if self.task > 0:
                '''perform generative replay for sequential generator'''
                weights = self.running_task_loss.softmax(dim=0)
                t = torch.multinomial(weights, y.shape[0], replacement=True)
                F_fake = self.teacher_seq_gen.decoder.generate(y, t).detach()
                F_prime = torch.cat((F, F_fake), dim=0)
                y_prime = torch.cat((y, y), dim=0)
                temp = torch.ones(y.shape[0], dtype=torch.int64, requires_grad=False, device=self.device) * self.task
                t_prime = torch.cat((temp, t), dim=0)
            else:
                F_prime = F
                y_prime = y
                t_prime = torch.ones(y.shape[0], dtype=torch.int64, requires_grad=False, device=self.device) * self.task
            F_hat, L_kl = self.seq_gen(F_prime, y_prime, t_prime)
            L_rec = self.mseloss(F_hat, F_prime)
            pred_true = self.net.classify(F_prime).detach()
            pred_fake = self.net.classify(F_hat)
            L_task = torch.sum(self.kldloss(nn.functional.log_softmax(pred_fake, dim=1), pred_true.softmax(dim=1)), dim=1)
            L_task = torch.mean(L_task)
            (L_rec + self.args.alpha * L_task + self.args.beta * L_kl).backward()
            self.optim_seq_gen.step()
            self.rec_loss += L_rec.item()
            self.task_loss += L_task.item()
            self.kl_loss += L_kl.item()
        self.cnt += 1

    def end_epoch(self, valid_dataset):
        if self.epoch < self.num_epochs_solver:
            super(EEGGRnetwork, self).end_epoch(valid_dataset)
            if self.task > 0:
                print(f'task replay distribution: {torch.round(self.running_task_loss.softmax(dim=0), decimals=2).data}')
        else:
            lr_seq_gen = self.optim_seq_gen.state_dict()['param_groups'][0]['lr']
            print(f'epoch: {self.epoch}, '
                  f'reconstruction loss: {self.rec_loss / self.cnt:.3f}, '
                  f'task loss: {self.task_loss / self.cnt:.3f}, '
                  f'kl loss: {self.kl_loss / self.cnt:.3f}, '
                  f"1000 lr: {lr_seq_gen * 1000:.3f}")
            self.epoch += 1
            self.sched_seq_gen.step()

    def end_task(self):
        super(EEGGRnetwork, self).end_task()
        seq_gen_path = './modelsaved/seq_gen_task' + str(self.task - 1) + '_fold' + str(self.flod_num) + '.pth'
        torch.save(self.seq_gen.state_dict(), seq_gen_path)
        self.seq_gen_memory.append(seq_gen_path)
