import torch.nn
from models import *
from . import generator
from baselines import MultiHeadSleepNet
from clnetworks import CLnetwork, linear_warmup_cosine_annealing
from metric import ConfusionMatrix, evaluate_tasks_multihead
from .HMM import HMMTaskGenerator


def knowledge_distillation_function(loss_type):
    kld_loss = nn.KLDivLoss(reduction='none')
    mse_loss = nn.MSELoss(reduction='none')

    def kl_divergence(y_hat, y, args):
        return torch.sum(kld_loss(nn.functional.log_softmax(y_hat / args.tau, dim=1), (y / args.tau).softmax(dim=1)), dim=1)

    def euclidean(y_hat, y, args):
        return torch.mean(mse_loss(y_hat, y), dim=1)

    def mixed(y_hat, y, args):
        L_kl = torch.sum(kld_loss(nn.functional.log_softmax(y_hat / args.tau, dim=1), (y / args.tau).softmax(dim=1)), dim=1)
        L_ed = torch.mean(mse_loss(y_hat, y), dim=1)
        return L_kl * (1 - args.mix_lambda) + L_ed * args.mix_lambda

    if loss_type == 'kl':
        return kl_divergence
    elif loss_type == 'ed':
        return euclidean
    else:
        return mixed


class EEGGRnetwork(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(EEGGRnetwork, self).__init__(args, fold_num, logs)
        self.net = MultiHeadSleepNet(self.num_channels, args.dropout, args.task_num, self.args.enable_multihead)
        self.net.apply(init_weight)
        self.net.to(self.device)
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
        self.kdloss = knowledge_distillation_function(self.args.distill_loss)
        '''replay settings'''
        self.teacher_model = MultiHeadSleepNet(self.num_channels, args.dropout, args.task_num, self.args.enable_multihead)
        self.teacher_seq_gen = generator.SequentialVAE(512, 0)
        self.teacher_model.to(self.device)
        self.teacher_seq_gen.to(self.device)
        self.seq_gen_memory = []
        '''statistics settings'''
        self.running_task_loss = None
        self.distill_loss, self.replay_loss = 0, 0
        '''HMM settings'''
        self.hmm_model = HMMTaskGenerator(args.task_num, self.device)

    def start_task(self):
        super(EEGGRnetwork, self).start_task()
        '''generator settings'''
        self.start_training_generator = False
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=linear_warmup_cosine_annealing(self.num_epochs_solver, self.args.min_epoch)
        )
        self.optim_seq_gen = torch.optim.Adam(self.seq_gen.parameters(), lr=self.args.lr_seq_gen)
        self.sched_seq_gen = torch.optim.lr_scheduler.LambdaLR(
            self.optim_seq_gen,
            lr_lambda=linear_warmup_cosine_annealing(self.args.num_epochs_generator, self.args.min_epoch)
        )
        '''replay settings'''
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')
            self.teacher_seq_gen.load_state_dict(torch.load(self.seq_gen_memory[-1], map_location=self.device, weights_only=True))
            print(f'generator loaded: {self.seq_gen_memory[-1]}')
        '''statistics settings'''
        if self.task > 0:
            self.running_task_loss = torch.ones(self.task, dtype=torch.float32, requires_grad=False, device=self.device)
            self.running_task_loss = self.running_task_loss / torch.norm(self.running_task_loss, p=2)
        '''HMM settings'''
        self.hmm_model.clear_samples()

    def start_epoch(self):
        super(EEGGRnetwork, self).start_epoch()
        '''generator settings'''
        self.rec_loss, self.kl_loss, self.task_loss = 0, 0, 0
        self.seq_gen.train()
        self.teacher_model.eval()
        self.teacher_seq_gen.eval()
        '''statistics settings'''
        self.distill_loss, self.replay_loss = 0, 0

    def update_running_task_loss(self, L, t, batch_size):
        delta = torch.zeros_like(self.running_task_loss)
        cnt = torch.zeros_like(self.running_task_loss)
        for i in range(batch_size):
            delta[t[i]] += L[i].item()
            cnt[t[i]] += 1
        delta = delta / torch.clamp(cnt, min=1.0)
        delta = delta / (torch.norm(delta, p=2) + 1e-8)
        for i in range(self.running_task_loss.shape[0]):
            if cnt[i] == 0:
                continue
            self.running_task_loss[i] = (1 - self.args.gamma) * self.running_task_loss[i] + self.args.gamma * delta[i]
        self.running_task_loss = self.running_task_loss / (torch.norm(self.running_task_loss, p=2) + 1e-8)

    def observe(self, X, y, first_time=False):
        if first_time:
            '''update hmm model samples'''
            self.hmm_model.add_sample(y)
        if self.epoch < self.num_epochs_solver:
            X, y = X.to(self.device), y.to(self.device)
            if self.task > 0:
                self.net.freeze_parameters()
            self.optimizer.zero_grad()
            y_hat = self.net(X, self.task)
            L_current = self.loss(y_hat, y.view(-1))
            L = torch.mean(L_current)
            self.train_loss += L.item()
            if self.task > 0:
                '''perform generative replay'''
                weights = self.running_task_loss.softmax(dim=0)
                t = torch.multinomial(weights, y.shape[0], replacement=True)
                y_hmm = self.hmm_model.generate(t)
                F_fake = self.teacher_seq_gen.decoder.generate(y_hmm, t).detach()
                y_fake = self.teacher_model.classify(F_fake, self.task - 1).detach()
                y_pred = self.net.classify(F_fake, self.task)
                L_replay = self.kdloss(y_pred, y_fake, self.args)
                L = L + torch.mean(L_replay) * self.args.replay_lambda
                self.replay_loss += torch.mean(L_replay).item() * self.args.replay_lambda
                '''distillation for sample feature extractor'''
                y_distill = self.teacher_model(X, self.task - 1).detach()
                L_distill = self.kdloss(y_hat, y_distill, self.args)
                L = L + torch.mean(L_distill) * self.args.distill_lambda
                self.distill_loss += torch.mean(L_distill).item() * self.args.distill_lambda
                '''update running task loss'''
                self.update_running_task_loss(L_replay, t, y.shape[0])
            L.backward()
            self.optimizer.step()
            self.confusion_matrix.count_task_separated(y_hat, y, 0)
        else:
            if not self.start_training_generator:
                if self.task + 1 == self.args.task_num:
                    print('skip last training...')
                    return
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
                y_hmm = self.hmm_model.generate(t)
                F_fake = self.teacher_seq_gen.decoder.generate(y_hmm, t).detach()
                F_prime = torch.cat((F_fake, F), dim=0)
                y_prime = torch.cat((y_hmm, y), dim=0)
                temp = torch.ones(y.shape[0], dtype=torch.int64, requires_grad=False, device=self.device) * self.task
                t_prime = torch.cat((t, temp), dim=0)
            else:
                F_prime = F
                y_prime = y
                t_prime = torch.ones(y.shape[0], dtype=torch.int64, requires_grad=False, device=self.device) * self.task
            F_hat, L_kl = self.seq_gen(F_prime, y_prime, t_prime)
            L_rec = self.mseloss(F_hat, F_prime)
            pred_true = self.net.classify(F_prime, self.task).detach()
            pred_fake = self.net.classify(F_hat, self.task)
            L_task = self.kdloss(pred_fake, pred_true, self.args)
            if self.task > 0:
                '''update running task loss'''
                self.update_running_task_loss(L_task.detach(), t_prime, y.shape[0])
            '''backprop'''
            L_task = torch.mean(L_task)
            (L_rec + L_task + self.args.beta * L_kl).backward()
            self.optim_seq_gen.step()
            self.rec_loss += L_rec.item()
            self.task_loss += L_task.item()
            self.kl_loss += L_kl.item() * self.args.beta
        self.cnt += 1

    def end_epoch(self, valid_dataset):
        if self.epoch < self.num_epochs_solver:
            learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
            train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
            print(
                f'epoch: {self.epoch}, '
                f'train loss: {self.train_loss / self.cnt:.3f}, '
                f'distill loss: {self.distill_loss / self.cnt:.3f}, '
                f'replay loss: {self.replay_loss / self.cnt:.3f}, '
                f'train accuracy: {train_acc:.3f}, '
                f"macro F1: {train_mf1:.3f}, 1000 lr: {learning_rate * 1000:.3f}")
            self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'epoch:{self.epoch}'], {
                'train loss': self.train_loss / self.cnt,
                'distill loss': self.distill_loss / self.cnt,
                'replay loss': self.replay_loss / self.cnt,
                'train accuracy': train_acc,
                'train mF1': train_mf1,
                '1000 lr': learning_rate * 1000
            })
            if (self.epoch + 1) % self.args.valid_epoch == 0:
                print(f'validating on the datasets...')
                valid_confusion = ConfusionMatrix(1)
                valid_confusion = evaluate_tasks_multihead(self.net, [valid_dataset], valid_confusion,
                                                           self.device, self.args.valid_batch, self.task)
                valid_acc, valid_mf1 = valid_confusion.accuracy(), valid_confusion.macro_f1()
                print(f'valid accuracy: {valid_acc:.3f}, valid macro F1: {valid_mf1:.3f}')
                self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'valid epoch:{self.epoch}'], {
                    'valid accuracy': valid_acc,
                    'valid mF1': valid_mf1
                })
                if valid_acc + valid_mf1 > self.best_valid_acc and self.epoch + 1 >= self.args.min_epoch:
                    self.logs.append(
                        ['train_info', f'task{self.task}_fold{self.fold_num}', f'valid epoch:{self.epoch}', 'saved'],
                        True)
                    self.best_train_loss = self.train_loss / self.cnt
                    self.best_train_acc = train_acc
                    self.best_valid_acc = valid_acc + valid_mf1
                    self.best_net = './modelsaved/' + str(self.args.replay_mode) + '_task' + str(
                        self.task) + '_fold' + str(self.fold_num) + '.pth'
                    print(f'model saved: {self.best_net}')
                    torch.save(self.net.state_dict(), self.best_net)
            self.epoch += 1
            self.scheduler.step()
        else:
            if self.task + 1 == self.args.task_num:
                return
            lr_seq_gen = self.optim_seq_gen.state_dict()['param_groups'][0]['lr']
            print(f'epoch: {self.epoch}, '
                  f'reconstruction loss: {self.rec_loss / self.cnt:.3f}, '
                  f'task loss: {self.task_loss / self.cnt:.3f}, '
                  f'kl loss: {self.kl_loss / self.cnt:.3f}, '
                  f"1000 lr: {lr_seq_gen * 1000:.3f}")
            self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'epoch:{self.epoch}'], {
                'train generator': True,
                'reconstruction loss': self.rec_loss / self.cnt,
                'task loss': self.task_loss / self.cnt,
                'kl loss': self.kl_loss / self.cnt,
                '1000 lr': lr_seq_gen * 1000
            })
            self.epoch += 1
            self.sched_seq_gen.step()
        if self.task > 0:
            self.logs.append(
                ['train_info', f'task{self.task}_fold{self.fold_num}', f'epoch:{self.epoch - 1}', 'Pk'],
                self.running_task_loss.cpu().numpy().tolist()
            )

    def end_task(self, dataset=None):
        super(EEGGRnetwork, self).end_task()
        seq_gen_path = './modelsaved/seq_gen_task' + str(self.task - 1) + '_fold' + str(self.fold_num) + '.pth'
        torch.save(self.seq_gen.state_dict(), seq_gen_path)
        self.seq_gen_memory.append(seq_gen_path)
        '''hmm settings'''
        print(f'start fitting hmm model on {self.hmm_model.total_samples} samples...')
        self.hmm_model.ready()
        print('hmm model fitted.')


if __name__ == '__main__':
    pass
