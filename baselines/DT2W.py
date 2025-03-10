import torch
import random
from models import *
from clnetworks import CLnetwork
from metric import ConfusionMatrix, evaluate_tasks
from torch.utils.data import DataLoader


def batched_soft_dtw_loss_4_short_seq(seriesX, seriesY, gamma=0.1, eps=1e-5):
    N, M, features = seriesX.shape[1], seriesY.shape[1], seriesX.shape[2]
    '''the input series should be organized in [batch_size, seq_length, features]'''
    dist = torch.cdist(seriesX, seriesY, p=2)
    '''the size of the distance matrix is [batch_size, N, M]'''
    dist = dist.permute(1, 2, 0)
    '''transpose the distance matrix into [N, M, batch_size]'''
    R = dist.clone()
    for i in range(1, N):
        R[i, 0] = R[i - 1, 0] + dist[i, 0]
    for j in range(1, M):
        R[0, j] = R[0, j - 1] + dist[0, j]
    for i in range(1, N):
        for j in range(1, M):
            c = -1 / gamma
            delta = torch.exp(R[i - 1, j] * c) + torch.exp(R[i, j - 1] * c) + torch.exp(R[i - 1, j - 1] * c)
            R[i, j] = dist[i, j] - gamma * torch.log(torch.clamp(delta, min=eps))
    return R[-1, -1, :]


class DTWSleepNet(SleepNet):
    def __init__(self, input_channels, dropout, **kwargs):
        super(DTWSleepNet, self).__init__(input_channels, dropout, **kwargs)

    def forward(self, X, with_features=False):
        batch_size, seq_length, num_channels, series = X.shape
        X = self.cnn(X)
        X = self.short_term_encoder(X)
        X = X.view(batch_size, seq_length, -1)
        r = self.resblock(X.view(batch_size * seq_length, -1))
        F1 = self.long_term_encoder(X)
        F2 = torch.cat((r, F1), dim=1)
        output = self.classifier(F2)
        if with_features:
            return output, F1.view(batch_size, seq_length, -1), F2
        else:
            return output

    def predict(self, F):
        F = self.classifier(F)
        return F


class DTWnetwork(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(DTWnetwork, self).__init__(args, fold_num, logs)
        self.net = DTWSleepNet(self.num_channels, args.dropout)
        self.net.apply(init_weight)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.net.to(self.device)
        self.kldloss = nn.KLDivLoss(reduction='none')
        '''distill settings'''
        self.teacher_model = DTWSleepNet(self.num_channels, args.dropout)
        self.teacher_model.to(self.device)
        self.dtw_loss, self.distill_loss = 0, 0
        '''prototype settings'''
        self.task_prototypes = []
        self.radius = None
        self.prot_loss = 0

    def start_task(self):
        super(DTWnetwork, self).start_task()
        '''load teacher models'''
        if self.task > 0:
            self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
            print(f'teacher model loaded: {self.best_net_memory[-1]}')

    def start_epoch(self):
        super(DTWnetwork, self).start_epoch()
        self.dtw_loss, self.distill_loss, self.prot_loss = 0, 0, 0
        self.teacher_model.eval()

    def sample_prototypes(self):
        t = random.randint(0, self.task - 1)
        mu = self.task_prototypes[t]
        mini_batch = torch.randint(0, 5, [self.args.batch_size * self.args.window_size],
                                   dtype=torch.int64, requires_grad=False, device=self.device)
        mus = mu[mini_batch]
        eps = torch.randn_like(mus)
        F_prot = mus + self.radius[mini_batch] * eps
        y_prot = torch.tensor(mini_batch, dtype=torch.int64, requires_grad=False, device=self.device)
        return F_prot, y_prot

    def observe(self, X, y, first_time=False):
        X, y = X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_hat, F1, F2 = self.net(X, True)
        L_current = self.loss(y_hat, y.view(-1))
        L = torch.mean(L_current)
        self.train_loss += L.item()
        if self.task > 0:
            '''perform knowledge distillation'''
            y_distill, F_distill, _ = self.teacher_model(X, True)
            y_distill, F_distill = y_distill.detach() / self.args.tau, F_distill.detach()
            L_distill = torch.sum(self.kldloss(nn.functional.log_softmax(y_hat / self.args.tau, dim=1), y_distill.softmax(dim=1)), dim=1)
            L_dtw = batched_soft_dtw_loss_4_short_seq(F1, F_distill)
            L = L + torch.mean(L_distill) + torch.mean(L_dtw) * self.args.dtw_lambda
            self.distill_loss += torch.mean(L_distill).item()
            self.dtw_loss += torch.mean(L_dtw).item() * self.args.dtw_lambda
            '''prototype loss'''
            F_prot, y_prot = self.sample_prototypes()
            y_prot_hat = self.net.predict(F_prot)
            L_prot = self.loss(y_prot_hat, y_prot)
            L = L + torch.mean(L_prot)
            self.prot_loss += torch.mean(L_prot).item()
        L.backward()
        self.optimizer.step()
        self.cnt += 1
        self.confusion_matrix.count_task_separated(y_hat, y, 0)

    def end_epoch(self, valid_dataset):
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr']
        train_acc, train_mf1 = self.confusion_matrix.accuracy(), self.confusion_matrix.macro_f1()
        print(f'epoch: {self.epoch}, '
              f'train loss: {self.train_loss / self.cnt:.3f}, '
              f'distill loss: {self.distill_loss / self.cnt:.3f}, '
              f'dtw loss: {self.dtw_loss / self.cnt:.3f}, '
              f'prot loss: {self.prot_loss / self.cnt:.3f}, '
              f'train accuracy: {train_acc:.3f}, '
              f"macro F1: {train_mf1:.3f}, 1000 lr: {learning_rate * 1000:.3f}")
        self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'epoch:{self.epoch}'], {
            'train loss': self.train_loss / self.cnt,
            'distill loss': self.distill_loss / self.cnt,
            'dtw loss': self.dtw_loss / self.cnt,
            'prot loss': self.prot_loss / self.cnt,
            'train accuracy': train_acc,
            'train mF1': train_mf1,
            '1000 lr': learning_rate * 1000
        })
        if (self.epoch + 1) % self.args.valid_epoch == 0:
            print(f'validating on the datasets...')
            valid_confusion = ConfusionMatrix(1)
            valid_confusion = evaluate_tasks(self.net, [valid_dataset], valid_confusion, self.device, self.args.valid_batch)
            valid_acc, valid_mf1 = valid_confusion.accuracy(), valid_confusion.macro_f1()
            print(f'valid accuracy: {valid_acc:.3f}, valid macro F1: {valid_mf1:.3f}')
            self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'valid epoch:{self.epoch}'], {
                'valid accuracy': valid_acc,
                'valid mF1': valid_mf1
            })
            if valid_acc + valid_mf1 > self.best_valid_acc and self.epoch + 1 >= self.args.min_epoch:
                self.logs.append(['train_info', f'task{self.task}_fold{self.fold_num}', f'valid epoch:{self.epoch}', 'saved'], True)
                self.best_train_loss = self.train_loss / self.cnt
                self.best_train_acc = train_acc
                self.best_valid_acc = valid_acc + valid_mf1
                self.best_net = './modelsaved/' + str(self.args.replay_mode) + '_task' + str(self.task) + '_fold' + str(self.fold_num) + '.pth'
                print(f'model saved: {self.best_net}')
                torch.save(self.net.state_dict(), self.best_net)
        self.epoch += 1
        self.scheduler.step()

    def update_prototypes(self, dataset):
        self.teacher_model.load_state_dict(torch.load(self.best_net_memory[-1], map_location=self.device, weights_only=True))
        print(f'best model loaded: {self.best_net_memory[-1]}')
        loader = DataLoader(dataset, self.args.batch_size, True)
        self.teacher_model.eval()
        num_samples = 0
        mu = torch.zeros((5, 1024), dtype=torch.float32, requires_grad=False, device=self.device)
        mu2 = torch.zeros_like(mu)
        print('start calculating task prototypes')
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.view(-1).to(self.device)
                _, _, F2 = self.teacher_model(X, True)
                for idx in range(F2.shape[0]):
                    mu[y[idx]] += F2[idx].data
                    mu2[y[idx]] += F2[idx].data.pow(2)
                num_samples += F2.shape[0]
        if self.task == 1:
            self.radius = (mu2 / num_samples - (mu / num_samples).pow(2)).pow(0.5)
            print('radius updated')
        self.task_prototypes.append(mu / num_samples)
        print('task prototpyes updated')

    def end_task(self, dataset=None):
        super(DTWnetwork, self).end_task(dataset)
        self.update_prototypes(dataset)


if __name__ == '__main__':
    '''
    net = Net()
    Y = torch.randn(1, 10, 512)
    optim = torch.optim.Adam(net.parameters(), lr=1)
    net.to(torch.device(f'cuda:{0}'))
    Y = Y.to(torch.device(f'cuda:{0}'))
    print(torch.nn.functional.mse_loss(net.X, Y))
    for epoch in range(20):
        optim.zero_grad()
        loss = batched_soft_dtw_loss_4_short_seq(net.X, Y)
        loss.backward()
        print(loss.item())
        optim.step()
    print(torch.nn.functional.mse_loss(net.X, Y))
    '''
    X = torch.randn(32, 10, 512)
    Y = torch.randn(32, 10, 512)
    print(batched_soft_dtw_loss_4_short_seq(X, Y))
