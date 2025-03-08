import torch
from models import *
from clnetworks import CLnetwork


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


class DTWnetwork(CLnetwork):
    def __init__(self, args, fold_num, logs):
        super(DTWnetwork, self).__init__(args, fold_num, logs)

    def start_task(self):
        super(DTWnetwork, self).start_task()

    def start_epoch(self):
        super(DTWnetwork, self).start_epoch()

    def observe(self, X, y, first_time=False):
        pass

    def end_epoch(self, valid_dataset):
        super(DTWnetwork, self).end_epoch(valid_dataset)

    def end_task(self, dataset=None):
        super(DTWnetwork, self).end_task(dataset)


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
