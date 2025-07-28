import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, number, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.k = nn.Parameter(torch.ones(number, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, p):
        Q = torch.dot(self.sigmoid(self.k), p)
        return torch.min(self.sigmoid(self.k) * p / Q)

    def getk(self):
        return self.sigmoid(self.k)


if __name__ == '__main__':
    num = torch.tensor([114, 25, 43, 15, 5], dtype=torch.float32)
    total = torch.sum(num)
    p = num / total
    net = Model(5)
    num_epoch, lr = 100, 1
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        L = (1 - net(p))
        print(L.item())
        L.backward()
        optimizer.step()
    print(net.getk())
