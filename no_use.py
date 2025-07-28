import torch
import torch.nn as nn

from BayesEEGNet import *


if __name__ == '__main__':
    net = BayesEEGNet(512, 5, 512, 2, 64)
    X = torch.randn((320, 2, 3000), device=torch.device('cuda:0'))
    net.to(torch.device('cuda:0'))
    y = net(X)
    print(y)
