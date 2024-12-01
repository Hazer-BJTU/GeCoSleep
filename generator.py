import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from models import *
from data_preprocessing import *


def conv_layer(input_channels, output_channels, kernel_size, stride, padding=None):
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(output_channels), nn.ReLU(inplace=True)
    )


class Attention(nn.Module):
    def __init__(self, input_features, hiddens, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.input_features = input_features
        self.hiddens = hiddens
        self.K = nn.Parameter(torch.randn((input_features, hiddens), dtype=torch.float32))
        self.norm1 = nn.LayerNorm(hiddens)
        self.Q = nn.Parameter(torch.randn((input_features, hiddens), dtype=torch.float32))
        self.norm2 = nn.LayerNorm(hiddens)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X, Y):
        batch_size, seq_length, features = X.shape[0], X.shape[1], X.shape[2]
        keys = self.norm1(Y @ self.K)
        queries = self.norm2(X @ self.Q)
        alpha = torch.bmm(queries, torch.transpose(keys, 1, 2)) / self.hiddens
        print(alpha)
        alpha = self.softmax(alpha)
        print(alpha)
        output = torch.bmm(alpha, Y)
        return output


if __name__ == '__main__':
    X = torch.randn(4, 5, 512)
    Y = torch.randn(4, 5, 512)
    net = Attention(512, 256)
    print(net(X, Y).shape)
