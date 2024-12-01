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


class Distribute(nn.Module):
    def __init__(self, input_channels, output_channels, hiddens, **kwargs):
        super(Distribute, self).__init__(**kwargs)
        self.mean = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=1, stride=1), nn.ReLU(),
            nn.Conv1d(hiddens, output_channels, kernel_size=1, stride=1)
        )
        self.std = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=1, stride=1), nn.ReLU(),
            nn.Conv1d(hiddens, output_channels, kernel_size=1, stride=1), nn.Softplus()
        )

    def forward(self, X):
        return self.mean(X), self.std(X)


class DownSampleUnit(nn.Module):
    def __init__(self, input_channels, output_channels, hiddens, **kwargs):
        super(DownSampleUnit, self).__init__(**kwargs)
        self.conv1 = conv_layer(input_channels, output_channels, 25, 1)
        self.conv2 = conv_layer(output_channels, output_channels, 25, 1)
        self.pooling = nn.MaxPool1d(kernel_size=5, stride=5)
        self.dist = Distribute(output_channels, output_channels // 2, hiddens)

    def forward(self, X):
        X = self.conv2(self.conv1(X))
        mu, sigma = self.dist(X)
        X_down = self.pooling(X)
        return X_down, (mu, sigma)


class Encoder(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.unit1 = DownSampleUnit(input_channels, 16, 32)
        self.unit2 = DownSampleUnit(16, 32, 64)
        self.unit3 = DownSampleUnit(32, 64, 128)
        self.unit4 = DownSampleUnit(64, 128, 256)
        self.unit5 = Distribute(128, 128, 256)

    def forward(self, X):
        X1, dist1 = self.unit1(X)
        X2, dist2 = self.unit2(X1)
        X3, dist3 = self.unit3(X2)
        X4, dist4 = self.unit4(X3)
        dist5 = self.unit5(X4)
        return [dist1, dist2, dist3, dist4, dist5]


class UpSampleUnit(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super(UpSampleUnit, self).__init__(**kwargs)
        self.conv1 = conv_layer(input_channels, output_channels, 25, 1)
        self.conv2 = conv_layer(output_channels, output_channels, 25, 1)
        self.up = nn.ConvTranspose1d(output_channels, output_channels, kernel_size=5, stride=5)

    def forward(self, X, skip):
        X = self.conv2(self.conv1(X))
        X_up = self.up(X)
        X_up = torch.cat((skip, X_up), dim=1)
        return X_up


class Decoder(nn.Module):
    def __init__(self, output_channels, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.unit1 = UpSampleUnit(128, 64)
        self.unit2 = UpSampleUnit(128, 32)
        self.unit3 = UpSampleUnit(64, 16)
        self.unit4 = UpSampleUnit(32, 8)
        self.unit5 = nn.Sequential(
            nn.Conv1d(16, 16, 25, 1, 12),
            nn.Conv1d(16, 2, 25, 1, 12)
        )

    def forward(self, X):
        output = self.unit1(X[4], X[3])
        output = self.unit2(output, X[2])
        output = self.unit3(output, X[1])
        output = self.unit4(output, X[0])
        output = self.unit5(output)
        print(output.shape)
        batch_size, num_channels, features = output.shape[0], output.shape[1], output.shape[2]
        output = output.permute(0, 2, 1).contiguous()
        output = output.view(batch_size, -1, 3000, num_channels)
        output = output.permute(0, 1, 3, 2).contiguous()
        return output


def kl_loss(mu, sigma):
    loss = -2 * torch.log(sigma) + sigma.pow(2) + mu.pow(2) - 1
    return 0.5 * loss


class VAE(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(input_channels)

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.permute(0, 1, 3, 2).contiguous()
        X = X.view(batch_size, seq_length * series, num_channels)
        X = X.permute(0, 2, 1).contiguous()
        samples, KL_loss = [], 0
        distributions = self.encoder(X)
        for mu, sigma in distributions:
            eps = torch.randn_like(sigma, dtype=torch.float32, requires_grad=False)
            z = eps * sigma + mu
            KL_loss += torch.mean(kl_loss(mu, sigma))
            samples.append(z)
        X_hat = self.decoder(samples)
        return X_hat, KL_loss


if __name__ == '__main__':
    net = VAE(2)
    X = torch.randn((8, 5, 2, 3000))
    X_hat, kl_loss = net(X)
    print(X_hat.shape, kl_loss)
