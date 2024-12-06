import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from models import *
from data_preprocessing import *


class ResBlock(nn.Module):
    def __init__(self, channels, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.channels = channels
        self.block1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm1d(channels, affine=True), nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4),
            nn.InstanceNorm1d(channels, affine=True)
        )
        self.activate = nn.LeakyReLU(0.1)

    def forward(self, X):
        return self.activate(self.block1(X) + X)


class Distribution(nn.Module):
    def __init__(self, input_channels, hiddens, output_channels, **kwargs):
        super(Distribution, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.hiddens = hiddens
        self.output_channels = output_channels
        self.mean = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=1, stride=1),
            nn.InstanceNorm1d(hiddens, affine=True), nn.LeakyReLU(0.1),
            nn.Conv1d(hiddens, output_channels, kernel_size=1, stride=1)
        )
        self.std = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=1, stride=1),
            nn.InstanceNorm1d(hiddens, affine=True), nn.LeakyReLU(0.1),
            nn.Conv1d(hiddens, output_channels, kernel_size=1, stride=1),
            nn.Softplus()
        )

    def forward(self, X):
        return self.mean(X), self.std(X)


class UpSampler(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, **kwargs):
        super(UpSampler, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1),
            ResBlock(output_channels)
        )

    def forward(self, X):
        return self.block(X)


class DownSampler(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, **kwargs):
        super(DownSampler, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1),
            ResBlock(output_channels)
        )

    def forward(self, X):
        return self.block(X)


class MultiScaleEncoder(nn.Module):
    def __init__(self, input_channels, kernel_size, stride, **kwargs):
        super(MultiScaleEncoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm1d(16, affine=True), nn.LeakyReLU(0.1),
            DownSampler(16, 32, 4, 4),
            DownSampler(32, 128, 4, 4)
        )

    def forward(self, X):
        return self.block(X)


class EncoderSampler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderSampler, self).__init__(**kwargs)
        self.dist1 = Distribution(128, 128, 128)
        self.up1 = UpSampler(128, 128, kernel_size=2, stride=2)
        self.dist2 = Distribution(256, 128, 128)
        self.up2 = UpSampler(128, 128, kernel_size=9, stride=2)
        self.dist3 = Distribution(256, 128, 128)
        self.up3 = UpSampler(128, 128, kernel_size=4, stride=2)
        self.dist4 = Distribution(256, 128, 128)

    def forward(self, X1, X2, X3, X4):
        mu1, sigma1 = self.dist1(X1)
        eps = torch.randn_like(mu1, requires_grad=False)
        z1 = sigma1 * eps + mu1
        mu2, sigma2 = self.dist2(torch.cat((X2, self.up1(z1)), dim=1))
        eps = torch.randn_like(mu2, requires_grad=False)
        z2 = sigma2 * eps + mu2
        mu3, sigma3 = self.dist3(torch.cat((X3, self.up2(z2)), dim=1))
        eps = torch.randn_like(mu3, requires_grad=False)
        z3 = sigma3 * eps + mu3
        mu4, sigma4 = self.dist4(torch.cat((X4, self.up3(z3)), dim=1))
        eps = torch.randn_like(mu4, requires_grad=False)
        z4 = sigma4 * eps + mu4
        return [z1, z2, z3, z4], [(mu1, sigma1), (mu2, sigma2), (mu3, sigma3), (mu4, sigma4)]


class Encoder(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.enc1 = MultiScaleEncoder(input_channels, 400, 50)
        self.enc2 = MultiScaleEncoder(input_channels, 200, 25)
        self.enc3 = MultiScaleEncoder(input_channels, 100, 12)
        self.enc4 = MultiScaleEncoder(input_channels, 50, 6)
        self.sampler = EncoderSampler()

    def forward(self, X):
        X1 = self.enc1(X)
        X2 = self.enc2(X)
        X3 = self.enc3(X)
        X4 = self.enc4(X)
        zs, distributions = self.sampler(X1, X2, X3, X4)
        return zs, distributions


def kl_gauss_gauss(mu1, sigma1, mu2, sigma2):
    kl_loss = 2 * torch.log(sigma2) - 2 * torch.log(sigma1) + (sigma1.pow(2) + (mu1 - mu2).pow(2)) / sigma2.pow(2) - 1
    return torch.mean(0.5 * kl_loss)


if __name__ == '__main__':
    X = torch.randn((4, 2, 30000), dtype=torch.float32, requires_grad=False)
    net = Encoder(2)
    zs, distributions = net(X)
    for z in zs:
        print(z.shape)
    for mu, sigma in distributions:
        print(mu.shape, sigma.shape)
