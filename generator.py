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
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(channels, affine=True), nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1),
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


class MultiScaleEncoder(nn.Module):
    def __init__(self, input_channels, kernel_size, stride, **kwargs):
        super(MultiScaleEncoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm1d(16, affine=True), nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=4, stride=4), ResBlock(32),
            nn.Conv1d(32, 128, kernel_size=4, stride=4), ResBlock(128),
            nn.Conv1d(128, 128, kernel_size=2, stride=2), ResBlock(128)
        )

    def forward(self, X):
        return self.block(X)


class MultiScaleDecoder(nn.Module):
    def __init__(self, output_channels, kernel_size, stride, window_size, **kwargs):
        super(MultiScaleDecoder, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.window_size = window_size
        self.block = nn.Sequential(
            ResBlock(128), nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2),
            ResBlock(128), nn.ConvTranspose1d(128, 32, kernel_size=4, stride=4),
            ResBlock(32), nn.ConvTranspose1d(32, 16, kernel_size=4, stride=4),
            ResBlock(16), nn.ConvTranspose1d(16, output_channels, kernel_size=kernel_size, stride=stride)
        )

    def forward(self, X):
        X = self.block(X)
        batch_size, num_channels, series = X.shape[0], X.shape[1], X.shape[2]
        delta = (series - self.window_size * 3000) // 2
        X = X[:, :, delta:3000 * self.window_size + delta]
        X = X.permute(0, 2, 1).contiguous()
        X = X.view(batch_size, self.window_size, 3000, num_channels)
        X = X.permute(0, 1, 3, 2).contiguous()
        return X


class EncoderSampler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderSampler, self).__init__(**kwargs)
        self.dist1 = Distribution(128, 128, 128)
        self.up1 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=2)
        self.dist2 = Distribution(256, 128, 128)
        self.up2 = nn.ConvTranspose1d(128, 128, kernel_size=5, stride=2)
        self.dist3 = Distribution(256, 128, 128)
        self.up3 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2)
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


class DecoderSampler(nn.Module):
    def __init__(self, **kwargs):
        super(DecoderSampler, self).__init__(**kwargs)
        self.up1 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=2)
        self.dist2 = Distribution(128, 128, 128)
        self.up2 = nn.ConvTranspose1d(128, 128, kernel_size=5, stride=2)
        self.dist3 = Distribution(128, 128, 128)
        self.up3 = nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2)
        self.dist4 = Distribution(128, 128, 128)

    def forward(self, z1):
        mu1, sigma1 = torch.zeros_like(z1, requires_grad=False), torch.ones_like(z1, requires_grad=False)
        mu2, sigma2 = self.dist2(self.up1(z1))
        eps = torch.randn_like(mu2, requires_grad=False)
        z2 = sigma2 * eps + mu2
        mu3, sigma3 = self.dist3(self.up2(z2))
        eps = torch.randn_like(mu3, requires_grad=False)
        z3 = sigma3 * eps + mu3
        mu4, sigma4 = self.dist4(self.up3(z3))
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


class Decoder(nn.Module):
    def __init__(self, output_channels, window_size, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.window_size = window_size
        self.sampler = DecoderSampler()
        self.dec1 = MultiScaleDecoder(output_channels, 400, 50, window_size)
        self.dec2 = MultiScaleDecoder(output_channels, 200, 25, window_size)
        self.dec3 = MultiScaleDecoder(output_channels, 100, 12, window_size)
        self.dec4 = MultiScaleDecoder(output_channels, 50, 6, window_size)

    def forward(self, z1):
        zs, distributions = self.sampler(z1)
        X1 = self.dec1(zs[0])
        X2 = self.dec2(zs[1])
        X3 = self.dec3(zs[2])
        X4 = self.dec4(zs[3])
        X = X1 + X2 + X3 + X4
        return X, distributions


def kl_gauss_gauss(mu1, sigma1, mu2, sigma2):
    kl_loss = 2 * torch.log(sigma2) - 2 * torch.log(sigma1) + (sigma1.pow(2) + (mu1 - mu2).pow(2)) / sigma2.pow(2) - 1
    return torch.mean(0.5 * kl_loss)


class EEGVAE(nn.Module):
    def __init__(self, input_channels, window_size, **kwargs):
        super(EEGVAE, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.windows_size = window_size
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(input_channels, window_size)

    def forward(self, X):
        batch_size, window_size, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.permute(0, 1, 3, 2).contiguous()
        X = X.view(batch_size, window_size * series, num_channels)
        X = X.permute(0, 2, 1).contiguous()
        zs, enc = self.encoder(X)
        X, dec = self.decoder(zs[0])
        kl_loss = 0
        for distenc, distdec in zip(enc, dec):
            kl_loss += kl_gauss_gauss(distenc[0], distenc[1], distdec[0], distdec[1])
        return X, kl_loss


if __name__ == '__main__':
    net = EEGVAE(2, 10)
    X = torch.randn((4, 10, 2, 3000), dtype=torch.float32, requires_grad=False)
    X_hat, kl_loss = net(X)
    print(X_hat.shape, kl_loss)
