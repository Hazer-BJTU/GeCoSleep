import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from models import *
from data_preprocessing import *


class ResBlock(nn.Module):
    def __init__(self, channels, norm_type, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.channels = channels
        self.norm_type = norm_type
        self.block = None
        if norm_type == 'instance':
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4),
                nn.InstanceNorm1d(channels, affine=True), nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4),
                nn.InstanceNorm1d(channels, affine=True)
            )
        elif norm_type == 'batch':
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4),
                nn.BatchNorm1d(channels), nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding=4),
                nn.BatchNorm1d(channels)
            )
        self.activate = nn.LeakyReLU(0.1)

    def forward(self, X):
        return self.activate(self.block(X) + X)


class Distribution(nn.Module):
    def __init__(self, input_channels, hiddens, output_channels, **kwargs):
        super(Distribution, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.hiddens = hiddens
        self.output_channels = output_channels
        self.mean = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(hiddens), nn.LeakyReLU(0.1),
            nn.Conv1d(hiddens, output_channels, kernel_size=9, stride=1, padding=4)
        )
        self.std = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(hiddens), nn.LeakyReLU(0.1),
            nn.Conv1d(hiddens, output_channels, kernel_size=9, stride=1, padding=4),
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
            ResBlock(input_channels, 'instance'), ResBlock(input_channels, 'instance'),
            nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1),
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
            nn.BatchNorm1d(output_channels), nn.LeakyReLU(0.1),
            ResBlock(output_channels, 'batch'), ResBlock(output_channels, 'batch')
        )

    def forward(self, X):
        return self.block(X)


class Encoder(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.down1 = DownSampler(input_channels, 16, kernel_size=8, stride=8)
        self.dist1 = Distribution(16, 16, 8)
        self.down2 = DownSampler(16, 32, kernel_size=4, stride=4)
        self.dist2 = Distribution(32, 32, 16)
        self.down3 = DownSampler(32, 64, kernel_size=4, stride=4)
        self.dist3 = Distribution(64, 64, 32)
        self.down4 = DownSampler(64, 128, kernel_size=4, stride=4)
        self.dist4 = Distribution(128, 128, 128)

    def forward(self, X):
        batch_size, window_size, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.permute(0, 1, 3, 2).contiguous()
        X = X.view(batch_size, window_size * series, num_channels)
        X = X.permute(0, 2, 1).contiguous()
        X1 = self.down1(X)
        mu1, sigma1 = self.dist1(X1)
        X2 = self.down2(X1)
        mu2, sigma2 = self.dist2(X2)
        X3 = self.down3(X2)
        mu3, sigma3 = self.dist3(X3)
        X4 = self.down4(X3)
        mu4, sigma4 = self.dist4(X4)
        return [(mu1, sigma1), (mu2, sigma2), (mu3, sigma3), (mu4, sigma4)]


class Decoder(nn.Module):
    def __init__(self, output_channels, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.up1 = UpSampler(128, 32, 6, 4)
        self.up2 = UpSampler(64, 16, 5, 4)
        self.up3 = UpSampler(32, 8, 6, 4)
        self.up4 = UpSampler(16, 16, 8, 8)
        self.last_layer = nn.Sequential(
            nn.Conv1d(16, 16, kernel_size=449, stride=1, padding='same'),
            nn.InstanceNorm1d(16), nn.LeakyReLU(0.1),
            nn.Conv1d(16, output_channels, kernel_size=449, stride=1, padding='same')
        )

    def forward(self, Z):
        Z1 = self.up1(Z[3])
        Z2 = self.up2(torch.cat((Z[2], Z1), dim=1))
        Z3 = self.up3(torch.cat((Z[1], Z2), dim=1))
        Z4 = self.up4(torch.cat((Z[0], Z3), dim=1))
        X = self.last_layer(Z4)
        batch_size, num_channels, series = X.shape[0], X.shape[1], X.shape[2]
        X = X.permute(0, 2, 1).contiguous()
        X = X.view(batch_size, -1, 3000, num_channels)
        X = X.permute(0, 1, 3, 2).contiguous()
        return X

    def generate(self, batch_size, device):
        Z = [torch.randn((batch_size, 8, 3750), dtype=torch.float32, requires_grad=False, device=device),
             torch.randn((batch_size, 16, 937), dtype=torch.float32, requires_grad=False, device=device),
             torch.randn((batch_size, 32, 234), dtype=torch.float32, requires_grad=False, device=device),
             torch.randn((batch_size, 128, 58), dtype=torch.float32, requires_grad=False, device=device)]
        return self.forward(Z)


class EEGVAE(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(EEGVAE, self).__init__(**kwargs)
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(input_channels)

    def forward(self, X):
        distributions = self.encoder(X)
        kl_loss, Z = 0, []
        for mu, sigma in distributions:
            eps = torch.randn_like(mu, requires_grad=False)
            z = sigma * eps + mu
            Z.append(z)
            kl_loss += torch.mean(-2 * torch.log(sigma) + sigma.pow(2) + mu.pow(2) - 1) * 0.5
        X_hat = self.decoder(Z)
        return X_hat, kl_loss


if __name__ == '__main__':
    X = torch.randn((4, 10, 2, 3000), dtype=torch.float32, requires_grad=False)
    net = EEGVAE(2)
    X_hat, kl_loss = net(X)
    print(X_hat.shape, kl_loss)
    print(net.decoder.generate(8).shape)
