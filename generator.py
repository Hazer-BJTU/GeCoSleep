import torch
from torch import nn
from models import init_weight
from data_preprocessing import *


def conv_layer(input_channels, output_channels, kernel_size, stride, padding=None):
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(output_channels), nn.ReLU(inplace=True)
    )


class Distribute(nn.Module):
    def __init__(self, input_channels, hiddens, output_channels, kernel_size=5, **kwargs):
        super(Distribute, self).__init__(**kwargs)
        self.mean = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm1d(hiddens), nn.ReLU(inplace=True),
            nn.Conv1d(hiddens, output_channels, kernel_size=kernel_size, stride=1, padding='same')
        )
        self.std = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm1d(hiddens), nn.ReLU(inplace=True),
            nn.Conv1d(hiddens, output_channels, kernel_size=kernel_size, stride=1, padding='same')
        )

    def forward(self, X):
        mu, sigma = self.mean(X), self.std(X)
        return mu, sigma.exp()


class Encoder(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.conv1_1 = conv_layer(input_channels, 8, 49, 4)
        self.conv1_2 = conv_layer(8, 32, 49, 4)
        self.enc1 = Distribute(32, 32, 16)
        self.pooling1 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.conv2_1 = conv_layer(32, 64, 5, 1)
        self.conv2_2 = conv_layer(64, 64, 5, 1)
        self.enc2 = Distribute(64, 64, 32)
        self.pooling2 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.conv3_1 = conv_layer(64, 128, 5, 1)
        self.conv3_2 = conv_layer(128, 128, 5, 1)
        self.enc3 = Distribute(128, 128, 64)
        self.pooling3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.conv4_1 = conv_layer(128, 256, 5, 1)
        self.conv4_2 = conv_layer(256, 256, 5, 1)
        self.enc4 = Distribute(256, 256, 256)

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.permute(0, 1, 3, 2).contiguous()
        X = X.view(batch_size, seq_length * series, num_channels)
        X = X.permute(0, 2, 1).contiguous()
        X1 = self.conv1_2(self.conv1_1(X))
        mu1, sigma1 = self.enc1(X1)
        X1_down = self.pooling1(X1)
        X2 = self.conv2_2(self.conv2_1(X1_down))
        mu2, sigma2 = self.enc2(X2)
        X2_down = self.pooling2(X2)
        X3 = self.conv3_2(self.conv3_1(X2_down))
        mu3, sigma3 = self.enc3(X3)
        X3_down = self.pooling3(X3)
        X4 = self.conv4_2(self.conv4_1(X3_down))
        mu4, sigma4 = self.enc4(X4)
        return [(mu1, sigma1), (mu2, sigma2), (mu3, sigma3), (mu4, sigma4)]


class Decoder(nn.Module):
    def __init__(self, output_channels, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.output_channels = output_channels
        self.conv1_1 = conv_layer(256, 128, 5, 1)
        self.conv1_2 = conv_layer(128, 128, 5, 1)
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=5)
        self.conv2_1 = conv_layer(128, 64, 5, 1)
        self.conv2_2 = conv_layer(64, 64, 5, 1)
        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5)
        self.conv3_1 = conv_layer(64, 32, 5, 1)
        self.conv3_2 = conv_layer(32, 32, 5, 1)
        self.up3 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=5)
        self.conv4_1 = nn.ConvTranspose1d(32, 8, kernel_size=4, stride=4)
        self.conv4_2 = nn.ConvTranspose1d(8, 4, kernel_size=4, stride=4)
        self.output = nn.Sequential(
            nn.Conv1d(4, 4, kernel_size=49, stride=1, padding='same'),
            nn.BatchNorm1d(4), nn.ReLU(),
            nn.Conv1d(4, output_channels, kernel_size=49, stride=1, padding='same')
        )

    def forward(self, samples):
        X1 = self.conv1_2(self.conv1_1(samples[3]))
        X1_up = torch.cat((self.up1(X1), samples[2]), dim=1)
        X2 = self.conv2_2(self.conv2_1(X1_up))
        X2_up = torch.cat((self.up2(X2), samples[1]), dim=1)
        X3 = self.conv3_2(self.conv3_1(X2_up))
        X3_up = torch.cat((self.up3(X3), samples[0]), dim=1)
        X4 = self.conv4_2(self.conv4_1(X3_up))
        X4 = self.output(X4)
        batch_size, num_channels, seq_length = X4.shape[0], X4.shape[1], X4.shape[2] // 3000
        X4 = X4.permute(0, 2, 1).contiguous()
        X4 = X4.view(batch_size, seq_length, -1, num_channels)
        X4 = X4.permute(0, 1, 3, 2).contiguous()
        return X4


class UVAE(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(UVAE, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.encoder = Encoder(input_channels)
        self.decoder = Decoder(input_channels)

    def forward(self, X):
        distributions = self.encoder(X)
        samples = []
        kl_loss = 0
        for mu, sigma in distributions:
            eps = torch.randn(mu.shape, dtype=torch.float32, requires_grad=False, device=X.device)
            z = mu + eps * sigma
            kl_divergence = -2 * torch.log(sigma) + sigma.pow(2) + mu.pow(2) - 1
            kl_loss += torch.mean(kl_divergence) * 0.5
            samples.append(z)
        output = self.decoder(samples)
        return output, kl_loss


if __name__ == '__main__':
    net = UVAE(2)
    X = torch.randn(4, 10, 2, 3000)
    y, kl_loss = net(X)
    print(y.shape, kl_loss)
    torch.save(net.state_dict(), 'uvae.pth')
