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
            nn.Conv1d(hiddens, output_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.Softplus()
        )

    def forward(self, X):
        mu, sigma = self.mean(X), self.std(X)
        return mu, sigma


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
        self.conv4_1 = nn.ConvTranspose1d(32, 32, kernel_size=4, stride=4)
        self.conv4_2 = nn.ConvTranspose1d(32, 32, kernel_size=4, stride=4)
        self.output = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=49, stride=1, padding='same'),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, output_channels, kernel_size=49, stride=1, padding='same')
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
            kl_divergence = -2 * torch.log(sigma + 1e-5) + sigma.pow(2) + mu.pow(2) - 1
            kl_loss += torch.mean(kl_divergence) * 0.5
            samples.append(z)
        output = self.decoder(samples)
        return output, kl_loss


if __name__ == '__main__':
    net = UVAE(2)
    critic = DeepSleepNet(0.25)
    net.apply(init_weight)
    critic.apply(init_weight)
    datas, labels = load_data_sleepedf('/home/ShareData/sleep-edf-153-3chs', 10, ['Fpz-Cz', 'EOG'], 50)
    train, _, _ = create_fold([idx for idx in range(50)], [], [], [datas], [labels])
    train_loader = DataLoader(train, batch_size=256, shuffle=False)
    device = torch.device(f'cuda:{0}')
    net.to(device)
    critic.to(device)
    num_epoch = 200
    lr = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(num_epoch // 6, 1), 0.6)
    optimizerC = torch.optim.Adam(critic.parameters(), lr=lr)
    schedulerC = torch.optim.lr_scheduler.StepLR(optimizerC, max(num_epoch // 6, 1), 0.6)
    recloss = nn.MSELoss()
    celoss = nn.CrossEntropyLoss()
    net.train(), critic.train()
    for epoch in range(num_epoch):
        total_rec_loss, total_kl_loss, total_task_loss, total_L, cnt = 0, 0, 0, 0, 0
        for X, y, _ in train_loader:
            X, y = X.to(device), y.to(device)
            '''train critic'''
            optimizerC.zero_grad()
            y_hat = critic(X)
            L = celoss(y_hat, y.view(-1))
            L.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=20, norm_type=2)
            optimizerC.step()
            total_L += L.item()
            '''train uvae'''
            optimizer.zero_grad()
            optimizerC.zero_grad()
            X_hat, kl_loss = net(X)
            rec_loss = recloss(X_hat, X)
            yc = critic(X_hat)
            task_loss = celoss(yc, y_hat.detach().softmax(dim=1))
            (rec_loss + kl_loss + task_loss).backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            total_rec_loss += rec_loss.item()
            total_kl_loss += kl_loss.item()
            total_task_loss += task_loss.item()
            cnt += 1
        print(f'epoch: {epoch}, rec loss: {total_rec_loss / cnt:.3f}, kl loss: {total_kl_loss / cnt:.3f}, '
              f'task loss: {total_task_loss / cnt:.3f}, L: {total_L / cnt:.3f}')
        scheduler.step()
        schedulerC.step()
    torch.save(net.state_dict(), 'uvae.pth')
    '''
    net = UVAE(2)
    net.apply(init_weight)
    net.load_state_dict(torch.load('uvae.pth', map_location='cpu', weights_only=True))
    net.eval()
    samples = [torch.randn((1, 16, 1875)), torch.randn((1, 32, 375)), torch.randn((1, 64, 75)), torch.randn((1, 256, 15))]
    X = net.decoder(samples)
    X = X.detach()
    datas, labels = load_data_sleepedf('/home/ShareData/sleep-edf-153-3chs', 10, ['Fpz-Cz', 'EOG'], 1)
    train, _, _ = create_fold([0], [], [], [datas], [labels])
    train_loader = DataLoader(train, batch_size=256, shuffle=False)
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(np.arange(3000), X[0][4][0].numpy(), 'b')
    axs[0].set_title('fake1')
    axs[1].plot(np.arange(3000), X[0][4][1].numpy(), 'b')
    axs[1].set_title('fake2')
    for X, _, _ in train_loader:
        X = X.detach()
        axs[2].plot(np.arange(3000), X[0][5][0].numpy(), 'b')
        axs[2].set_title('real1')
        axs[3].plot(np.arange(3000), X[0][5][1].numpy(), 'b')
        axs[3].set_title('real2')
        break
    plt.show()
    plt.savefig('generater.png')
    '''
