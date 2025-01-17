import torch

from models import *
from data_preprocessing import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float32, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float32, requires_grad=False).unsqueeze(1)
        div_term = torch.arange(0, d_model, 2, dtype=torch.float32, requires_grad=False)
        div_term = torch.exp(div_term * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, X):
        batch_size, seq_length, d_model = X.shape
        X = X + self.pe[:, :seq_length, :].to(X.device)
        return X


class SequentialVAEencoder(nn.Module):
    def __init__(self, embeddings, hiddens, heads, layers, dropout, **kwargs):
        super(SequentialVAEencoder, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.hiddens = hiddens
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.positional_encoding = PositionalEncoding(embeddings)
        self.block1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embeddings, heads, dropout=dropout, batch_first=True),
            num_layers=layers
        )
        self.label2vec = nn.Embedding(5, embeddings)
        self.block2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embeddings, heads, dropout=dropout, batch_first=True),
            num_layers=layers
        )
        self.mean = nn.Sequential(
            nn.Linear(embeddings, hiddens), nn.LeakyReLU(0.1),
            nn.Linear(hiddens, embeddings)
        )
        self.std = nn.Sequential(
            nn.Linear(embeddings, hiddens), nn.LeakyReLU(0.1),
            nn.Linear(hiddens, embeddings), nn.Softplus()
        )

    def forward(self, X, y):
        batch_size, seq_length, embeddings = X.shape
        X = self.positional_encoding(X)
        X = self.block1(X)
        y = y.view(batch_size * seq_length)
        y = self.label2vec(y)
        y = y.view(batch_size, seq_length, embeddings)
        X = X + y
        X = self.block2(X)
        X = X.view(batch_size * seq_length, embeddings)
        mu, sigma = self.mean(X), self.std(X)
        mu = mu.view(batch_size, seq_length, embeddings)
        sigma = sigma.view(batch_size, seq_length, embeddings)
        return mu, sigma


class SequentialVAEdecoder(nn.Module):
    def __init__(self, embeddings, heads, layers, dropout, **kwargs):
        super(SequentialVAEdecoder, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.positional_encoding = PositionalEncoding(embeddings)
        self.block1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embeddings, heads, dropout=dropout, batch_first=True),
            num_layers=layers
        )
        self.label2vec = nn.Embedding(5, embeddings)
        self.block2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embeddings, heads, dropout=dropout, batch_first=True),
            num_layers=layers
        )

    def forward(self, X, y):
        batch_size, seq_length, embeddings = X.shape
        X = self.positional_encoding(X)
        X = self.block1(X)
        y = y.view(batch_size * seq_length)
        y = self.label2vec(y)
        y = y.view(batch_size, seq_length, embeddings)
        X = X + y
        X = self.block2(X)
        return X

    def generate(self, y):
        batch_size, seq_length = y.shape
        z = torch.randn((batch_size, seq_length, self.embeddings),
                        dtype=torch.float32, requires_grad=False, device=y.device)
        X_hat = self.forward(z, y)
        return X_hat


class SequentialVAE(nn.Module):
    def __init__(self, embeddings, dropout, **kwargs):
        super(SequentialVAE, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.hiddens = int(embeddings * 1.5)
        self.dropout = dropout
        self.encoder = SequentialVAEencoder(embeddings, self.hiddens, 8, 2, dropout)
        self.decoder = SequentialVAEdecoder(embeddings, 8, 2, dropout)

    def forward(self, X, y):
        mu, sigma = self.encoder(X, y)
        eps = torch.randn_like(mu, requires_grad=False)
        z = eps * sigma + mu
        X_hat = self.decoder(z, y)
        kl_loss = torch.mean(-2 * torch.log(sigma) + sigma.pow(2) + mu.pow(2) - 1) * 0.5
        return X_hat, kl_loss


class Distribution(nn.Module):
    def __init__(self, input_channels, hiddens, output_channels, **kwargs):
        super(Distribution, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.hiddens = hiddens
        self.output_channels = output_channels
        self.mean = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=1, stride=1), nn.LeakyReLU(0.1),
            nn.Conv1d(hiddens, output_channels, kernel_size=1, stride=1)
        )
        self.std = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=1, stride=1), nn.LeakyReLU(0.1),
            nn.Conv1d(hiddens, output_channels, kernel_size=1, stride=1), nn.Softplus()
        )

    def forward(self, X):
        return self.mean(X), self.std(X)


class DownSample(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=0, **kwargs):
        super(DownSample, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1)
        )

    def forward(self, X):
        X = self.block1(X)
        X_down = self.block2(X)
        return X, X_down


class UpSample(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=0, **kwargs):
        super(UpSample, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1),
            nn.ConvTranspose1d(output_channels, output_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm1d(output_channels, affine=True), nn.LeakyReLU(0.1)
        )

    def forward(self, X, X_next):
        X = self.block(X)
        X = torch.cat((X, X_next), dim=1)
        return X


class SampleVAEencoder(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(SampleVAEencoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.down1 = DownSample(input_channels, 4, 4, 4)
        self.down2 = DownSample(4, 32, 5, 5)
        self.down3 = DownSample(32, 128, 5, 5)
        self.down4 = DownSample(128, 256, 5, 5)
        self.label2vec = nn.Embedding(5, 1536)
        self.dist1 = Distribution(4, 4, 4)
        self.dist2 = Distribution(32, 32, 32)
        self.dist3 = Distribution(128, 128, 128)
        self.dist4 = Distribution(256, 256, 256)
        self.dist5 = Distribution(512, 256, 256)

    def forward(self, X, y):
        batch_size, seq_length, num_channels, series = X.shape
        X = X.view(batch_size * seq_length, num_channels, series)
        y = y.view(batch_size * seq_length)
        X1, X1_down = self.down1(X)
        X2, X2_down = self.down2(X1_down)
        X3, X3_down = self.down3(X2_down)
        X4, X5 = self.down4(X3_down)
        y = self.label2vec(y)
        y = y.view(X5.shape)
        X5 = torch.cat((X5, y), dim=1)
        mu1, sigma1 = self.dist1(X1)
        mu2, sigma2 = self.dist2(X2)
        mu3, sigma3 = self.dist3(X3)
        mu4, sigma4 = self.dist4(X4)
        mu5, sigma5 = self.dist5(X5)
        return [(mu1, sigma1), (mu2, sigma2), (mu3, sigma3), (mu4, sigma4), (mu5, sigma5)]


class SampleVAEdecoder(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(SampleVAEdecoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.label2vec = nn.Embedding(5, 1536)
        self.projection = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1, stride=1), nn.LeakyReLU(0.1),
            nn.Conv1d(256, 256, kernel_size=1, stride=1)
        )
        self.up1 = UpSample(256, 256, 5, 5)
        self.up2 = UpSample(512, 128, 5, 5)
        self.up3 = UpSample(256, 32, 5, 5)
        self.up4 = UpSample(64, 4, 4, 4)
        self.output = nn.Conv1d(8, input_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, Zs, y):
        batch_size, seq_length = y.shape
        y = y.view(batch_size * seq_length)
        y = self.label2vec(y)
        Z1 = Zs[4]
        y = y.view(Z1.shape)
        Z1 = torch.cat((Z1, y), dim=1)
        Z1 = self.projection(Z1)
        Z2 = self.up1(Z1, Zs[3])
        Z3 = self.up2(Z2, Zs[2])
        Z4 = self.up3(Z3, Zs[1])
        Z5 = self.up4(Z4, Zs[0])
        X_hat = self.output(Z5)
        X_hat = X_hat.view(batch_size, seq_length, -1, 3000)
        return X_hat

    def generate(self, y):
        batch_size, seq_length = y.shape
        Zs = [
            torch.randn((batch_size * seq_length, 4, 3000), dtype=torch.float32, requires_grad=False, device=y.device),
            torch.randn((batch_size * seq_length, 32, 750), dtype=torch.float32, requires_grad=False, device=y.device),
            torch.randn((batch_size * seq_length, 128, 150), dtype=torch.float32, requires_grad=False, device=y.device),
            torch.randn((batch_size * seq_length, 256, 30), dtype=torch.float32, requires_grad=False, device=y.device),
            torch.randn((batch_size * seq_length, 256, 6), dtype=torch.float32, requires_grad=False, device=y.device)
        ]
        X_hat = self.forward(Zs, y)
        return X_hat


class SampleVAE(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(SampleVAE, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.encoder = SampleVAEencoder(input_channels)
        self.decoder = SampleVAEdecoder(input_channels)

    def forward(self, X, y):
        distributions = self.encoder(X, y)
        Zs, kl_loss = [], 0
        for mu, sigma in distributions:
            eps = torch.randn_like(mu, requires_grad=False)
            z = eps * sigma + mu
            Zs.append(z)
            kl_loss += torch.mean(-2 * torch.log(sigma) + sigma.pow(2) + mu.pow(2) - 1) * 0.5
        kl_loss = kl_loss / 5
        X_hat = self.decoder(Zs, y)
        return X_hat, kl_loss


if __name__ == '__main__':
    '''
    net = SequentialVAE(512, 0.1)
    X = torch.randn((4, 10, 512), dtype=torch.float32, requires_grad=False)
    y = torch.randint(0, 5, (4, 10), dtype=torch.int64, requires_grad=False)
    X_hat, kl_loss = net(X, y)
    print(X_hat.shape, kl_loss)
    torch.save(net.state_dict(), 'VAE.pth')
    net = SampleVAE(2)
    X = torch.randn((8, 10, 2, 3000), dtype=torch.float32, requires_grad=False)
    y = torch.randint(0, 5, (8, 10), dtype=torch.int64, requires_grad=False)
    X_hat, kl_loss = net(X, y)
    print(X_hat.shape, kl_loss)
    '''
