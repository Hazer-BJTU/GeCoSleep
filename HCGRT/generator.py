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


class ResBlock(nn.Module):
    def __init__(self, channels, norm_type, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.channels = channels
        self.norm_type = norm_type
        self.block = None
        if norm_type == 'batch':
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding='same'),
                nn.BatchNorm1d(channels), nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding='same'),
                nn.BatchNorm1d(channels)
            )
        elif norm_type == 'instance':
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding='same'),
                nn.InstanceNorm1d(channels), nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size=9, stride=1, padding='same'),
                nn.InstanceNorm1d(channels)
            )
        self.activate = nn.LeakyReLU(0.1)

    def forward(self, X):
        return self.activate(self.block(X) + X)


class DownSample(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=0, **kwargs):
        super(DownSample, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(output_channels),
            ResBlock(output_channels, 'batch'), ResBlock(output_channels, 'batch')
        )

    def forward(self, X):
        return self.block(X)


class UpSample(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding=0, **kwargs):
        super(UpSample, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.block = nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm1d(output_channels),
            ResBlock(output_channels, 'instance'), ResBlock(output_channels, 'instance')
        )

    def forward(self, X):
        return self.block(X)


class SampleVAEencoder(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(SampleVAEencoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(0.1),
            DownSample(16, 32, kernel_size=4, stride=4),
            DownSample(32, 64, kernel_size=5, stride=5),
            DownSample(64, 128, kernel_size=5, stride=5)
        )
        self.label2vec = nn.Embedding(5, 960)
        self.mean = nn.Sequential(
            nn.Conv1d(160, 128, kernel_size=9, stride=1, padding='same'), nn.LeakyReLU(0.1),
            nn.Conv1d(128, 128, kernel_size=9, stride=1, padding='same')
        )
        self.std = nn.Sequential(
            nn.Conv1d(160, 128, kernel_size=9, stride=1, padding='same'), nn.LeakyReLU(0.1),
            nn.Conv1d(128, 128, kernel_size=9, stride=1, padding='same'), nn.Softplus()
        )

    def forward(self, X, y):
        batch_size, seq_length, num_channels, series = X.shape
        X = X.view(batch_size * seq_length, num_channels, series)
        y = y.view(batch_size * seq_length)
        X = self.block(X)
        y = self.label2vec(y)
        y = y.view(batch_size * seq_length, 32, -1)
        X = torch.cat((X, y), dim=1)
        return self.mean(X), self.std(X)


class SampleVAEdecoder(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(SampleVAEdecoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.label2vec = nn.Embedding(5, 960)
        self.projection = nn.Sequential(
            nn.Conv1d(160, 128, kernel_size=9, stride=1, padding='same'),
            nn.InstanceNorm1d(128), nn.LeakyReLU(0.1),
        )
        self.block = nn.Sequential(
            UpSample(128, 64, 5, 5),
            UpSample(64, 32, 5, 5),
            UpSample(32, 16, 4, 4)
        )
        self.output = nn.Conv1d(16, input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, X, y):
        batch_size, seq_length = y.shape
        y = y.view(batch_size * seq_length)
        y = self.label2vec(y)
        y = y.view(batch_size * seq_length, 32, -1)
        X = torch.cat((X, y), dim=1)
        X = self.projection(X)
        X = self.block(X)
        X = self.output(X)
        X = X.view(batch_size, seq_length, self.input_channels, -1)
        return X


class SampleVAE(nn.Module):
    def __init__(self, input_channels, **kwargs):
        super(SampleVAE, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.encoder = SampleVAEencoder(input_channels)
        self.decoder = SampleVAEdecoder(input_channels)

    def forward(self, X, y):
        mu, sigma = self.encoder(X, y)
        eps = torch.randn_like(mu, requires_grad=False)
        z = eps * sigma + mu
        X_hat = self.decoder(z, y)
        kl_loss = torch.mean(-2 * torch.log(sigma) + sigma.pow(2) + mu.pow(2) - 1) * 0.5
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
