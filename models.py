import torch
import math
import torch.nn as nn


class MultiScaleCNN(nn.Module):
    def __init__(self, input_channels, hiddens, output_channels, kernel_size, stride, pooling, dropout, **kwargs):
        super(MultiScaleCNN, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.hiddens = hiddens
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pooling = pooling
        self.dropout = dropout
        self.block0 = nn.Sequential(
            nn.Conv1d(input_channels, hiddens, kernel_size=kernel_size[0], stride=stride),
            nn.BatchNorm1d(hiddens), nn.ReLU(), nn.MaxPool1d(kernel_size=pooling, stride=pooling),
            nn.Dropout(dropout)
        )
        self.block1 = nn.Sequential(
            nn.Conv1d(hiddens, output_channels, kernel_size=kernel_size[1], stride=1, padding='same'),
            nn.BatchNorm1d(output_channels), nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size[1], stride=1, padding='same'),
            nn.BatchNorm1d(output_channels), nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size[1], stride=1, padding='same'),
            nn.BatchNorm1d(output_channels)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hiddens, output_channels, kernel_size=kernel_size[1], stride=1, padding='same'),
            nn.BatchNorm1d(output_channels)
        )
        self.block3 = nn.Sequential(
            nn.ReLU(), nn.MaxPool1d(kernel_size=pooling // 4, stride=pooling // 4)
        )

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape
        X = X.view(batch_size * seq_length, num_channels, series)
        X = self.block0(X)
        output = self.block3(self.block1(X) + self.block2(X))
        return output


class CNNencoders(nn.Module):
    def __init__(self, input_channels, dropout, **kwargs):
        super(CNNencoders, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.encoder1 = MultiScaleCNN(input_channels, 64, 128, [400, 3], 50, 8, dropout)
        self.encoder2 = MultiScaleCNN(input_channels, 64, 128, [200, 5], 25, 8, dropout)
        self.encoder3 = MultiScaleCNN(input_channels, 64, 128, [100, 7], 12, 8, dropout)
        self.encoder4 = MultiScaleCNN(input_channels, 64, 128, [50, 9], 6, 8, dropout)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, X):
        X1 = self.encoder1(X)
        X2 = self.encoder2(X)
        X3 = self.encoder3(X)
        X4 = self.encoder4(X)
        output = torch.cat((X1, X2, X3, X4), dim=2)
        output = self.dropout_layer(output)
        output = output.permute(0, 2, 1).contiguous()
        return output


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


class ShortTermEncoder(nn.Module):
    def __init__(self, embeddings, heads, layers, dropout, keep=4, **kwargs):
        super(ShortTermEncoder, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.keep = keep
        self.positional_encoding = PositionalEncoding(embeddings)
        self.transformers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embeddings, heads, dim_feedforward=1024, dropout=dropout, batch_first=True),
            num_layers=layers
        )
        self.CLS_tokens = nn.Parameter(torch.randn((1, keep, embeddings), dtype=torch.float32, requires_grad=True))

    def forward(self, X):
        batch_size, seq_length, features = X.shape
        X = torch.cat((self.CLS_tokens.expand(batch_size, self.keep, self.embeddings), X), dim=1)
        X = self.positional_encoding(X)
        X = self.transformers(X)
        X = X[:, :self.keep, :]
        X = X.view(X.shape[0], -1)
        return X


class LongTermEncoder(nn.Module):
    def __init__(self, embeddings, heads, layers, dropout, **kwargs):
        super(LongTermEncoder, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.heads = heads
        self.layers = layers
        self.dropout = dropout
        self.positional_encoding = PositionalEncoding(embeddings)
        self.transformers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embeddings, heads, dim_feedforward=1024, dropout=dropout, batch_first=True),
            num_layers=layers
        )

    def forward(self, X):
        batch_size, seq_length, embeddings = X.shape
        X = self.positional_encoding(X)
        X = self.transformers(X)
        X = X.view(batch_size * seq_length, embeddings)
        return X


class SleepNet(nn.Module):
    def __init__(self, input_channels, dropout, **kwargs):
        super(SleepNet, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.cnn = CNNencoders(input_channels, dropout)
        self.short_term_encoder = ShortTermEncoder(128, 8, 4, dropout)
        self.long_term_encoder = LongTermEncoder(512, 8, 4, dropout)
        self.resblock = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 5)
        )

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape
        X = self.cnn(X)
        X = self.short_term_encoder(X)
        X = X.view(batch_size, seq_length, -1)
        r = self.resblock(X.view(batch_size * seq_length, -1))
        X = self.long_term_encoder(X)
        X = torch.cat((r, X), dim=1)
        X = self.classifier(X)
        return X

    def features(self, X):
        batch_size, seq_length, num_channels, series = X.shape
        X = self.cnn(X)
        X = self.short_term_encoder(X)
        X = X.view(batch_size, seq_length, -1)
        return X

    def classify(self, X):
        batch_size, seq_length, embeddings = X.shape
        r = self.resblock(X.view(batch_size * seq_length, -1))
        X = self.long_term_encoder(X)
        X = torch.cat((r, X), dim=1)
        X = self.classifier(X)
        return X

    def freeze_parameters(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        for name, param in self.cnn.named_parameters():
            param.requires_grad = False
        for name, param in self.short_term_encoder.named_parameters():
            param.requires_grad = False


def init_weight(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)


if __name__ == '__main__':
    X = torch.randn((4, 10, 2, 3000))
    net = SleepNet(2, 0.25)
    print(net(X).shape)
    torch.save(net.state_dict(), 'SleepNet.pth')
