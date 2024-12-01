import torch
import torch.nn as nn


class RawDataEncoder(nn.Module):
    def __init__(self, input_channels, dropout, **kwargs):
        super(RawDataEncoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=50, stride=6), nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
            nn.Conv1d(128, 32, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
            nn.AvgPool1d(kernel_size=4, stride=4)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=400, stride=50), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
            nn.Conv1d(128, 32, kernel_size=3, stride=1, padding='same'), nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )
        self.block3 = nn.Dropout(dropout)

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.view(batch_size * seq_length, num_channels, series)
        y1 = self.block1(X)
        y2 = self.block2(X)
        y = torch.cat((y1, y2), dim=2)
        y = self.block3(y)
        y = y.view(batch_size, seq_length, -1)
        return y


class FrequencyEncoder(nn.Module):
    def __init__(self, input_channels, dropout, output_features=512, sample_rate=100, **kwargs):
        super(FrequencyEncoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.output_features = output_features
        self.sample_rate = sample_rate
        self.pooling = nn.AvgPool1d(kernel_size=5, stride=5)
        self.block = nn.Sequential(
            nn.Linear(600, output_features),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(output_features, output_features),
            nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.view(batch_size * seq_length, num_channels, series)
        y = torch.fft.rfft(X, dim=2)
        y = self.pooling(torch.abs(y))
        y = y.view(batch_size * seq_length, -1)
        y = self.block(y)
        y = y.view(batch_size, seq_length, -1)
        return y


class GRULayer(nn.Module):
    def __init__(self, input_features, hiddens, **kwargs):
        super(GRULayer, self).__init__(**kwargs)
        self.input_features = input_features
        self.hiddens = hiddens
        self.block = nn.GRU(input_features, hiddens, batch_first=True, bidirectional=True)

    def get_initial_state(self, batch_size, device):
        return torch.zeros((2, batch_size, self.hiddens), dtype=torch.float32, device=device)

    def forward(self, X):
        H0 = self.get_initial_state(X.shape[0], X.device)
        (output, Hn) = self.block(X, H0)
        if not output.is_contiguous():
            output = output.contiguous()
        return output


class AttentionLayer(nn.Module):
    def __init__(self, input_features, hiddens, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.input_features = input_features
        self.hiddens = hiddens
        self.K = nn.Parameter(torch.randn((input_features, hiddens), dtype=torch.float32))
        self.norm1 = nn.LayerNorm(hiddens)
        self.Q = nn.Parameter(torch.randn((input_features, hiddens), dtype=torch.float32))
        self.norm2 = nn.LayerNorm(hiddens)
        self.V = nn.Parameter(torch.randn((input_features, input_features), dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(input_features, dtype=torch.float32))
        self.norm3 = nn.LayerNorm(input_features)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, X):
        batch_size, seq_length, featrues = X.shape[0], X.shape[1], X.shape[2]
        mid = (seq_length + 1) // 2
        keys = self.norm1(X @ self.K)
        query = self.norm2(torch.unsqueeze(X[:, mid, :], dim=1) @ self.Q)
        values = self.norm3(X @ self.V + self.bias)
        alpha = torch.bmm(query, torch.transpose(keys, 1, 2)) / self.hiddens
        alpha = self.softmax(alpha)
        output = torch.bmm(alpha, values).squeeze(dim=1)
        return output


class SleepNet(nn.Module):
    def __init__(self, input_channels, dropout, **kwargs):
        super(SleepNet, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.encoder1 = RawDataEncoder(input_channels, dropout)
        self.encoder2 = FrequencyEncoder(input_channels, dropout, 512)
        self.norm1 = nn.LayerNorm(672)
        self.norm2 = nn.LayerNorm(512)
        self.gru1 = GRULayer(672, 256)
        self.gru2 = GRULayer(512, 256)
        self.att1 = AttentionLayer(512, 256)
        self.att2 = AttentionLayer(512, 256)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 5)
        )

    def forward(self, X):
        f1 = self.encoder1(X)
        f1 = self.norm1(f1)
        f1 = self.gru1(f1)
        f1 = self.att1(f1)
        f2 = self.encoder2(X)
        f2 = self.norm2(f2)
        f2 = self.gru2(f2)
        f2 = self.att2(f2)
        f = torch.cat((f1, f2), dim=1)
        output = self.classifier(f)
        return output


def init_weight(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)


if __name__ == '__main__':
    net = SleepNet(2, 0.25)
    X = torch.randn((8, 5, 2, 3000), dtype=torch.float32)
    print(net(X).shape)
    torch.save(net.state_dict(), 'MSleepNet.pth')
