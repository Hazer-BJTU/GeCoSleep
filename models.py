import torch
import torch.nn as nn


class RawDataEncoder(nn.Module):
    def __init__(self, input_channels, dropout, **kwargs):
        super(RawDataEncoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=50, stride=6),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AvgPool1d(kernel_size=4, stride=4)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=400, stride=50),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
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
    def __init__(self, input_channels, dropout, sample_rate=100, **kwargs):
        super(FrequencyEncoder, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.sample_rate = sample_rate
        self.block = nn.Sequential(
            nn.Conv1d(input_channels + 1, 64, kernel_size=5, stride=5),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AvgPool1d(kernel_size=4, stride=4), nn.Dropout(dropout)
        )

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.view(batch_size * seq_length, num_channels, series)
        y = torch.abs(torch.fft.rfft(X, dim=2))
        freq = torch.fft.rfftfreq(series, 1 / self.sample_rate).expand(y.shape[0], 1, y.shape[2]).to(y.device)
        y = torch.cat((y, freq), dim=1)
        y = self.block(y)
        y = y.view(batch_size, seq_length, -1)
        return y


class AttentionLayer(nn.Module):
    def __init__(self, input_feature1, input_feaure2, hiddens, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.V1 = nn.Sequential(
            nn.Linear(input_feature1, hiddens),
            nn.LayerNorm(hiddens)
        )
        self.K1 = nn.Sequential(
            nn.Linear(input_feature1, hiddens),
            nn.LayerNorm(hiddens)
        )
        self.V2 = nn.Sequential(
            nn.Linear(input_feaure2, hiddens),
            nn.LayerNorm(hiddens)
        )
        self.K2 = nn.Sequential(
            nn.Linear(input_feaure2, hiddens),
            nn.LayerNorm(hiddens)
        )
        self.Q = nn.Sequential(
            nn.Linear(hiddens, 1),
            nn.Softplus()
        )

    def forward(self, X1, X2):
        batch_size, seq_length = X1.shape[0], X1.shape[1]
        X1, X2 = X1.view(batch_size * seq_length, -1), X2.view(batch_size * seq_length, -1)
        v1, k1 = self.V1(X1), self.K1(X1)
        v2, k2 = self.V2(X2), self.K2(X2)
        c1, c2 = self.Q(k1), self.Q(k2)
        c = (c1 + c2).detach()
        output = (c1 * v1 + c2 * v2) / c
        output = output.view(batch_size, seq_length, -1)
        return output


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


class SleepNet(nn.Module):
    def __init__(self, input_channels, dropout, **kwargs):
        super(SleepNet, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.encoder1 = RawDataEncoder(input_channels, dropout)
        self.encoder2 = FrequencyEncoder(input_channels, dropout)
        self.attention = AttentionLayer(2688, 1152, 512)
        self.resblock = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.gru = GRULayer(512, 256)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 5)
        )

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        f1 = self.encoder1(X)
        f2 = self.encoder2(X)
        f = self.attention(f1, f2)
        r = self.resblock(f.view(batch_size * seq_length, -1))
        f = self.gru(f).view(batch_size * seq_length, -1)
        f = torch.cat((f, r), dim=1)
        output = self.classifier(f)
        return output


def init_weight(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)


if __name__ == '__main__':
    X = torch.randn((4, 10, 2, 3000))
    net = SleepNet(2, 0.5)
    print(net(X).shape)
    torch.save(net.state_dict(), 'MSleepNet.pth')
