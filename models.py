import torch
import torch.nn as nn


class FeatureExtraction(nn.Module):
    def __init__(self, input_channels, dropout, **kwargs):
        super(FeatureExtraction, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.block1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=50, stride=6),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=9, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=9, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=9, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=400, stride=50),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        X = X.view(batch_size * seq_length, num_channels, series)
        y1 = self.block1(X)
        y2 = self.block2(X)
        y = torch.cat((y1, y2), dim=2)
        y = y.view(batch_size, seq_length, -1)
        return self.dropout_layer(y)


class LSTMunit(nn.Module):
    def __init__(self, input_size, hiddens, dropout=0.25, bidirectional=True, **kwargs):
        super(LSTMunit, self).__init__(**kwargs)
        self.input_size = input_size
        self.hiddens = hiddens
        self.block = nn.LSTM(input_size, hiddens, num_layers=2,
                             batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.d = 2 if bidirectional else 1

    def get_initial_states(self, batch_size, device):
        H0 = torch.zeros((self.d * 2, batch_size, self.hiddens), dtype=torch.float32, device=device)
        C0 = torch.zeros((self.d * 2, batch_size, self.hiddens), dtype=torch.float32, device=device)
        return H0, C0

    def forward(self, X):
        batch_size, seq_length, features = X.shape[0], X.shape[1], X.shape[2]
        H0, C0 = self.get_initial_states(batch_size, X.device)
        output, (Hn, Cn) = self.block(X, (H0, C0))
        if not output.is_contiguous():
            output = output.contiguous()
        return output


class DeepSleepNet(nn.Module):
    def __init__(self, dropout, input_channels=2, **kwargs):
        super(DeepSleepNet, self).__init__(**kwargs)
        self.dropout = dropout
        self.input_channels = input_channels
        self.feature_extraction = FeatureExtraction(input_channels, dropout)
        self.lstm = LSTMunit(2688, 256, dropout)
        self.resblock = nn.Linear(2688, 512)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 5)
        )

    def forward(self, X):
        batch_size, seq_length, num_channels, series = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        y1 = self.feature_extraction(X)
        y2 = self.lstm(y1).view(batch_size * seq_length, -1)
        y3 = self.resblock(y1.view(batch_size * seq_length, -1))
        y4 = torch.cat((y2, y3), dim=1)
        return self.classifier(y4)


def init_weight(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    elif hasattr(module, 'weight'):
        nn.init.normal_(module.weight)


if __name__ == '__main__':
    net = DeepSleepNet(0.25)
    X = torch.randn((4, 10, 2, 3000), dtype=torch.float32, requires_grad=False)
    print(net(X).shape)
    torch.save(net.state_dict(), 'deepSleepNet.pth')
