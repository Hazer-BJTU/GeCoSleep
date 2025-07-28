import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSleepNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=5, fs=100, window_size=30, dropout=0.5):
        super(DeepSleepNet, self).__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.fs = fs
        self.window_size = window_size
        self.dropout_p = dropout

        self.cnn_small = self._build_cnn(filter_size=fs // 2, stride_size=fs // 16)
        self.cnn_large = self._build_cnn(filter_size=fs * 4, stride_size=fs // 2)

        self.bilstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=2,
                              bidirectional=True, batch_first=True)
        self.fc_residual = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.fc_out = nn.Linear(1024, num_classes)

        self.dropout = nn.Dropout(self.dropout_p)

    def _build_cnn(self, filter_size, stride_size):
        return nn.Sequential(
            nn.Conv1d(self.num_channels, 64, kernel_size=filter_size, stride=stride_size, padding=filter_size // 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(self.dropout_p),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        batch_size, window_size, num_channels, len_series = x.size()
        x = x.view(batch_size * window_size, num_channels, len_series)
        h_small = self.cnn_small(x)  # (batch*window, 512)
        h_large = self.cnn_large(x)  # (batch*window, 512)
        a = torch.cat((h_small, h_large), dim=1)  # (batch*window, 1024)
        a_seq = a.view(batch_size, window_size, -1)  # (batch, window, 1024)
        lstm_out, _ = self.bilstm(a_seq)  # (batch, window, 1024)
        lstm_out = lstm_out.contiguous().view(batch_size * window_size, -1)  # (batch*window, 1024)
        a = a.view(batch_size * window_size, -1)
        residual = self.fc_residual(a)  # (batch*window, 1024)
        out = lstm_out + residual
        out = self.dropout(out)
        logits = self.fc_out(out)  # (batch*window, num_classes)
        return logits


if __name__ == "__main__":
    '''
    batch_size = 32
    window_size = 10
    num_channels = 2
    len_series = 100 * 30
    num_classes = 5

    model = DeepSleepNet(num_channels=num_channels, num_classes=num_classes)

    x = torch.randn(batch_size, window_size, num_channels, len_series)

    logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    '''
