import torch
import torch.nn as nn
from models import CNNencoders, ShortTermEncoder, LongTermEncoder


class LwFSleepNet(nn.Module):
    def __init__(self, input_channels, dropout, num_tasks, enable_multihead, **kwargs):
        super(LwFSleepNet, self).__init__(**kwargs)
        self.input_channels = input_channels
        self.dropout = dropout
        self.enable_multihead = enable_multihead
        self.cnn = CNNencoders(input_channels, dropout)
        self.short_term_encoder = ShortTermEncoder(128, 8, 4, dropout)
        self.long_term_encoder = LongTermEncoder(512, 8, 2, dropout)
        self.resblock = nn.Sequential(
            nn.Linear(512, 768),
            nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.LeakyReLU(0.1), nn.LayerNorm(512)
        )
        self.classifiers = nn.ModuleList()
        if enable_multihead:
            for idx in range(num_tasks):
                classifier = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.1), nn.Dropout(dropout),
                    nn.Linear(512, 5)
                )
                self.classifiers.append(classifier)
        else:
            self.classifiers.append(
                nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.1), nn.Dropout(dropout),
                    nn.Linear(512, 5)
                )
            )

    def forward(self, X, task_idx):
        if not self.enable_multihead:
            task_idx = 0
        batch_size, seq_length, num_channels, series = X.shape
        X = self.cnn(X)
        X = self.short_term_encoder(X)
        X = X.view(batch_size, seq_length, -1)
        r = self.resblock(X.view(batch_size * seq_length, -1))
        X = self.long_term_encoder(X)
        X = torch.cat((r, X), dim=1)
        X = self.classifiers[task_idx](X)
        return X

    def features(self, X):
        batch_size, seq_length, num_channels, series = X.shape
        X = self.cnn(X)
        X = self.short_term_encoder(X)
        X = X.view(batch_size, seq_length, -1)
        return X

    def classify(self, X, task_idx):
        if not self.enable_multihead:
            task_idx = 0
        batch_size, seq_length, embeddings = X.shape
        r = self.resblock(X.view(batch_size * seq_length, -1))
        X = self.long_term_encoder(X)
        X = torch.cat((r, X), dim=1)
        X = self.classifiers[task_idx](X)
        return X

    def freeze_parameters(self):
        for m in self.cnn.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        for name, param in self.cnn.named_parameters():
            param.requires_grad = False
        for name, param in self.short_term_encoder.named_parameters():
            param.requires_grad = False


if __name__ == '__main__':
    net = LwFSleepNet(2, 0.15, 4)
    for name, module in net.named_modules():
        print(name)
