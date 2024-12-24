import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
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


class VAEencoder(nn.Module):
    def __init__(self, embeddings, hiddens, heads, layers, dropout, **kwargs):
        super(VAEencoder, self).__init__(**kwargs)
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


class VAEdecoder(nn.Module):
    def __init__(self, embeddings, heads, layers, dropout, **kwargs):
        super(VAEdecoder, self).__init__(**kwargs)
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
                        dtype=torch.float32, requires_grad=False, device=y.deivce)
        X_hat = self.forward(z, y)
        return X_hat


class VAE(nn.Module):
    def __init__(self, embeddings, dropout, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.hiddens = int(embeddings * 1.5)
        self.dropout = dropout
        self.encoder = VAEencoder(embeddings, self.hiddens, 8, 2, dropout)
        self.decoder = VAEdecoder(embeddings, 8, 1, dropout)

    def forward(self, X, y):
        mu, sigma = self.encoder(X, y)
        eps = torch.randn_like(mu, requires_grad=False)
        z = eps * sigma + mu
        X_hat = self.decoder(z, y)
        kl_loss = torch.mean(-2 * torch.log(sigma) + sigma.pow(2) + mu.pow(2) - 1) * 0.5
        return X_hat, kl_loss


if __name__ == '__main__':
    net = VAE(512, 0.1)
    X = torch.randn((4, 10, 512), dtype=torch.float32, requires_grad=False)
    y = torch.randint(0, 5, (4, 10), dtype=torch.int64, requires_grad=False)
    X_hat, kl_loss = net(X, y)
    print(X_hat.shape, kl_loss)
    torch.save(net.state_dict(), 'VAE.pth')
