"""Compact classifiers with the public [batch,128,6] -> [batch,6] interface."""

import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        first, second = channels
        self.layers = nn.Sequential(
            nn.Conv1d(6, first, kernel_size=5, padding=2),
            nn.BatchNorm1d(first), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(first, second, kernel_size=5, padding=2),
            nn.BatchNorm1d(second), nn.ReLU(), nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2))


class CNN(nn.Module):
    def __init__(self, channels=(32, 64), dropout=0.2):
        super().__init__()
        self.encoder = ConvEncoder(channels)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1] * 2, 6)

    def forward(self, x):
        z = self.encoder(x)
        pooled = torch.cat((z.mean(dim=2), z.amax(dim=2)), dim=1)
        return self.head(self.dropout(pooled))


class GRU(nn.Module):
    def __init__(self, hidden=64, dropout=0.2):
        super().__init__()
        self.encoder = nn.GRU(6, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 6)

    def forward(self, x):
        z, _ = self.encoder(x)
        return self.head(self.dropout(z.mean(dim=1)))


class CNNGRU(nn.Module):
    def __init__(self, channels=(32, 64), hidden=64, dropout=0.2):
        super().__init__()
        self.conv = ConvEncoder(channels)
        self.encoder = nn.GRU(channels[-1], hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 6)

    def forward(self, x):
        z, _ = self.encoder(self.conv(x).transpose(1, 2))
        return self.head(self.dropout(z.mean(dim=1)))


def build_model(name: str, config: dict) -> nn.Module:
    if name == "cnn":
        return CNN(config["conv_channels"], config["dropout"])
    if name == "gru":
        return GRU(config["gru_hidden"], config["dropout"])
    if name == "cnn_gru":
        return CNNGRU(config["conv_channels"], config["gru_hidden"], config["dropout"])
    raise ValueError(f"Unknown neural model: {name}")
