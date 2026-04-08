"""Spectrogram decoder: embedding (192-dim) -> log-magnitude spectrogram (256x51)."""

import torch
from torch import nn


class SpectrogramDecoder(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.GELU(),
            nn.Linear(512, 8 * 8 * 8), nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(8, 64, 4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.BatchNorm2d(64), nn.GELU(),
            nn.ConvTranspose2d(64, 1, 4, stride=(4, 1), padding=(0, 0)),  # 64x64 -> 256x64
        )

    def forward(self, x):
        """x: (B, 192) -> (B, 256, 51)"""
        x = self.fc(x)
        x = x.view(-1, 8, 8, 8)
        x = self.deconv(x)  # (B, 1, 256, 64)
        x = x[:, 0, :, :51]  # crop to 256x51
        return x
