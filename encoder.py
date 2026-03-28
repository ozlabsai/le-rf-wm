"""Spectrogram Vision Transformer for RF data."""

import torch
from torch import nn
from einops import rearrange

from module import Block


class SpectrogramViT(nn.Module):
    """ViT encoder for 2D spectrograms with rectangular patches
    and separate frequency/time positional embeddings."""

    def __init__(
        self,
        in_channels=2,
        freq_bins=256,
        time_bins=51,
        patch_freq=16,
        patch_time=3,
        hidden_dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()

        assert freq_bins % patch_freq == 0, f"freq_bins {freq_bins} not divisible by patch_freq {patch_freq}"
        assert time_bins % patch_time == 0, f"time_bins {time_bins} not divisible by patch_time {patch_time}"

        self.n_freq = freq_bins // patch_freq
        self.n_time = time_bins // patch_time
        self.num_patches = self.n_freq * self.n_time

        # patch embedding: Conv2d with rectangular kernel
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=(patch_freq, patch_time),
            stride=(patch_freq, patch_time),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # separate freq and time positional embeddings, broadcast-summed
        self.freq_pos = nn.Parameter(torch.randn(1, self.n_freq, 1, hidden_dim))
        self.time_pos = nn.Parameter(torch.randn(1, 1, self.n_time, hidden_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # transformer body
        self.blocks = nn.ModuleList([
            Block(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, C, F, T) — e.g. (B, 2, 256, 51)
        returns: (B, hidden_dim) — CLS token embedding
        """
        B = x.size(0)

        # patch embed → (B, hidden_dim, n_freq, n_time)
        x = self.patch_embed(x)
        # → (B, n_freq, n_time, hidden_dim)
        x = rearrange(x, "b d f t -> b f t d")

        # add factored positional embeddings
        x = x + self.freq_pos + self.time_pos

        # flatten spatial dims → (B, num_patches, hidden_dim)
        x = rearrange(x, "b f t d -> b (f t) d")

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos
        x = torch.cat([cls, x], dim=1)

        x = self.dropout(x)

        # transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # return CLS token
        return x[:, 0]
