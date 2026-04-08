"""Patch-level spectrogram decoder.

Reconstructs spectrograms from the 272 pre-pooled patch tokens (each 192-dim),
NOT from the single 192-dim pooled embedding. Each patch independently
reconstructs its own 16x3 spatial region.

Architecture:
  Input: (B, n_freq=16, n_time=17, 192) — patch tokens in spatial layout
  Per-patch MLP: 192 -> 256 -> patch_freq * patch_time (=48)
  Reshape each patch output to (patch_freq, patch_time) = (16, 3)
  Assemble grid: (16*16, 17*3) = (256, 51)
  Output: (B, 256, 51)
"""

import torch
from torch import nn
from einops import rearrange


class PatchDecoder(nn.Module):
    def __init__(self, hidden_dim=192, patch_freq=16, patch_time=3):
        super().__init__()
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        patch_pixels = patch_freq * patch_time  # 48

        self.patch_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, patch_pixels),
        )

    def forward(self, patch_tokens):
        """
        patch_tokens: (B, n_freq, n_time, hidden_dim) e.g. (B, 16, 17, 192)
        returns: (B, freq_bins, time_bins) e.g. (B, 256, 51)
        """
        B, nf, nt, D = patch_tokens.shape

        # Flatten spatial dims, apply MLP per patch
        flat = rearrange(patch_tokens, "b f t d -> (b f t) d")
        pixels = self.patch_mlp(flat)  # (B*nf*nt, patch_freq*patch_time)
        pixels = rearrange(pixels, "(b f t) (pf pt) -> b (f pf) (t pt)",
                           b=B, f=nf, t=nt, pf=self.patch_freq, pt=self.patch_time)
        return pixels  # (B, 256, 51)
