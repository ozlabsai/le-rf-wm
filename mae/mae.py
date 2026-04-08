"""Masked Autoencoder for RF spectrograms.

Architecture matches the world model's 272-patch grid exactly:
  Conv2d(1, 256, kernel_size=(16,3), stride=(16,3)) -> 16 freq x 17 time = 272 patches.

Encoder: 6-layer transformer, 256-dim, 8 heads
Decoder: 4-layer transformer, 128-dim, 4 heads
Total: ~10M params
"""

import math
import torch
from torch import nn
from einops import rearrange


# ---------------------------------------------------------------------------
# Sinusoidal positional embeddings (2D, for freq x time patch grid)
# ---------------------------------------------------------------------------

def sinusoidal_pos_embed_2d(n_freq: int, n_time: int, dim: int) -> torch.Tensor:
    """Generate 2D sinusoidal positional embeddings.

    Returns: (n_freq * n_time, dim) — one embedding per patch, row-major order.
    Half the channels encode frequency position, half encode time position.
    """
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sinusoidal embeddings"
    d = dim // 4

    freq_pos = torch.arange(n_freq, dtype=torch.float32).unsqueeze(1)  # (n_freq, 1)
    time_pos = torch.arange(n_time, dtype=torch.float32).unsqueeze(1)  # (n_time, 1)
    omega = 1.0 / (10000.0 ** (torch.arange(d, dtype=torch.float32) / d))  # (d,)

    freq_sin = torch.sin(freq_pos * omega)  # (n_freq, d)
    freq_cos = torch.cos(freq_pos * omega)  # (n_freq, d)
    time_sin = torch.sin(time_pos * omega)  # (n_time, d)
    time_cos = torch.cos(time_pos * omega)  # (n_time, d)

    # broadcast: each freq position paired with each time position
    # freq part: (n_freq, 1, 2d) broadcast to (n_freq, n_time, 2d)
    freq_emb = torch.cat([freq_sin, freq_cos], dim=-1).unsqueeze(1).expand(-1, n_time, -1)
    time_emb = torch.cat([time_sin, time_cos], dim=-1).unsqueeze(0).expand(n_freq, -1, -1)

    pos = torch.cat([freq_emb, time_emb], dim=-1)  # (n_freq, n_time, dim)
    return pos.reshape(n_freq * n_time, dim)


# ---------------------------------------------------------------------------
# Transformer building blocks (lightweight, self-contained)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        B, N, _ = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x


# ---------------------------------------------------------------------------
# MAE Encoder
# ---------------------------------------------------------------------------

class SpectrogramMAEEncoder(nn.Module):
    """Encodes log-magnitude spectrograms [B, 1, 256, 51] into patch tokens.

    Patch grid: 16 freq x 17 time = 272 patches (matches world model exactly).
    """

    def __init__(
        self,
        in_channels=1,
        freq_bins=256,
        time_bins=51,
        patch_freq=16,
        patch_time=3,
        hidden_dim=256,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dim_head=32,
        dropout=0.0,
    ):
        super().__init__()
        self.n_freq = freq_bins // patch_freq   # 16
        self.n_time = time_bins // patch_time   # 17
        self.num_patches = self.n_freq * self.n_time  # 272
        self.hidden_dim = hidden_dim
        self.patch_freq = patch_freq
        self.patch_time = patch_time

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=(patch_freq, patch_time),
            stride=(patch_freq, patch_time),
        )

        # Sinusoidal positional embeddings (fixed, not learned)
        pos_embed = sinusoidal_pos_embed_2d(self.n_freq, self.n_time, hidden_dim)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, 272, dim)

        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask_indices=None):
        """Encode with optional masking.

        Args:
            x: (B, 1, 256, 51) log-magnitude spectrogram
            mask_indices: (B, num_visible) indices of visible patches, or None for full encoding

        Returns:
            tokens: (B, N, hidden_dim) where N = num_visible or 272
        """
        B = x.shape[0]
        # Patch embed -> (B, hidden_dim, n_freq, n_time) -> (B, 272, hidden_dim)
        tokens = self.patch_embed(x)
        tokens = rearrange(tokens, "b d f t -> b (f t) d")

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Apply masking: keep only visible patches
        if mask_indices is not None:
            # mask_indices: (B, num_visible)
            idx = mask_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            tokens = tokens.gather(1, idx)
            # Positional embeddings already added before gathering

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens


# ---------------------------------------------------------------------------
# MAE Decoder
# ---------------------------------------------------------------------------

class SpectrogramMAEDecoder(nn.Module):
    """Decodes patch token sequence back to spectrogram pixels.

    Input: (B, 272, 256) full or masked+unmasked token sequence
    Output: (B, 256, 51) reconstructed log-magnitude spectrogram
    """

    def __init__(
        self,
        encoder_dim=256,
        decoder_dim=128,
        num_patches=272,
        n_freq=16,
        n_time=17,
        patch_freq=16,
        patch_time=3,
        depth=4,
        heads=4,
        mlp_dim=512,
        dim_head=32,
        dropout=0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.n_freq = n_freq
        self.n_time = n_time
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        self.patch_pixels = patch_freq * patch_time  # 48

        # Project from encoder dim to decoder dim
        self.embed_proj = nn.Linear(encoder_dim, decoder_dim)

        # Mask token (learnable, replaces masked patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Sinusoidal positional embeddings for decoder
        pos_embed = sinusoidal_pos_embed_2d(n_freq, n_time, decoder_dim)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))

        # Transformer decoder
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        # Predict pixel values per patch
        self.head = nn.Linear(decoder_dim, self.patch_pixels)

    def forward(self, visible_tokens, visible_indices=None, num_patches=None):
        """Decode from visible tokens to full spectrogram.

        Args:
            visible_tokens: (B, N_vis, encoder_dim) — encoded visible patches
            visible_indices: (B, N_vis) — which patch indices are visible. None = all visible.
            num_patches: total number of patches (default: self.num_patches)

        Returns:
            pred_pixels: (B, num_patches, patch_pixels) — per-patch pixel predictions
            recon: (B, 256, 51) — reassembled spectrogram
        """
        B = visible_tokens.shape[0]
        N = num_patches or self.num_patches

        # Project to decoder dim
        visible_tokens = self.embed_proj(visible_tokens)

        if visible_indices is not None:
            # Build full sequence: mask tokens everywhere, then scatter visible tokens
            full_tokens = self.mask_token.expand(B, N, -1).clone()
            idx = visible_indices.unsqueeze(-1).expand(-1, -1, visible_tokens.shape[-1])
            full_tokens.scatter_(1, idx, visible_tokens)
        else:
            full_tokens = visible_tokens

        # Add positional embeddings
        full_tokens = full_tokens + self.pos_embed[:, :N]

        for block in self.blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.norm(full_tokens)

        # Predict pixels for each patch
        pred_pixels = self.head(full_tokens)  # (B, N, 48)

        # Reassemble into spectrogram
        recon = rearrange(
            pred_pixels, "b (f t) (pf pt) -> b (f pf) (t pt)",
            f=self.n_freq, t=self.n_time, pf=self.patch_freq, pt=self.patch_time,
        )
        return pred_pixels, recon


# ---------------------------------------------------------------------------
# Full MAE
# ---------------------------------------------------------------------------

class SpectrogramMAE(nn.Module):
    """Masked Autoencoder for RF spectrograms.

    Wraps encoder + decoder with masking logic.
    """

    def __init__(
        self,
        encoder_kwargs=None,
        decoder_kwargs=None,
    ):
        super().__init__()
        enc_kw = encoder_kwargs or {}
        dec_kw = decoder_kwargs or {}

        self.encoder = SpectrogramMAEEncoder(**enc_kw)
        self.decoder = SpectrogramMAEDecoder(**dec_kw)
        self.num_patches = self.encoder.num_patches

    def random_masking(self, B, mask_ratio, device):
        """Generate random mask indices.

        Returns:
            visible_indices: (B, num_visible) — sorted patch indices to keep
            masked_indices: (B, num_masked) — sorted patch indices that are masked
            restore_indices: (B, num_patches) — indices to unshuffle back to original order
        """
        N = self.num_patches
        num_visible = max(1, int(N * (1 - mask_ratio)))
        num_masked = N - num_visible

        # Random permutation per sample
        noise = torch.rand(B, N, device=device)
        shuffle_indices = noise.argsort(dim=1)

        visible_indices = shuffle_indices[:, :num_visible].sort(dim=1).values
        masked_indices = shuffle_indices[:, num_visible:].sort(dim=1).values

        return visible_indices, masked_indices

    def forward(self, x, mask_ratio=0.75):
        """MAE forward pass with random masking.

        Args:
            x: (B, 1, 256, 51) log-magnitude spectrogram (normalized [0,1])
            mask_ratio: fraction of patches to mask

        Returns:
            loss: MSE loss on masked patches only
            pred_pixels: (B, num_patches, 48) per-patch predictions
            masked_indices: (B, num_masked) which patches were masked
        """
        B = x.shape[0]
        device = x.device

        # Ground truth per-patch pixels
        gt_pixels = rearrange(
            x.squeeze(1), "b (f pf) (t pt) -> b (f t) (pf pt)",
            f=self.encoder.n_freq, t=self.encoder.n_time,
            pf=self.encoder.patch_freq, pt=self.encoder.patch_time,
        )  # (B, 272, 48)

        # Generate mask
        visible_indices, masked_indices = self.random_masking(B, mask_ratio, device)

        # Encode visible patches only
        visible_tokens = self.encoder(x, mask_indices=visible_indices)

        # Decode full sequence
        pred_pixels, recon = self.decoder(visible_tokens, visible_indices)

        # Loss on masked patches only
        masked_idx_expanded = masked_indices.unsqueeze(-1).expand(-1, -1, gt_pixels.shape[-1])
        masked_pred = pred_pixels.gather(1, masked_idx_expanded)
        masked_gt = gt_pixels.gather(1, masked_idx_expanded)
        loss = nn.functional.mse_loss(masked_pred, masked_gt)

        return loss, pred_pixels, masked_indices

    def encode(self, x):
        """Full forward pass through encoder, no masking.

        Args:
            x: (B, 1, 256, 51)
        Returns:
            tokens: (B, 272, 256) patch token sequence
        """
        return self.encoder(x, mask_indices=None)

    def decode(self, tokens):
        """Decoder only, from full token sequence.

        Args:
            tokens: (B, 272, 256) patch tokens
        Returns:
            recon: (B, 256, 51) reconstructed spectrogram
        """
        _, recon = self.decoder(tokens, visible_indices=None)
        return recon

    def reconstruct(self, x):
        """Full encode→decode, no masking.

        Args:
            x: (B, 1, 256, 51)
        Returns:
            recon: (B, 256, 51)
        """
        tokens = self.encode(x)
        return self.decode(tokens)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_mae():
    """Build MAE with default hyperparameters matching the spec."""
    return SpectrogramMAE(
        encoder_kwargs=dict(
            in_channels=1, freq_bins=256, time_bins=51,
            patch_freq=16, patch_time=3,
            hidden_dim=256, depth=6, heads=8, mlp_dim=1024, dim_head=32,
        ),
        decoder_kwargs=dict(
            encoder_dim=256, decoder_dim=128, num_patches=272,
            n_freq=16, n_time=17, patch_freq=16, patch_time=3,
            depth=4, heads=4, mlp_dim=512, dim_head=32,
        ),
    )


if __name__ == "__main__":
    model = build_mae()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MAE parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    x = torch.randn(4, 1, 256, 51)
    loss, pred, masked = model(x, mask_ratio=0.75)
    print(f"Forward pass — loss: {loss.item():.4f}, pred: {pred.shape}, masked: {masked.shape}")

    tokens = model.encode(x)
    print(f"Encode — tokens: {tokens.shape}")

    recon = model.reconstruct(x)
    print(f"Reconstruct — recon: {recon.shape}")
