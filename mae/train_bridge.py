"""Stage 3: Train the projection bridge from WM patch embeddings to MAE latent space.

Uses patch-level WM encoder outputs [B, 272, 192] (not mean-pooled) to preserve
spatial information. A lightweight per-patch transformer maps these to MAE-compatible
tokens, then the frozen MAE decoder renders pixels.

Usage:
    python mae/train_bridge.py
    python mae/train_bridge.py --epochs 30 --bs 1024
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent))
from mae import sinusoidal_pos_embed_2d, TransformerBlock, build_mae


# ---------------------------------------------------------------------------
# Bridge architecture: patch-level transformer
# ---------------------------------------------------------------------------

class LatentBridge(nn.Module):
    """Maps WM patch tokens to MAE-compatible patch tokens.

    Per-patch projection + lightweight self-attention refinement.
    Spatial information is preserved since input is already 272 tokens.

    Input:  [B, 272, 192]  -- WM encoder patch tokens (pre-pooling)
    Output: [B, 272, 384]  -- MAE-compatible tokens for frozen decoder
    """

    def __init__(self, wm_dim=192, mae_dim=384, num_patches=272,
                 n_freq=16, n_time=17,
                 depth=4, heads=8, dim_head=32, mlp_dim=1024):
        super().__init__()

        # Project WM patch dim to MAE dim
        self.input_proj = nn.Linear(wm_dim, mae_dim)

        # Sinusoidal positional embeddings
        pos = sinusoidal_pos_embed_2d(n_freq, n_time, mae_dim)
        self.register_buffer("pos_embed", pos.unsqueeze(0))

        # Self-attention transformer to refine tokens
        self.blocks = nn.ModuleList([
            TransformerBlock(mae_dim, heads, dim_head, mlp_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(mae_dim)

    def forward(self, x):
        """x: (B, 272, wm_dim) -> (B, 272, mae_dim)"""
        x = self.input_proj(x)  # (B, 272, mae_dim)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Dataset: paired (wm_embedding, logmag_frame)
# ---------------------------------------------------------------------------

class PairedPatchDataset(Dataset):
    """Paired dataset: WM patch embeddings + log-magnitude frame.

    Flattens trajectory dimension: [N, 16, ...] -> [N*16, ...]
    """

    def __init__(self, patch_h5_path, logmag_h5_path, vmin, vmax):
        with h5py.File(patch_h5_path, "r") as f:
            patches = f["patch_embeddings"][()]  # [N, 16, 272, 192]
        with h5py.File(logmag_h5_path, "r") as f:
            logmag = f["logmag"][()]  # [N, 16, 256, 51]

        N, T = patches.shape[:2]
        self.patches = patches.reshape(N * T, 272, 192).astype(np.float32)  # [N*16, 272, 192]
        logmag_flat = logmag.reshape(N * T, 256, 51).astype(np.float32)

        # Normalize to [0, 1]
        self.vmin = vmin
        self.vmax = vmax
        scale = max(vmax - vmin, 1e-8)
        self.frames = np.clip((logmag_flat - vmin) / scale, 0.0, 1.0)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patches = torch.from_numpy(self.patches[idx])         # (272, 192)
        frame = torch.from_numpy(self.frames[idx]).unsqueeze(0)  # (1, 256, 51)
        return patches, frame


# ---------------------------------------------------------------------------
# SSIM helpers
# ---------------------------------------------------------------------------

def compute_bridge_ssim(bridge, mae_decoder, dataloader, device):
    """SSIM for bridge -> MAE decoder reconstruction vs ground truth."""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    bridge.eval()
    with torch.no_grad():
        for emb_batch, frame_batch in dataloader:
            emb_batch = emb_batch.to(device)
            frame_batch = frame_batch.to(device)

            tokens = bridge(emb_batch)  # (B, 272, 384)
            _, recon = mae_decoder(tokens, visible_indices=None)  # (B, 256, 51)
            recon = recon.unsqueeze(1).clamp(0, 1)
            ssim_metric.update(recon, frame_batch)

    return ssim_metric.compute().item()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    mae_dir = Path(args.mae_dir)

    # Load norm stats
    norm_path = mae_dir / "cache" / "norm_stats.json"
    stats = json.load(open(norm_path))
    vmin, vmax = stats["min"], stats["max"]
    print(f"Norm stats: min={vmin:.4f}, max={vmax:.4f}")

    # Load frozen MAE (we use its decoder for rendering)
    mae_model = build_mae().to(device)
    mae_ckpt = mae_dir / "mae_best.ckpt"
    mae_model.load_state_dict(torch.load(mae_ckpt, map_location=device, weights_only=True))
    mae_model.requires_grad_(False)
    mae_decoder = mae_model.decoder  # frozen, but gradients flow through for bridge training
    print(f"Loaded frozen MAE decoder from {mae_ckpt}")

    # Datasets (patch-level embeddings, not mean-pooled)
    print("Loading paired datasets...")
    train_ds = PairedPatchDataset(
        str(cache_dir / "patch_embeddings_train.h5"),
        str(cache_dir / "logmag_train.h5"),
        vmin, vmax,
    )
    val_ds = PairedPatchDataset(
        str(cache_dir / "patch_embeddings_val.h5"),
        str(cache_dir / "logmag_val.h5"),
        vmin, vmax,
    )
    print(f"Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_loader = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Bridge (patch-level projection + refinement -> frozen MAE decoder)
    bridge = LatentBridge(
        wm_dim=192, mae_dim=384, num_patches=272,
        n_freq=16, n_time=17,
        depth=4, heads=8, dim_head=32, mlp_dim=1024,
    ).to(device)
    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"Bridge parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 3 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_ssim = 0.0
    best_epoch = -1
    step = 0

    for epoch in range(args.epochs):
        bridge.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for patch_batch, frame_batch in train_loader:
            patch_batch = patch_batch.to(device)  # (B, 272, 192)
            frame_batch = frame_batch.to(device)
            gt_spec = frame_batch.squeeze(1)  # (B, 256, 51)

            # Bridge: WM patches -> MAE tokens -> frozen MAE decoder -> spectrogram
            tokens = bridge(patch_batch)  # (B, 272, 384)
            _, pred_spec = mae_decoder(tokens, visible_indices=None)  # (B, 256, 51)
            loss = nn.functional.mse_loss(pred_spec, gt_spec)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{args.epochs} | recon_mse={avg_loss:.6f} | lr={lr_now:.2e} | {dt:.1f}s")

        # Periodic SSIM check
        if (epoch + 1) % args.check_every == 0 or epoch == args.epochs - 1:
            bridge_ssim = compute_bridge_ssim(bridge, mae_decoder, val_loader, device)
            print(f"  Bridge SSIM: {bridge_ssim:.4f}")

            if bridge_ssim > best_ssim:
                best_ssim = bridge_ssim
                best_epoch = epoch + 1
                ckpt_path = mae_dir / "bridge_best.ckpt"
                torch.save(bridge.state_dict(), ckpt_path)
                print(f"  -> New best! Saved to {ckpt_path}")

    # Final summary
    print()
    print("=== BRIDGE TRAINING SUMMARY ===")
    print(f"Val SSIM (bridge direct):   {best_ssim:.3f}  (epoch {best_epoch})")

    if best_ssim < 0.50:
        print(f"\nWARNING: SSIM {best_ssim:.3f} < 0.50 target.")
    else:
        print(f"\nPASS: SSIM {best_ssim:.3f} >= 0.50 target.")


def main():
    parser = argparse.ArgumentParser(description="Train projection bridge: WM -> MAE latent space")
    parser.add_argument("--cache_dir", default="decoder/cache")
    parser.add_argument("--mae_dir", default="mae")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--recon_weight", type=float, default=0.3)
    parser.add_argument("--check_every", type=int, default=5, help="SSIM check interval (epochs)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
