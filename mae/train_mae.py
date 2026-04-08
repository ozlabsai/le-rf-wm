"""Stage 2: Train the Masked Autoencoder on cached log-magnitude spectrograms.

Usage:
    python mae/train_mae.py                         # defaults
    python mae/train_mae.py --epochs 100 --bs 512   # override
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

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))
from mae import build_mae


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LogMagFrameDataset(Dataset):
    """Single-frame dataset from cached logmag HDF5.

    Each sample is one frame [256, 51] reshaped to [1, 256, 51].
    Normalization to [0, 1] applied using global min/max.
    """

    def __init__(self, h5_path, vmin, vmax):
        with h5py.File(h5_path, "r") as f:
            # [N, 16, 256, 51] -> flatten to [N*16, 256, 51]
            data = f["logmag"][()]
        N, T = data.shape[:2]
        self.data = data.reshape(N * T, 256, 51).astype(np.float32)
        self.vmin = vmin
        self.vmax = vmax
        self.scale = max(vmax - vmin, 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]  # (256, 51)
        frame = (frame - self.vmin) / self.scale  # normalize to [0, 1]
        frame = np.clip(frame, 0.0, 1.0)
        return torch.from_numpy(frame).unsqueeze(0)  # (1, 256, 51)


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def compute_norm_stats(h5_path):
    """Compute global min/max from train split logmag cache."""
    with h5py.File(h5_path, "r") as f:
        data = f["logmag"]
        # Stream in chunks to avoid loading all into RAM
        N = data.shape[0]
        chunk = 500
        global_min = float("inf")
        global_max = float("-inf")
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            block = data[start:end]
            global_min = min(global_min, float(block.min()))
            global_max = max(global_max, float(block.max()))
    return global_min, global_max


# ---------------------------------------------------------------------------
# SSIM assessment
# ---------------------------------------------------------------------------

def compute_ssim(model, dataloader, device):
    """Compute SSIM on full reconstructions (no masking)."""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)  # (B, 1, 256, 51)
            recon = model.reconstruct(batch)  # (B, 256, 51)
            recon = recon.unsqueeze(1).clamp(0, 1)  # (B, 1, 256, 51)
            ssim_metric.update(recon, batch)

    return ssim_metric.compute().item()


def compute_masked_mse(model, dataloader, device, mask_ratio=0.75):
    """Compute masked MSE on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            loss, _, _ = model(batch, mask_ratio=mask_ratio)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    mae_dir = Path(args.mae_dir)
    mae_dir.mkdir(parents=True, exist_ok=True)
    (mae_dir / "cache").mkdir(parents=True, exist_ok=True)

    # --- Norm stats ---
    norm_path = mae_dir / "cache" / "norm_stats.json"
    train_h5 = cache_dir / "logmag_train.h5"
    val_h5 = cache_dir / "logmag_val.h5"

    if norm_path.exists():
        stats = json.load(open(norm_path))
        vmin, vmax = stats["min"], stats["max"]
        print(f"Loaded norm stats: min={vmin:.4f}, max={vmax:.4f}")
    else:
        print("Computing normalization stats from train split...")
        vmin, vmax = compute_norm_stats(str(train_h5))
        json.dump({"min": vmin, "max": vmax}, open(norm_path, "w"))
        print(f"Saved norm stats: min={vmin:.4f}, max={vmax:.4f}")

    # --- Datasets ---
    print("Loading datasets...")
    train_ds = LogMagFrameDataset(str(train_h5), vmin, vmax)
    val_ds = LogMagFrameDataset(str(val_h5), vmin, vmax)
    print(f"Train: {len(train_ds)} frames, Val: {len(val_ds)} frames")

    train_loader = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # --- Model ---
    model = build_mae().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MAE parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Cosine decay with linear warmup
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- Training ---
    best_ssim = 0.0
    best_epoch = -1
    step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(device)

            # Primary: masked reconstruction
            loss, _, _ = model(batch, mask_ratio=args.mask_ratio)

            # Every 4th step: add full-reconstruction loss (no masking)
            # This directly trains the decode path used by reconstruct()
            if step % 4 == 0:
                recon = model.reconstruct(batch).unsqueeze(1)  # (B, 1, 256, 51)
                recon_loss = nn.functional.mse_loss(recon, batch)
                loss = loss + 0.5 * recon_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={avg_loss:.5f} | lr={lr_now:.2e} | {dt:.1f}s")

        # --- Periodic SSIM check ---
        if (epoch + 1) % args.check_every == 0 or epoch == args.epochs - 1:
            val_mse = compute_masked_mse(model, val_loader, device, mask_ratio=args.mask_ratio)
            val_ssim = compute_ssim(model, val_loader, device)
            print(f"  Val MSE (masked): {val_mse:.5f} | Val SSIM (full recon): {val_ssim:.4f}")

            if val_ssim > best_ssim:
                best_ssim = val_ssim
                best_epoch = epoch + 1
                ckpt_path = mae_dir / "mae_best.ckpt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  -> New best SSIM! Saved to {ckpt_path}")

    # --- Final assessment ---
    final_mse = compute_masked_mse(model, val_loader, device, mask_ratio=args.mask_ratio)
    final_ssim = compute_ssim(model, val_loader, device)

    print()
    print("=== MAE TRAINING SUMMARY ===")
    status = "PASS" if best_ssim >= 0.70 else "FAIL"
    print(f"Val SSIM (best):        {best_ssim:.3f}  [{status} vs 0.70]  (epoch {best_epoch})")
    print(f"Val SSIM (final):       {final_ssim:.3f}")
    print(f"Val MSE (masked):       {final_mse:.5f}")
    print(f"Norm stats:             min={vmin:.2f} max={vmax:.2f}")

    if best_ssim < 0.70:
        print("\nWARNING: SSIM < 0.70 — reconstruction quality is below target.")
        print("Proceeding to Stage 3 anyway (bridge may still be useful).")


def main():
    parser = argparse.ArgumentParser(description="Train MAE on cached log-magnitude spectrograms")
    parser.add_argument("--cache_dir", default="decoder/cache", help="Directory with logmag HDF5 files")
    parser.add_argument("--mae_dir", default="mae", help="Output directory for MAE artifacts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--mask_ratio", type=float, default=0.60)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--check_every", type=int, default=5, help="SSIM check interval (epochs)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
