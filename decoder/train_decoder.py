"""Stage 3: Train the post-hoc spectrogram decoder.

v2: Normalized input range [0,1], signal-weighted MSE + SSIM loss.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from decoder import SpectrogramDecoder

CACHE = Path(__file__).parent / "cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH = 512
LR_MAX = 1e-3
LR_MIN = 1e-5
SSIM_THRESHOLD = 0.50  # revised threshold after normalization fix
SIGNAL_THRESHOLD = 0.4  # in [0,1] normalized space — roughly top 30% power
SIGNAL_WEIGHT = 10.0    # upweight signal-bearing regions


class EmbLogmagDataset(Dataset):
    """Paired (embedding, logmag_frame) dataset — normalized to [0,1]."""
    def __init__(self, split, lm_min, lm_range):
        with h5py.File(CACHE / f"embeddings_{split}.h5", "r") as f:
            self.emb = torch.from_numpy(f["embeddings"][()]).float()
        with h5py.File(CACHE / f"logmag_{split}.h5", "r") as f:
            logmag = torch.from_numpy(f["logmag"][()]).float()
        # Normalize to [0, 1]
        self.logmag = (logmag - lm_min) / lm_range
        self.logmag = self.logmag.clamp(0, 1)
        self.N, self.T = self.emb.shape[:2]

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, idx):
        n, t = divmod(idx, self.T)
        return self.emb[n, t], self.logmag[n, t]


def signal_weighted_mse(pred, target, threshold=SIGNAL_THRESHOLD, weight=SIGNAL_WEIGHT):
    """MSE with higher weight on signal-bearing regions (high power)."""
    signal_mask = (target > threshold).float()
    w = 1.0 + (weight - 1.0) * signal_mask
    return (w * (pred - target).pow(2)).mean()


def compute_ssim(model, loader):
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    with torch.no_grad():
        for emb, target in loader:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            pred = model(emb).clamp(0, 1)
            ssim_fn.update(pred.unsqueeze(1), target.unsqueeze(1))
    return ssim_fn.compute().item()


def ssim_loss_fn(pred, target):
    """Differentiable SSIM loss (1 - SSIM)."""
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    return 1.0 - ssim_fn(pred.unsqueeze(1), target.unsqueeze(1))


if __name__ == "__main__":
    print("=== Stage 3: Training decoder (v2 — normalized + weighted) ===")

    # Compute normalization range from train set
    with h5py.File(CACHE / "logmag_train.h5", "r") as f:
        subset = f["logmag"][:500].astype(np.float32)
        lm_min = float(subset.min())
        lm_max = float(subset.max())
    lm_range = lm_max - lm_min
    print(f"Log-mag range: [{lm_min:.3f}, {lm_max:.3f}], normalizing to [0,1]")

    # Save normalization params for later use
    norm_params = {"lm_min": lm_min, "lm_max": lm_max, "lm_range": lm_range}
    import json
    with open(Path(__file__).parent / "logmag_norm.json", "w") as f:
        json.dump(norm_params, f)

    train_ds = EmbLogmagDataset("train", lm_min, lm_range)
    val_ds = EmbLogmagDataset("val", lm_min, lm_range)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Check target distribution after normalization
    sample_target = train_ds.logmag[:100].reshape(-1)
    print(f"Normalized target stats: mean={sample_target.mean():.3f}, std={sample_target.std():.3f}")
    print(f"Fraction > {SIGNAL_THRESHOLD}: {(sample_target > SIGNAL_THRESHOLD).float().mean():.3f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    # Embedding stats
    with h5py.File(CACHE / "embeddings_val.h5", "r") as f:
        val_emb_sample = np.array(f["embeddings"][:100])
    emb_norms = np.linalg.norm(val_emb_sample.reshape(-1, 192), axis=1)
    print(f"Emb norms (val): mean={emb_norms.mean():.2f} std={emb_norms.std():.2f}")

    model = SpectrogramDecoder(embed_dim=192).to(DEVICE)
    print(f"Decoder: {sum(p.numel() for p in model.parameters()):,} params on {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=LR_MIN)

    best_ssim = 0.0
    best_path = Path(__file__).parent / "decoder_best.ckpt"

    for epoch in range(EPOCHS):
        model.train(True)
        total_loss = 0
        for emb, target in train_loader:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            pred = model(emb).clamp(0, 1)

            # Combined loss: signal-weighted MSE + SSIM
            mse = signal_weighted_mse(pred, target)
            ssim_l = ssim_loss_fn(pred, target)
            loss = 0.7 * mse + 0.3 * ssim_l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model.train(False)
            ssim = compute_ssim(model, val_loader)
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:>3d}/{EPOCHS}  loss={avg_loss:.5f}  val_SSIM={ssim:.4f}  lr={lr:.2e}")
            if ssim > best_ssim:
                best_ssim = ssim
                torch.save(model.state_dict(), best_path)
            model.train(True)
        else:
            print(f"Epoch {epoch+1:>3d}/{EPOCHS}  loss={avg_loss:.5f}")

    # Final summary
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.train(False)
    final_ssim = compute_ssim(model, val_loader)

    val_loss = 0
    with torch.no_grad():
        for emb, target in val_loader:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            pred = model(emb).clamp(0, 1)
            val_loss += (pred - target).pow(2).mean().item()
    val_mse = val_loss / len(val_loader)

    status = "PASS" if final_ssim >= SSIM_THRESHOLD else "FAIL"
    print(f"\n{'='*45}")
    print(f"=== DECODER SUMMARY (v2) ===")
    print(f"Val SSIM:      {final_ssim:.4f}  [{status} vs {SSIM_THRESHOLD}]")
    print(f"Val MSE:       {val_mse:.5f}")
    print(f"Log-mag range: [{lm_min:.3f}, {lm_max:.3f}]")
    print(f"Emb norms:     mean={emb_norms.mean():.2f} std={emb_norms.std():.2f}")
    print(f"{'='*45}")
    if status == "FAIL":
        print(f"\nWARNING: SSIM {final_ssim:.4f} < {SSIM_THRESHOLD}. Decoder may not be viable.")
        print("Consider using attention proxy (Stage 5) as primary Panel A visualization.")
