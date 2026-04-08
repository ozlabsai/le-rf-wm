"""Stage 3: Train the post-hoc spectrogram decoder."""

import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from decoder import SpectrogramDecoder

CACHE = Path(__file__).parent / "cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH = 512
LR_MAX = 1e-3
LR_MIN = 1e-5
SSIM_THRESHOLD = 0.70


class EmbLogmagDataset(Dataset):
    def __init__(self, split):
        with h5py.File(CACHE / f"embeddings_{split}.h5", "r") as f:
            self.emb = torch.from_numpy(f["embeddings"][()]).float()
        with h5py.File(CACHE / f"logmag_{split}.h5", "r") as f:
            self.logmag = torch.from_numpy(f["logmag"][()]).float()
        self.N, self.T = self.emb.shape[:2]

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, idx):
        n, t = divmod(idx, self.T)
        return self.emb[n, t], self.logmag[n, t]


def compute_ssim(model, loader, data_range):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=data_range).to(DEVICE)
    with torch.no_grad():
        for emb, target in loader:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            pred = model(emb)
            ssim_fn.update(pred.unsqueeze(1), target.unsqueeze(1))
    return ssim_fn.compute().item()


if __name__ == "__main__":
    print("=== Stage 3: Training decoder ===")

    train_ds = EmbLogmagDataset("train")
    val_ds = EmbLogmagDataset("val")
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    # Log-magnitude range
    with h5py.File(CACHE / "logmag_train.h5", "r") as f:
        subset = f["logmag"][:500].astype(np.float32)
        lm_min, lm_max = float(subset.min()), float(subset.max())
    data_range = lm_max - lm_min
    print(f"Log-mag range: [{lm_min:.3f}, {lm_max:.3f}]")

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
        model.train()
        total_loss = 0
        for emb, target in train_loader:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            pred = model(emb)
            loss = (pred - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model.train(False)
            ssim = compute_ssim(model, val_loader, data_range)
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
    final_ssim = compute_ssim(model, val_loader, data_range)
    val_loss = 0
    with torch.no_grad():
        for emb, target in val_loader:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            val_loss += (model(emb) - target).pow(2).mean().item()
    val_mse = val_loss / len(val_loader)

    status = "PASS" if final_ssim >= SSIM_THRESHOLD else "FAIL"
    print(f"\n{'='*40}")
    print(f"=== DECODER SUMMARY ===")
    print(f"Val SSIM:     {final_ssim:.4f}  [{status} vs {SSIM_THRESHOLD}]")
    print(f"Val MSE:      {val_mse:.5f}")
    print(f"Log-mag range: [{lm_min:.3f}, {lm_max:.3f}]")
    print(f"Emb norms:    mean={emb_norms.mean():.2f} std={emb_norms.std():.2f}")
    print(f"{'='*40}")
    if status == "FAIL":
        print(f"\nWARNING: SSIM {final_ssim:.4f} < {SSIM_THRESHOLD}. Do NOT proceed to Stage 4.")
