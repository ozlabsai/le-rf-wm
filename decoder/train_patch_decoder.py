"""Train patch-level decoder: 272 patch tokens -> spectrogram reconstruction.

Stage A: Cache patch tokens from frozen encoder (if not already cached)
Stage B: Train PatchDecoder on (patch_tokens, logmag) pairs
Stage C: Evaluate SSIM

Run: uv run python decoder/train_patch_decoder.py
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from torchmetrics.image import StructuralSimilarityIndexMeasure

sys.path.insert(0, str(Path(__file__).parent.parent))

from patch_decoder import PatchDecoder

CACHE = Path(__file__).parent / "cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt"
NORM_STATS = "/workspace/data/norm_stats.json"
EPOCHS = 50
BATCH = 256
LR = 1e-3
LR_MIN = 1e-5


# ── Stage A: Cache patch tokens ──────────────────────────────────────────

def cache_patch_tokens():
    """Extract and cache patch tokens (pre-pooling) from frozen encoder."""
    from dataset import load_norm_stats

    for split in ["train", "val", "test"]:
        dst = CACHE / f"patches_{split}.h5"
        if dst.exists():
            print(f"  {split}: already cached")
            continue

        print(f"  Loading model for {split}...")
        model = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        model.requires_grad_(False)

        stats = load_norm_stats(NORM_STATS)
        nm = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 1, 2, 1, 1)
        ns = torch.tensor(stats["std"], dtype=torch.float32).view(1, 1, 2, 1, 1)

        src = Path(f"/workspace/data/{split}.h5")
        with h5py.File(src, "r") as fin:
            obs_raw = torch.from_numpy(fin["observations"][()]).float()
        obs = obs_raw.permute(0, 1, 4, 2, 3)  # (N, 16, 2, 256, 51)
        obs = (obs - nm) / ns
        N = obs.shape[0]
        print(f"  {split}: {N} trajectories")

        # Determine output shape from one forward pass
        with torch.no_grad():
            sample = obs[0:1, 0:1].to(DEVICE)
            sample_flat = rearrange(sample, "b t ... -> (b t) ...")
            patches = model.encoder.forward_patches(sample_flat)
            _, nf, nt, D = patches.shape
        print(f"  Patch shape: ({nf}, {nt}, {D})")

        all_patches = np.zeros((N, 16, nf, nt, D), dtype=np.float32)
        B_SIZE = 32

        with torch.no_grad():
            for start in range(0, N, B_SIZE):
                end = min(start + B_SIZE, N)
                batch = obs[start:end].to(DEVICE)  # (B, 16, 2, 256, 51)
                B, T = batch.shape[:2]
                flat = rearrange(batch, "b t ... -> (b t) ...")
                p = model.encoder.forward_patches(flat)  # (B*T, nf, nt, D)
                p = rearrange(p, "(b t) f ti d -> b t f ti d", b=B)
                all_patches[start:end] = p.cpu().numpy()
                if start % (B_SIZE * 10) == 0:
                    print(f"    {end}/{N}")

        with h5py.File(dst, "w") as fout:
            fout.create_dataset("patches", data=all_patches, dtype="float32",
                                chunks=(min(100, N), 16, nf, nt, D))
        print(f"  {split}: saved {all_patches.shape} to {dst}")
        del model, all_patches  # free memory


# ── Stage B: Train ────────────────────────────────────────────────────────

class PatchLogmagDataset(Dataset):
    """Paired (patch_tokens, logmag_frame) — flattened across timesteps."""
    def __init__(self, split, lm_min, lm_range):
        with h5py.File(CACHE / f"patches_{split}.h5", "r") as f:
            self.patches = torch.from_numpy(f["patches"][()]).float()  # (N, 16, nf, nt, D)
        with h5py.File(CACHE / f"logmag_{split}.h5", "r") as f:
            logmag = torch.from_numpy(f["logmag"][()]).float()  # (N, 16, 256, 51)
        self.logmag = ((logmag - lm_min) / lm_range).clamp(0, 1)
        self.N, self.T = self.patches.shape[:2]

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, idx):
        n, t = divmod(idx, self.T)
        return self.patches[n, t], self.logmag[n, t]


def compute_ssim(model, loader):
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    with torch.no_grad():
        for patches, target in loader:
            patches, target = patches.to(DEVICE), target.to(DEVICE)
            pred = model(patches).clamp(0, 1)
            ssim_fn.update(pred.unsqueeze(1), target.unsqueeze(1))
    return ssim_fn.compute().item()


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)

    # ── Stage A ──
    print("=== Stage A: Caching patch tokens ===")
    cache_patch_tokens()

    # ── Normalization ──
    with h5py.File(CACHE / "logmag_train.h5", "r") as f:
        subset = f["logmag"][:500].astype(np.float32)
        lm_min, lm_max = float(subset.min()), float(subset.max())
    lm_range = lm_max - lm_min
    print(f"\nLog-mag range: [{lm_min:.3f}, {lm_max:.3f}], normalized to [0,1]")

    # ── Stage B ──
    print("\n=== Stage B: Training patch decoder ===")
    train_ds = PatchLogmagDataset("train", lm_min, lm_range)
    val_ds = PatchLogmagDataset("val", lm_min, lm_range)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    # Get patch dimensions from data
    sample_patches = train_ds.patches[0, 0]  # (nf, nt, D)
    nf, nt, D = sample_patches.shape
    print(f"Patches: ({nf}, {nt}, {D}), reconstructing to ({nf*16}, {nt*3})")

    model = PatchDecoder(hidden_dim=D, patch_freq=16, patch_time=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PatchDecoder: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_steps = EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=LR_MIN)

    best_ssim = 0.0
    best_path = Path(__file__).parent / "patch_decoder_best.ckpt"

    for epoch in range(EPOCHS):
        model.train(True)
        total_loss = 0
        for patches, target in train_loader:
            patches, target = patches.to(DEVICE), target.to(DEVICE)
            pred = model(patches).clamp(0, 1)
            loss = (pred - target).pow(2).mean()
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
            print(f"Epoch {epoch+1:>3d}/{EPOCHS}  loss={avg_loss:.6f}  val_SSIM={ssim:.4f}  lr={lr:.2e}")
            if ssim > best_ssim:
                best_ssim = ssim
                torch.save(model.state_dict(), best_path)
            model.train(True)
        else:
            print(f"Epoch {epoch+1:>3d}/{EPOCHS}  loss={avg_loss:.6f}")

    # ── Stage C: Final eval ──
    print("\n=== Stage C: Final evaluation ===")
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.train(False)
    final_ssim = compute_ssim(model, val_loader)

    val_mse = 0
    with torch.no_grad():
        for patches, target in val_loader:
            patches, target = patches.to(DEVICE), target.to(DEVICE)
            val_mse += (model(patches).clamp(0, 1) - target).pow(2).mean().item()
    val_mse /= len(val_loader)

    status = "PASS" if final_ssim >= 0.70 else "FAIL"
    print(f"\n{'='*50}")
    print(f"=== PATCH DECODER SUMMARY ===")
    print(f"Val SSIM:       {final_ssim:.4f}  [{status} vs 0.70]")
    print(f"Val MSE:        {val_mse:.6f}")
    print(f"Params:         {n_params:,}")
    print(f"Patch grid:     {nf}x{nt} patches, each {D}-dim -> 16x3 pixels")
    print(f"{'='*50}")
