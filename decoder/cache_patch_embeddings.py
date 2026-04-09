"""Cache patch-level encoder embeddings for all splits.

Stores the 272 patch tokens (before mean pooling) for each frame.
Output shape: [N, 16, 272, 192] per split.

Usage:
    python decoder/cache_patch_embeddings.py --checkpoint /path/to/model.ckpt
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import load_norm_stats

CACHE_DIR = Path(__file__).parent / "cache"
DEFAULT_CHECKPOINT = "/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt"
NORM_STATS = "/workspace/data/norm_stats.json"
BATCH = 64  # smaller batch — 272 tokens per frame uses more memory


def cache_split(name, model, norm_mean, norm_std):
    src = Path(f"/workspace/data/{name}.h5")
    dst = CACHE_DIR / f"patch_embeddings_{name}.h5"
    if dst.exists():
        print(f"  {name}: already cached")
        return

    with h5py.File(src, "r") as fin:
        obs_raw = torch.from_numpy(fin["observations"][()]).float()  # [N, 16, 256, 51, 2]
    obs = obs_raw.permute(0, 1, 4, 2, 3)  # [N, 16, 2, 256, 51]
    obs = (obs - norm_mean.cpu()) / norm_std.cpu()
    N = obs.shape[0]
    print(f"  {name}: {N} trajectories")

    device = next(model.parameters()).device
    all_patches = []

    with torch.no_grad():
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            batch = obs[start:end].to(device)  # [B, 16, 2, 256, 51]
            B, T = batch.shape[:2]
            flat = rearrange(batch, "b t ... -> (b t) ...")
            # Get patch tokens before pooling: (B*T, 16, 17, 192)
            patches = model.encoder.forward_patches(flat)
            # Flatten spatial: (B*T, 272, 192)
            patches = rearrange(patches, "bt f t d -> bt (f t) d")
            # Reshape back: (B, T, 272, 192)
            patches = rearrange(patches, "(b t) p d -> b t p d", b=B)
            all_patches.append(patches.cpu().numpy())
            if (start // BATCH) % 5 == 0:
                print(f"    {end}/{N}")

    all_patches = np.concatenate(all_patches, axis=0)  # [N, 16, 272, 192]
    with h5py.File(dst, "w") as fout:
        fout.create_dataset("patch_embeddings", data=all_patches, dtype="float32")
    print(f"  {name}: saved {all_patches.shape} to {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm_stats", default=NORM_STATS)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Caching patch embeddings ===")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.checkpoint} on {DEVICE}...")
    model = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.to(DEVICE)
    model.requires_grad_(False)

    stats = load_norm_stats(args.norm_stats)
    norm_mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 1, 2, 1, 1).to(DEVICE)
    norm_std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 1, 2, 1, 1).to(DEVICE)

    for split in ["train", "val", "test"]:
        cache_split(split, model, norm_mean, norm_std)
    print("Done!")
