"""Stage 2: Cache encoder embeddings for all splits."""

import sys
from pathlib import Path
import h5py
import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import load_norm_stats

CACHE_DIR = Path(__file__).parent / "cache"
CHECKPOINT = "/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt"
NORM_STATS = "/workspace/data/norm_stats.json"
BATCH = 128


def cache_split(name, model, norm_mean, norm_std):
    src = Path(f"/workspace/data/{name}.h5")
    dst = CACHE_DIR / f"embeddings_{name}.h5"
    if dst.exists():
        print(f"  {name}: already cached")
        return

    with h5py.File(src, "r") as fin:
        obs_raw = torch.from_numpy(fin["observations"][()]).float()  # [N, 16, 256, 51, 2]
    obs = obs_raw.permute(0, 1, 4, 2, 3)  # [N, 16, 2, 256, 51]
    obs = (obs - norm_mean) / norm_std
    N = obs.shape[0]
    print(f"  {name}: {N} trajectories")

    all_embs = []
    with torch.no_grad():
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            batch = obs[start:end]  # [B, 16, 2, 256, 51]
            B, T = batch.shape[:2]
            flat = rearrange(batch, "b t ... -> (b t) ...")
            emb = model.encoder(flat)
            emb = model.projector(emb)  # [B*T, 192]
            emb = rearrange(emb, "(b t) d -> b t d", b=B)
            all_embs.append(emb.numpy())
            if (start // BATCH) % 10 == 0:
                norms = emb.norm(dim=-1)
                print(f"    {end}/{N} norm: mean={norms.mean():.2f} std={norms.std():.2f}")

    all_embs = np.concatenate(all_embs, axis=0)  # [N, 16, 192]
    with h5py.File(dst, "w") as fout:
        fout.create_dataset("embeddings", data=all_embs, dtype="float32")
    print(f"  {name}: saved {all_embs.shape} to {dst}")


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Stage 2: Caching embeddings ===")

    print("Loading model...")
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.requires_grad_(False)

    stats = load_norm_stats(NORM_STATS)
    norm_mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 1, 2, 1, 1)
    norm_std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 1, 2, 1, 1)

    for split in ["train", "val", "test"]:
        cache_split(split, model, norm_mean, norm_std)
    print("Done!")
