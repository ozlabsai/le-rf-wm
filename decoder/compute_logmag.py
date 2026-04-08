"""Stage 1: Compute log-magnitude spectrograms from raw STFT observations."""

import h5py
import numpy as np
from pathlib import Path

SPLITS = {
    "train": "../../labs/ChangShuoRadioData/data/lewm/processed/train.h5",
    "val": "../../labs/ChangShuoRadioData/data/lewm/processed/val.h5",
    "test": "../../labs/ChangShuoRadioData/data/lewm/processed/test.h5",
}
CACHE_DIR = Path(__file__).parent / "cache"
CHUNK = 500


def compute_split(name, src_path):
    dst = CACHE_DIR / f"logmag_{name}.h5"
    if dst.exists():
        print(f"  {name}: already cached at {dst}")
        return

    with h5py.File(src_path, "r") as fin:
        obs = fin["observations"]  # [N, 16, 256, 51, 2]
        N = obs.shape[0]
        print(f"  {name}: {N} trajectories from {src_path}")

        with h5py.File(dst, "w") as fout:
            ds = fout.create_dataset("logmag", shape=(N, 16, 256, 51), dtype="float32")
            for start in range(0, N, CHUNK):
                end = min(start + CHUNK, N)
                chunk = obs[start:end].astype(np.float32)  # [B, 16, 256, 51, 2]
                real, imag = chunk[..., 0], chunk[..., 1]
                mag = np.sqrt(real**2 + imag**2 + 1e-12)
                logmag = np.log(mag + 1e-6)
                ds[start:end] = logmag
                if (start // CHUNK) % 5 == 0:
                    print(f"    {end}/{N} ({100*end/N:.0f}%)")

    print(f"  {name}: saved to {dst}")


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Stage 1: Computing log-magnitude ===")
    for name, path in SPLITS.items():
        src = Path(__file__).parent.parent / path
        if not src.exists():
            # Try workspace path (RunPod)
            src = Path(f"/workspace/data/{name}.h5")
        compute_split(name, str(src))
    print("Done!")
