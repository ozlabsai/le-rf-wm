"""Stage 5: Attention proxy — per-patch embedding norms as attention map."""

import sys
from pathlib import Path
import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent))

CHECKPOINT = "/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt"
NORM_STATS = "/workspace/data/norm_stats.json"


def get_attention_maps(model, obs_normalized):
    """Extract per-patch norms as attention proxy.
    obs_normalized: (16, 2, 256, 51) — normalized observations
    Returns: (16, 256, 51) — attention maps normalized 0-1 per frame
    """
    with torch.no_grad():
        _, patch_norms = model.encoder(obs_normalized, return_patch_norms=True)
    # patch_norms: (16, n_freq, n_time) e.g. (16, 16, 17)
    n_freq, n_time = patch_norms.shape[1], patch_norms.shape[2]

    # Upsample to full spectrogram resolution
    maps = torch.nn.functional.interpolate(
        patch_norms.unsqueeze(1).float(),  # (16, 1, 16, 17)
        size=(256, 51), mode="bilinear", align_corners=False,
    ).squeeze(1)  # (16, 256, 51)

    # Normalize 0-1 per frame
    for i in range(maps.shape[0]):
        mn, mx = maps[i].min(), maps[i].max()
        if mx > mn:
            maps[i] = (maps[i] - mn) / (mx - mn)
    return maps.numpy()


def visualize_attention(test_h5_path, traj_idx=0, frame_idx=8, output_path=None):
    """Save a matplotlib figure: spectrogram + attention overlay."""
    import h5py
    import matplotlib.pyplot as plt
    from dataset import load_norm_stats

    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.requires_grad_(False)

    stats = load_norm_stats(NORM_STATS)
    norm_mean = torch.tensor(stats["mean"]).float().view(1, 2, 1, 1)
    norm_std = torch.tensor(stats["std"]).float().view(1, 2, 1, 1)

    with h5py.File(test_h5_path, "r") as f:
        obs = torch.from_numpy(f["observations"][traj_idx]).float()  # (16, 256, 51, 2)
    obs = obs.permute(0, 3, 1, 2)  # (16, 2, 256, 51)
    obs = (obs - norm_mean) / norm_std

    attn = get_attention_maps(model, obs)  # (16, 256, 51)

    # Compute log-magnitude for display
    obs_raw = obs * norm_std + norm_mean
    mag = (obs_raw[:, 0]**2 + obs_raw[:, 1]**2).sqrt()
    logmag = torch.log(mag + 1e-6).numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    frame = frame_idx

    ax1.imshow(logmag[frame], aspect="auto", origin="lower", cmap="viridis")
    ax1.set_title(f"Log-Magnitude (t={frame})")
    ax1.set_ylabel("Frequency bin")
    ax1.set_xlabel("Time bin")

    ax2.imshow(attn[frame], aspect="auto", origin="lower", cmap="hot")
    ax2.set_title(f"Attention Map (t={frame})")
    ax2.set_xlabel("Time bin")

    ax3.imshow(logmag[frame], aspect="auto", origin="lower", cmap="viridis", alpha=0.6)
    ax3.imshow(attn[frame], aspect="auto", origin="lower", cmap="hot", alpha=0.4)
    ax3.set_title(f"Overlay (t={frame})")
    ax3.set_xlabel("Time bin")

    plt.tight_layout()
    out = output_path or f"decoder/attention_traj{traj_idx}_t{frame}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/workspace/data/test.h5")
    parser.add_argument("--traj", type=int, default=0)
    parser.add_argument("--frame", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    visualize_attention(args.data_path, args.traj, args.frame, args.output)
