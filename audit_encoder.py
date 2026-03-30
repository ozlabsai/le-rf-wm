"""Stage-by-stage audit of the SpectrogramViT encoder.

Checks where input variance disappears in the encoder pipeline.
Run: python audit_encoder.py --data_path /path/to/train.h5
"""

import argparse
import torch
from einops import rearrange
from dataset import RFSpectralDataset, compute_norm_stats
from encoder import SpectrogramViT
from module import MLP


def batch_std(x, name):
    """Print std stats for a tensor across the batch dimension."""
    if x.ndim == 2:
        # (B, D)
        std = x.std(dim=0)
    elif x.ndim == 3:
        # (B, T, D) — flatten batch
        std = rearrange(x, "b t d -> (b t) d").std(dim=0)
    else:
        # flatten all but last dim
        flat = x.reshape(-1, x.shape[-1])
        std = flat.std(dim=0)

    print(f"  {name:40s}  shape={str(list(x.shape)):20s}  "
          f"std: mean={std.mean():.6f}  min={std.min():.6f}  max={std.max():.6f}")
    return std


def run(data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a batch of DIFFERENT trajectories (not consecutive frames from one traj)
    stats = compute_norm_stats(data_path)
    ds = RFSpectralDataset(data_path, history_size=3, num_preds=2, norm_stats=stats)

    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    obs = batch["observations"]  # (B, 5, 2, 256, 51)

    # Use just the first frame from each sample to get 32 different spectrograms
    x = obs[:, 0].to(device)  # (32, 2, 256, 51)
    print(f"Input batch: {x.shape}")
    print()

    # Check input variance
    print("=== 1. Raw Input ===")
    batch_std(x.reshape(x.size(0), -1), "raw input (flattened)")
    batch_std(x[:, 0], "channel 0 (real)")
    batch_std(x[:, 1], "channel 1 (imag)")

    # Check if inputs are actually different
    diffs = []
    for i in range(min(5, x.size(0))):
        for j in range(i+1, min(5, x.size(0))):
            d = (x[i] - x[j]).abs().mean().item()
            diffs.append(d)
    print(f"\n  Mean pairwise input diff (first 5): {sum(diffs)/len(diffs):.6f}")

    # Build encoder
    encoder = SpectrogramViT(hidden_dim=192, depth=12, heads=3, mlp_dim=768).to(device)
    projector = MLP(192, 2048, 192, norm_fn=torch.nn.LayerNorm).to(device)

    print("\n=== 2. Encoder Stages (at init, no training) ===")

    with torch.no_grad():
        B = x.size(0)

        # Stage 1: patch embedding
        h = encoder.patch_embed(x)  # (B, hidden_dim, n_freq, n_time)
        batch_std(h.reshape(B, -1), "after patch_embed (flat)")
        batch_std(rearrange(h, "b d f t -> b (f t) d"), "after patch_embed (per patch)")

        # Stage 2: rearrange + positional embeddings
        h = rearrange(h, "b d f t -> b f t d")
        h = h + encoder.freq_pos + encoder.time_pos
        h_flat = rearrange(h, "b f t d -> b (f t) d")
        batch_std(h_flat, "after pos embeddings")

        # Stage 3: add CLS token
        cls = encoder.cls_token.expand(B, -1, -1) + encoder.cls_pos
        h_with_cls = torch.cat([cls, h_flat], dim=1)
        batch_std(h_with_cls, "after CLS prepend")

        # Check: is CLS the same for all samples?
        cls_std = cls.squeeze(1).std(dim=0)
        print(f"  {'CLS token std across batch':40s}  mean={cls_std.mean():.6f}")

        # Stage 4: through transformer blocks one by one
        h = encoder.dropout(h_with_cls)
        for i, block in enumerate(encoder.blocks):
            h = block(h)
            if i < 3 or i == len(encoder.blocks) - 1:
                # Check CLS token and patch tokens separately
                cls_h = h[:, 0]  # (B, D)
                patch_h = h[:, 1:]  # (B, num_patches, D)
                batch_std(cls_h, f"block {i:2d} CLS token")
                batch_std(patch_h, f"block {i:2d} patch tokens")

        # Stage 5: final norm
        h = encoder.norm(h)
        cls_final = h[:, 0]
        batch_std(cls_final, "after final LayerNorm (CLS)")

        # Stage 6: projector
        proj_out = projector(cls_final)
        batch_std(proj_out, "after projector")

    # Summary
    print("\n=== 3. Summary ===")
    print("If std drops to 0 at some stage, that's where the encoder loses input variation.")
    print("If std is already 0 at 'after patch_embed', the patch embedding is broken.")
    print("If std is nonzero through transformer but 0 after projector, the projector is broken.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    run(args.data_path)
