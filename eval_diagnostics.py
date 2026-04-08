"""Diagnostic suite for RF-LeWM embedding space analysis.

Answers three questions:
1. What do target embeddings look like? (norm, scale, centering)
2. Does performance vary by RF regime / source scene?
3. Does cosine similarity tell a different story than MSE?

Run: python eval_diagnostics.py --data_path /path/to/test.h5 --model_policy lewm_rf_epoch_99
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from einops import rearrange

from dataset import RFSpectralDataset, load_norm_stats


def run(data_path, model_policy, output_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load norm stats
    stats_path = Path(data_path).parent / "norm_stats.json"
    norm_stats = load_norm_stats(stats_path) if stats_path.exists() else None

    # Load model
    model = swm.policy.AutoCostModel(model_policy)
    model = model.to(device)
    model.requires_grad_(False)

    history_size = 3
    max_rollout = 12

    # Load dataset
    dataset = RFSpectralDataset(
        data_path, history_size=history_size,
        num_preds=16 - history_size, norm_stats=norm_stats,
    )

    # Load source_ids for regime breakdown
    with h5py.File(data_path, "r") as f:
        source_ids = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in f["source_ids"][()]]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0,
    )

    # Collect per-sample metrics
    all_emb_norms = []
    all_emb_stds = []
    all_target_norms = []
    all_ctx_norms = []

    # Per-sample: model vs baselines, MSE and cosine
    per_sample_mse_model = []
    per_sample_mse_copy = []
    per_sample_mse_zero = []
    per_sample_mse_mean = []
    per_sample_cos_model = []
    per_sample_cos_copy = []
    per_sample_cos_mean = []
    per_sample_cos_delta_model = []
    per_sample_cos_delta_mean = []

    # Track which source each sample belongs to
    sample_sources = []

    sample_idx = 0
    start = time.time()

    for batch in loader:
        obs = batch["observations"].to(device)
        B = obs.size(0)

        # Map samples to source IDs
        for i in range(B):
            # Each sample comes from a trajectory; map back
            traj_idx = (sample_idx + i) // dataset.subs_per_traj
            if traj_idx < len(source_ids):
                sample_sources.append(source_ids[traj_idx])
            else:
                sample_sources.append("unknown")
        sample_idx += B

        with torch.no_grad():
            info = {"observations": obs}
            info = model.encode_rf(info)
            all_emb = info["emb"]  # (B, T, D)

            T = all_emb.size(1)
            ctx = all_emb[:, :history_size]       # (B, 3, D)
            targets = all_emb[:, history_size:]    # (B, T-3, D)

            # === 1. Embedding statistics ===
            emb_flat = rearrange(all_emb, "b t d -> (b t) d")
            all_emb_norms.append(emb_flat.norm(dim=-1).cpu())
            all_emb_stds.append(emb_flat.std(dim=-1).cpu())
            all_target_norms.append(
                rearrange(targets, "b t d -> (b t) d").norm(dim=-1).cpu()
            )
            all_ctx_norms.append(
                rearrange(ctx, "b t d -> (b t) d").norm(dim=-1).cpu()
            )

            # === 2. One-step predictions ===
            target_one = all_emb[:, history_size]  # (B, D)

            # Model prediction
            pred_model = model.predict(ctx)[:, -1]  # (B, D)

            # Baselines
            pred_copy = ctx[:, -1]                   # (B, D)
            pred_zero = torch.zeros_like(target_one)  # (B, D)
            pred_mean = ctx.mean(dim=1)               # (B, D)

            # MSE per sample
            per_sample_mse_model.append(
                (pred_model - target_one).pow(2).mean(dim=-1).cpu())
            per_sample_mse_copy.append(
                (pred_copy - target_one).pow(2).mean(dim=-1).cpu())
            per_sample_mse_zero.append(
                (pred_zero - target_one).pow(2).mean(dim=-1).cpu())
            per_sample_mse_mean.append(
                (pred_mean - target_one).pow(2).mean(dim=-1).cpu())

            # Cosine similarity per sample (absolute embeddings)
            per_sample_cos_model.append(
                F.cosine_similarity(pred_model, target_one, dim=-1).cpu())
            per_sample_cos_copy.append(
                F.cosine_similarity(pred_copy, target_one, dim=-1).cpu())
            per_sample_cos_mean.append(
                F.cosine_similarity(pred_mean, target_one, dim=-1).cpu())

            # Residual cosine similarity (direction of change — matches training objective)
            anchor = ctx[:, -1]  # last context frame
            tgt_delta = target_one - anchor
            pred_delta_model = pred_model - anchor
            pred_delta_copy = pred_copy - anchor  # = zero vector (copy predicts no change)
            pred_delta_mean = pred_mean - anchor

            # Avoid NaN when delta is zero
            delta_norm = tgt_delta.norm(dim=-1, keepdim=True)
            valid = (delta_norm.squeeze() > 1e-6)

            if valid.any():
                cos_delta_model = F.cosine_similarity(
                    pred_delta_model[valid], tgt_delta[valid], dim=-1)
                cos_delta_mean = F.cosine_similarity(
                    pred_delta_mean[valid], tgt_delta[valid], dim=-1)
                per_sample_cos_delta_model.append(cos_delta_model.cpu())
                per_sample_cos_delta_mean.append(cos_delta_mean.cpu())

    elapsed = time.time() - start

    # Aggregate
    emb_norms = torch.cat(all_emb_norms)
    target_norms = torch.cat(all_target_norms)
    ctx_norms = torch.cat(all_ctx_norms)

    mse_model = torch.cat(per_sample_mse_model)
    mse_copy = torch.cat(per_sample_mse_copy)
    mse_zero = torch.cat(per_sample_mse_zero)
    mse_mean = torch.cat(per_sample_mse_mean)
    cos_model = torch.cat(per_sample_cos_model)
    cos_copy = torch.cat(per_sample_cos_copy)
    cos_mean = torch.cat(per_sample_cos_mean)

    # ==========================================
    # REPORT
    # ==========================================
    print(f"\n{'='*70}")
    print(f"RF-LeWM Diagnostics ({model_policy}, {len(dataset)} test samples)")
    print(f"{'='*70}")

    # --- Section 1: Embedding Space ---
    print(f"\n--- 1. Embedding Space Statistics ---")
    print(f"  All embeddings:    norm mean={emb_norms.mean():.4f}  std={emb_norms.std():.4f}")
    print(f"  Context embeddings: norm mean={ctx_norms.mean():.4f}  std={ctx_norms.std():.4f}")
    print(f"  Target embeddings:  norm mean={target_norms.mean():.4f}  std={target_norms.std():.4f}")
    print(f"  Embedding dim-wise std: {torch.cat(all_emb_stds).mean():.4f}")
    print(f"")
    print(f"  If target norms are small (<<1), zero baseline wins because")
    print(f"  targets are close to zero. If norms ~1, the space is well-scaled.")

    # --- Section 2: MSE vs Cosine ---
    print(f"\n--- 2. MSE vs Cosine Similarity (one-step) ---")
    print(f"  {'Method':<20s} {'MSE':>8s} {'CosSim':>8s}")
    print(f"  {'-'*38}")
    print(f"  {'RF-LeWM':<20s} {mse_model.mean():.4f} {cos_model.mean():.4f}")
    print(f"  {'Copy-last':<20s} {mse_copy.mean():.4f} {cos_copy.mean():.4f}")
    print(f"  {'Mean-context':<20s} {mse_mean.mean():.4f} {cos_mean.mean():.4f}")
    print(f"  {'Zero':<20s} {mse_zero.mean():.4f} {'N/A':>8s}")
    print(f"")
    print(f"  If model has higher cosine sim than baselines despite worse MSE,")
    print(f"  the model predicts the right direction but wrong scale.")

    # --- Section 2b: Residual cosine similarity (matches training objective) ---
    print(f"\n--- 2b. Residual Cosine Similarity (direction of change) ---")
    if per_sample_cos_delta_model:
        cos_delta_model_all = torch.cat(per_sample_cos_delta_model)
        cos_delta_mean_all = torch.cat(per_sample_cos_delta_mean)
        print(f"  {'Method':<20s} {'Delta CosSim':>12s}")
        print(f"  {'-'*34}")
        print(f"  {'RF-LeWM':<20s} {cos_delta_model_all.mean():.4f}")
        print(f"  {'Mean-context':<20s} {cos_delta_mean_all.mean():.4f}")
        print(f"  {'Copy-last':<20s} {'0.0000':>12s}  (predicts zero change)")
        print(f"")
        print(f"  This measures cosine similarity between predicted and actual")
        print(f"  temporal CHANGE (delta), which is what the model was trained on.")
        print(f"  Higher = better directional prediction of dynamics.")
    else:
        print(f"  No valid deltas found (all targets identical to anchor)")

    # --- Section 3: Per-regime breakdown ---
    print(f"\n--- 3. Per-Regime Breakdown (one-step MSE) ---")

    # Group by source
    regime_metrics = defaultdict(lambda: {"model": [], "copy": [], "zero": [], "mean": [],
                                           "cos_model": [], "cos_copy": []})
    for i in range(len(sample_sources)):
        src = sample_sources[i]
        if i < len(mse_model):
            regime_metrics[src]["model"].append(mse_model[i].item())
            regime_metrics[src]["copy"].append(mse_copy[i].item())
            regime_metrics[src]["zero"].append(mse_zero[i].item())
            regime_metrics[src]["mean"].append(mse_mean[i].item())
            regime_metrics[src]["cos_model"].append(cos_model[i].item())
            regime_metrics[src]["cos_copy"].append(cos_copy[i].item())

    print(f"  {'Regime':<30s} {'N':>5s} {'Model':>8s} {'Copy':>8s} {'Zero':>8s} {'Mean':>8s} {'Mdl>Copy?':>10s} {'CosSim':>8s}")
    print(f"  {'-'*90}")

    for src in sorted(regime_metrics.keys()):
        m = regime_metrics[src]
        n = len(m["model"])
        avg_model = np.mean(m["model"])
        avg_copy = np.mean(m["copy"])
        avg_zero = np.mean(m["zero"])
        avg_mean = np.mean(m["mean"])
        avg_cos = np.mean(m["cos_model"])
        beats_copy = "YES" if avg_model < avg_copy else "no"
        print(f"  {src:<30s} {n:>5d} {avg_model:>8.4f} {avg_copy:>8.4f} {avg_zero:>8.4f} {avg_mean:>8.4f} {beats_copy:>10s} {avg_cos:>8.4f}")

    # --- Section 4: Embedding norm distribution ---
    print(f"\n--- 4. Target Embedding Norm Distribution ---")
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    vals = np.percentile(target_norms.numpy(), percentiles)
    for p, v in zip(percentiles, vals):
        print(f"  p{p:>3d}: {v:.4f}")

    print(f"\n  If p50 (median) is near 0, most targets are trivially zero-like.")
    print(f"  If p50 >> 0, the space has meaningful structure.")

    print(f"\nTime: {elapsed:.1f}s")
    print(f"{'='*70}\n")

    # Save structured results
    if output_dir:
        import json
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_data = {
            "model_policy": model_policy, "num_samples": len(dataset), "time_seconds": elapsed,
            "embedding_stats": {
                "all_norm_mean": emb_norms.mean().item(), "all_norm_std": emb_norms.std().item(),
                "target_norm_mean": target_norms.mean().item(), "ctx_norm_mean": ctx_norms.mean().item(),
            },
            "onestep": {
                "model_mse": mse_model.mean().item(), "copy_mse": mse_copy.mean().item(),
                "mean_mse": mse_mean.mean().item(), "zero_mse": mse_zero.mean().item(),
                "model_cossim": cos_model.mean().item(), "copy_cossim": cos_copy.mean().item(),
                "mean_cossim": cos_mean.mean().item(),
            },
            "residual_cossim": {},
        }
        if per_sample_cos_delta_model:
            save_data["residual_cossim"] = {
                "model_delta_cossim": torch.cat(per_sample_cos_delta_model).mean().item(),
                "mean_delta_cossim": torch.cat(per_sample_cos_delta_mean).mean().item(),
            }
        with open(out / "diagnostics.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {out / 'diagnostics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_policy", type=str, default="lewm_rf_epoch_99")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    run(args.data_path, args.model_policy, args.output_dir)
