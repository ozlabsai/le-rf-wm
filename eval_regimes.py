"""Regime-aware evaluation for RF-LeWM.

Breaks down model performance by RF activity regime (quiet, dense, bursty,
ramp_up, interference_event, correlated_alternating, correlated_leader_follower, random)
and by SNR range.

Run: python eval_regimes.py --data_path /path/to/test.h5 --model_policy lewm_rf_epoch_99
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F

from dataset import RFSpectralDataset, load_norm_stats


def run(data_path, model_policy, metadata_path="scene_metadata.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load scene metadata (regime, SNR, etc.)
    with open(metadata_path) as f:
        scene_meta = json.load(f)

    # Load model
    stats_path = Path(data_path).parent / "norm_stats.json"
    norm_stats = load_norm_stats(stats_path) if stats_path.exists() else None
    model = swm.policy.AutoCostModel(model_policy)
    model = model.to(device)
    model.requires_grad_(False)

    history_size = 3
    max_rollout = 12

    dataset = RFSpectralDataset(
        data_path, history_size=history_size,
        num_preds=16 - history_size, norm_stats=norm_stats,
    )

    # Load source_ids
    with h5py.File(data_path, "r") as f:
        source_ids = [s.decode() if isinstance(s, bytes) else str(s)
                      for s in f["source_ids"][()]]

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0,
    )

    # Per-sample metrics
    sample_regimes = []
    sample_snrs = []
    mse_model_all = []
    mse_copy_all = []
    mse_mean_all = []
    cos_model_all = []
    cos_copy_all = []
    rollout_model_all = []  # list of (B, n_steps) tensors
    rollout_copy_all = []

    sample_idx = 0
    start = time.time()

    for batch in loader:
        obs = batch["observations"].to(device)
        B = obs.size(0)

        # Map samples to regimes
        for i in range(B):
            traj_idx = (sample_idx + i) // dataset.subs_per_traj
            if traj_idx < len(source_ids):
                sid = source_ids[traj_idx]
                meta = scene_meta.get(sid, {})
                sample_regimes.append(meta.get("regime", "unknown"))
                sample_snrs.append(meta.get("snr_db", 0))
            else:
                sample_regimes.append("unknown")
                sample_snrs.append(0)
        sample_idx += B

        with torch.no_grad():
            info = {"observations": obs}
            info = model.encode_rf(info)
            all_emb = info["emb"]
            T = all_emb.size(1)

            ctx = all_emb[:, :history_size]
            target_one = all_emb[:, history_size]

            # Model prediction
            pred_model = model.predict(ctx)[:, -1]
            pred_copy = ctx[:, -1]
            pred_mean = ctx.mean(dim=1)

            # One-step MSE
            mse_model_all.append((pred_model - target_one).pow(2).mean(dim=-1).cpu())
            mse_copy_all.append((pred_copy - target_one).pow(2).mean(dim=-1).cpu())
            mse_mean_all.append((pred_mean - target_one).pow(2).mean(dim=-1).cpu())

            # One-step cosine
            cos_model_all.append(F.cosine_similarity(pred_model, target_one, dim=-1).cpu())
            cos_copy_all.append(F.cosine_similarity(pred_copy, target_one, dim=-1).cpu())

            # Multi-step rollout
            n_steps = min(max_rollout, T - history_size)
            rolled = model.rollout_unconditional(ctx.clone(), n_steps=n_steps, history_size=history_size)

            model_rollout = []
            copy_rollout = []
            for t in range(n_steps):
                pred_t = rolled[:, history_size + t]
                target_t = all_emb[:, history_size + t]
                copy_t = ctx[:, -1]  # copy-last doesn't change
                model_rollout.append((pred_t - target_t).pow(2).mean(dim=-1).cpu())
                copy_rollout.append((copy_t - target_t).pow(2).mean(dim=-1).cpu())

            rollout_model_all.append(torch.stack(model_rollout, dim=1))
            rollout_copy_all.append(torch.stack(copy_rollout, dim=1))

    elapsed = time.time() - start

    # Concatenate
    mse_model = torch.cat(mse_model_all).numpy()
    mse_copy = torch.cat(mse_copy_all).numpy()
    mse_mean = torch.cat(mse_mean_all).numpy()
    cos_model = torch.cat(cos_model_all).numpy()
    cos_copy = torch.cat(cos_copy_all).numpy()
    rollout_model = torch.cat(rollout_model_all).numpy()
    rollout_copy = torch.cat(rollout_copy_all).numpy()

    # ==========================================
    # REPORT
    # ==========================================
    print(f"\n{'='*90}")
    print(f"RF-LeWM Regime Analysis ({model_policy}, {len(dataset)} test samples)")
    print(f"{'='*90}")

    # --- By Regime ---
    print(f"\n--- 1. Performance by Regime (one-step) ---")
    print(f"  {'Regime':<30s} {'N':>5s} {'Model MSE':>10s} {'Copy MSE':>10s} {'Impr%':>7s} {'Mdl Cos':>8s} {'Cpy Cos':>8s} {'Cos Impr':>9s}")
    print(f"  {'-'*92}")

    regime_order = ["quiet", "dense", "bursty", "ramp_up", "interference_event",
                    "correlated_alternating", "correlated_leader_follower", "random"]

    regime_indices = defaultdict(list)
    for i, r in enumerate(sample_regimes):
        regime_indices[r].append(i)

    for regime in regime_order:
        idx = regime_indices.get(regime, [])
        if not idx:
            continue
        idx = np.array(idx)
        m_mse = mse_model[idx].mean()
        c_mse = mse_copy[idx].mean()
        impr = (1 - m_mse / c_mse) * 100
        m_cos = cos_model[idx].mean()
        c_cos = cos_copy[idx].mean()
        cos_impr = f"{m_cos - c_cos:+.4f}"
        print(f"  {regime:<30s} {len(idx):>5d} {m_mse:>10.4f} {c_mse:>10.4f} {impr:>+6.1f}% {m_cos:>8.4f} {c_cos:>8.4f} {cos_impr:>9s}")

    # Total
    print(f"  {'-'*92}")
    print(f"  {'TOTAL':<30s} {len(mse_model):>5d} {mse_model.mean():>10.4f} {mse_copy.mean():>10.4f} {(1-mse_model.mean()/mse_copy.mean())*100:>+6.1f}% {cos_model.mean():>8.4f} {cos_copy.mean():>8.4f} {cos_model.mean()-cos_copy.mean():>+9.4f}")

    # --- Rollout by Regime ---
    print(f"\n--- 2. Rollout Error by Regime (Model MSE at steps 1, 4, 8, 12) ---")
    print(f"  {'Regime':<30s} {'Step 1':>8s} {'Step 4':>8s} {'Step 8':>8s} {'Step 12':>8s} {'Degrad.':>8s}")
    print(f"  {'-'*74}")

    for regime in regime_order:
        idx = regime_indices.get(regime, [])
        if not idx:
            continue
        idx = np.array(idx)
        r = rollout_model[idx]
        n = min(r.shape[1], 12)
        s1 = r[:, 0].mean()
        s4 = r[:, min(3, n-1)].mean()
        s8 = r[:, min(7, n-1)].mean()
        s12 = r[:, n-1].mean()
        degrad = s12 / s1 if s1 > 0 else 0
        print(f"  {regime:<30s} {s1:>8.4f} {s4:>8.4f} {s8:>8.4f} {s12:>8.4f} {degrad:>7.2f}x")

    # --- Model vs Copy-last Rollout Improvement by Regime ---
    print(f"\n--- 3. Model vs Copy-last Improvement Over Rollout Horizon ---")
    print(f"  {'Regime':<30s}", end="")
    for step in [1, 4, 8, 12]:
        print(f" {'Step '+str(step):>8s}", end="")
    print()
    print(f"  {'-'*66}")

    for regime in regime_order:
        idx = regime_indices.get(regime, [])
        if not idx:
            continue
        idx = np.array(idx)
        rm = rollout_model[idx]
        rc = rollout_copy[idx]
        n = min(rm.shape[1], 12)
        print(f"  {regime:<30s}", end="")
        for step in [1, 4, 8, 12]:
            s = min(step - 1, n - 1)
            impr = (1 - rm[:, s].mean() / rc[:, s].mean()) * 100
            print(f" {impr:>+7.1f}%", end="")
        print()

    # --- By SNR ---
    print(f"\n--- 4. Performance by SNR Range ---")
    snr_bins = [(-10, 0), (0, 10), (10, 20), (20, 35)]
    print(f"  {'SNR Range':>12s} {'N':>5s} {'Model MSE':>10s} {'Copy MSE':>10s} {'Impr%':>7s} {'Mdl Cos':>8s}")
    print(f"  {'-'*56}")

    for lo, hi in snr_bins:
        idx = [i for i, s in enumerate(sample_snrs) if lo <= s < hi and i < len(mse_model)]
        if not idx:
            continue
        idx = np.array(idx)
        m_mse = mse_model[idx].mean()
        c_mse = mse_copy[idx].mean()
        impr = (1 - m_mse / c_mse) * 100
        m_cos = cos_model[idx].mean()
        print(f"  {f'[{lo}, {hi})':>12s} {len(idx):>5d} {m_mse:>10.4f} {c_mse:>10.4f} {impr:>+6.1f}% {m_cos:>8.4f}")

    print(f"\nTime: {elapsed:.1f}s")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_policy", type=str, default="lewm_rf_epoch_99")
    parser.add_argument("--metadata", type=str, default="scene_metadata.json")
    args = parser.parse_args()
    run(args.data_path, args.model_policy, args.metadata)
