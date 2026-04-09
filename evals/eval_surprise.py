"""Perturbation-based surprise evaluation for RF-LeWM.

Tests whether the model assigns higher surprise scores to anomalous/perturbed
RF trajectories compared to normal ones.

Perturbation types:
1. Signal injection: add a synthetic signal mid-trajectory
2. Signal dropout: zero out frequency bands mid-trajectory
3. Temporal reversal: reverse the second half of the trajectory
4. Gaussian noise burst: add strong noise to a few frames

Run: python eval_surprise.py --data_path /path/to/test.h5 --model_policy lewm_rf_epoch_99
"""

import argparse
import time
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch

import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from dataset import RFSpectralDataset, load_norm_stats


def perturb_signal_injection(obs, strength=3.0):
    """Inject a synthetic narrowband signal into frames 8-15."""
    perturbed = obs.clone()
    B, T, C, F, Tb = perturbed.shape
    # Add a strong sinusoidal signal across a narrow frequency band
    freq_band = slice(F // 4, F // 4 + 10)
    perturbed[:, 8:, :, freq_band, :] += strength * torch.randn(1, device=obs.device)
    return perturbed


def perturb_signal_dropout(obs):
    """Zero out a frequency band in frames 8-15 (signal disappears)."""
    perturbed = obs.clone()
    B, T, C, F, Tb = perturbed.shape
    freq_band = slice(F // 3, F // 3 + 30)
    perturbed[:, 8:, :, freq_band, :] = 0
    return perturbed


def perturb_temporal_reversal(obs):
    """Reverse the temporal order of the second half (frames 8-15)."""
    perturbed = obs.clone()
    perturbed[:, 8:] = perturbed[:, 8:].flip(dims=[1])
    return perturbed


def perturb_noise_burst(obs, strength=5.0):
    """Add strong Gaussian noise to frames 8-10."""
    perturbed = obs.clone()
    noise = strength * torch.randn_like(perturbed[:, 8:11])
    perturbed[:, 8:11] += noise
    return perturbed


def compute_surprise(model, obs, history_size=3, max_rollout=12):
    """Compute per-trajectory surprise as mean rollout MSE."""
    B, T = obs.shape[:2]
    info = {"observations": obs}
    info = model.encode_rf(info)
    all_emb = info["emb"]

    ctx = all_emb[:, :history_size]
    n_steps = min(max_rollout, T - history_size)
    rolled = model.rollout_unconditional(ctx.clone(), n_steps=n_steps, history_size=history_size)

    errors = []
    for t in range(n_steps):
        pred_t = rolled[:, history_size + t]
        target_t = all_emb[:, history_size + t]
        err = (pred_t - target_t).pow(2).mean(dim=-1)
        errors.append(err)

    # Mean error across rollout steps per trajectory
    surprise = torch.stack(errors, dim=1).mean(dim=1)  # (B,)
    return surprise


def run(data_path, model_policy, output_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stats_path = Path(data_path).parent / "norm_stats.json"
    norm_stats = load_norm_stats(stats_path) if stats_path.exists() else None

    model = swm.policy.AutoCostModel(model_policy)
    model = model.to(device)
    model.requires_grad_(False)

    history_size = 3

    # Load full trajectories
    dataset = RFSpectralDataset(
        data_path, history_size=history_size,
        num_preds=16 - history_size, norm_stats=norm_stats,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0,
    )

    perturbations = {
        "Normal (no perturbation)": lambda obs: obs,
        "Signal injection": perturb_signal_injection,
        "Signal dropout": perturb_signal_dropout,
        "Temporal reversal": perturb_temporal_reversal,
        "Noise burst": perturb_noise_burst,
    }

    results = {name: [] for name in perturbations}

    start = time.time()

    for batch in loader:
        obs = batch["observations"].to(device)

        for name, perturb_fn in perturbations.items():
            with torch.no_grad():
                perturbed = perturb_fn(obs)
                surprise = compute_surprise(model, perturbed, history_size)
                results[name].append(surprise.cpu())

    elapsed = time.time() - start

    # Aggregate
    print(f"\n{'='*70}")
    print(f"RF-LeWM Surprise / Anomaly Detection ({model_policy})")
    print(f"{'='*70}")
    print(f"\n{'Perturbation':<30s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'p95':>8s} {'Ratio':>7s}")
    print(f"{'-'*72}")

    normal_mean = torch.cat(results["Normal (no perturbation)"]).mean().item()

    for name in perturbations:
        scores = torch.cat(results[name])
        mean = scores.mean().item()
        std = scores.std().item()
        median = scores.median().item()
        p95 = np.percentile(scores.numpy(), 95)
        ratio = mean / normal_mean if normal_mean > 0 else 0
        print(f"  {name:<28s} {mean:>8.4f} {std:>8.4f} {median:>8.4f} {p95:>8.4f} {ratio:>6.2f}x")

    print(f"\n  A ratio > 1.0 means the perturbation increases surprise.")
    print(f"  Higher ratio = model detects the anomaly more strongly.")

    # Statistical significance: compare normal vs each perturbation
    print(f"\n--- Statistical Comparison (normal vs perturbed) ---")
    normal_scores = torch.cat(results["Normal (no perturbation)"]).numpy()

    for name in perturbations:
        if name == "Normal (no perturbation)":
            continue
        perturbed_scores = torch.cat(results[name]).numpy()
        # Simple: what fraction of perturbed trajectories have higher surprise than normal median?
        normal_median = np.median(normal_scores)
        frac_above = (perturbed_scores > normal_median).mean()
        print(f"  {name:<28s}: {frac_above*100:.1f}% of perturbed trajs above normal median")

    print(f"\nTime: {elapsed:.1f}s")
    print(f"{'='*70}\n")

    # Save structured results
    if output_dir:
        import json
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_data = {"model_policy": model_policy, "time_seconds": elapsed, "perturbations": {}}
        for name in perturbations:
            scores = torch.cat(results[name])
            normal_median = np.median(torch.cat(results["Normal (no perturbation)"]).numpy())
            frac_above = float((scores.numpy() > normal_median).mean()) if name != "Normal (no perturbation)" else 0.5
            save_data["perturbations"][name] = {
                "mean": float(scores.mean()), "std": float(scores.std()),
                "median": float(scores.median()), "p95": float(np.percentile(scores.numpy(), 95)),
                "ratio": float(scores.mean() / normal_mean) if normal_mean > 0 else 0,
                "frac_above_normal_median": frac_above,
            }
        with open(out / "surprise.json", "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {out / 'surprise.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_policy", type=str, default="lewm_rf_epoch_99")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    run(args.data_path, args.model_policy, args.output_dir)
