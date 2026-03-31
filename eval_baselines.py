"""Baseline comparison for RF-LeWM evaluation.

Compares the learned world model against trivial baselines:
1. Copy-last: predict next embedding = last context embedding
2. Mean: predict next embedding = mean of context embeddings
3. Zero: predict next embedding = zero vector

Run: python eval_baselines.py --data_path /path/to/test.h5 --model_policy lewm_rf_epoch_99
"""

import argparse
import time

import stable_worldmodel as swm
import torch
from pathlib import Path

from dataset import RFSpectralDataset, load_norm_stats


def evaluate_rollout(all_emb, history_size, max_rollout, predict_fn):
    """Evaluate rollout error for a given prediction function.
    predict_fn(ctx) -> next_emb, where ctx is (B, H, D) and next_emb is (B, D)
    """
    T = all_emb.size(1)
    n_steps = min(max_rollout, T - history_size)

    # one-step
    ctx = all_emb[:, :history_size]
    pred_one = predict_fn(ctx)
    target_one = all_emb[:, history_size]
    onestep_mse = (pred_one - target_one).pow(2).mean(dim=-1)

    # rollout
    rollout_emb = ctx.clone()
    rollout_errors = []
    for t in range(n_steps):
        ctx_window = rollout_emb[:, -history_size:]
        pred = predict_fn(ctx_window)
        rollout_emb = torch.cat([rollout_emb, pred.unsqueeze(1)], dim=1)

        target = all_emb[:, history_size + t]
        err = (pred - target).pow(2).mean(dim=-1)
        rollout_errors.append(err)

    return onestep_mse, rollout_errors


def run(data_path, model_policy):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load normalization stats
    stats_path = Path(data_path).parent / "norm_stats.json"
    norm_stats = load_norm_stats(stats_path) if stats_path.exists() else None

    # Load test data — full 16-step trajectories
    history_size = 3
    max_rollout = 12
    dataset = RFSpectralDataset(
        data_path, history_size=history_size,
        num_preds=16 - history_size, norm_stats=norm_stats,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0,
    )

    # Load trained model
    model = swm.policy.AutoCostModel(model_policy)
    model = model.to(device)
    model.requires_grad_(False)

    # Define baselines
    def copy_last(ctx):
        return ctx[:, -1]  # just copy last frame

    def mean_ctx(ctx):
        return ctx.mean(dim=1)  # average context

    def zero_pred(ctx):
        return torch.zeros_like(ctx[:, 0])  # predict zero

    def model_pred(ctx):
        return model.predict(ctx)[:, -1]  # trained model, last output

    baselines = {
        "RF-LeWM (trained)": model_pred,
        "Copy-last": copy_last,
        "Mean-context": mean_ctx,
        "Zero": zero_pred,
    }

    # Collect results
    results = {name: {"onestep": [], "rollout": {t: [] for t in range(max_rollout)}}
               for name in baselines}

    start = time.time()
    for batch in loader:
        obs = batch["observations"].to(device)
        info = {"observations": obs}
        info = model.encode_rf(info)
        all_emb = info["emb"]  # (B, T, D)

        for name, pred_fn in baselines.items():
            with torch.no_grad():
                onestep, rollout = evaluate_rollout(
                    all_emb, history_size, max_rollout, pred_fn
                )
            results[name]["onestep"].append(onestep.cpu())
            for t, err in enumerate(rollout):
                results[name]["rollout"][t].append(err.cpu())

    elapsed = time.time() - start

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"RF-LeWM Baseline Comparison (test set, {len(dataset)} samples)")
    print(f"{'='*70}")

    # One-step MSE
    print(f"\n{'Method':<25s} {'1-step MSE':>10s} {'12-step MSE':>12s} {'Improvement':>12s}")
    print("-" * 62)

    copy_last_1step = torch.cat(results["Copy-last"]["onestep"]).mean().item()

    for name in baselines:
        mse_1 = torch.cat(results[name]["onestep"]).mean().item()
        mse_12 = torch.cat(results[name]["rollout"][max_rollout - 1]).mean().item()
        if name == "Copy-last":
            imp = "baseline"
        else:
            imp = f"{(1 - mse_1/copy_last_1step)*100:+.1f}%"
        print(f"{name:<25s} {mse_1:>10.4f} {mse_12:>12.4f} {imp:>12s}")

    # Rollout curves
    print(f"\nRollout error by horizon:")
    print(f"{'Step':>4s}", end="")
    for name in baselines:
        print(f"  {name:>18s}", end="")
    print()
    print("-" * (4 + 20 * len(baselines)))

    for t in range(max_rollout):
        print(f"{t+1:>4d}", end="")
        for name in baselines:
            mse = torch.cat(results[name]["rollout"][t]).mean().item()
            print(f"  {mse:>18.4f}", end="")
        print()

    print(f"\nTime: {elapsed:.1f}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_policy", type=str, default="lewm_rf_epoch_99")
    args = parser.parse_args()
    run(args.data_path, args.model_policy)
