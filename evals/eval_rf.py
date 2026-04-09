"""RF-LeWM evaluation: prediction quality, rollout error, and surprise."""

import time
from pathlib import Path

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from dataset import RFSpectralDataset, load_norm_stats


@hydra.main(version_base=None, config_path="./config/eval", config_name="rf")
def run(cfg: DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to(device)
    model.requires_grad_(False)

    # load normalization stats (computed from train split)
    norm_stats = None
    if cfg.data.get("normalize", True):
        stats_path = Path(cfg.data.get("norm_stats_path",
            str(Path(cfg.data.test_path).parent / "norm_stats.json")))
        if stats_path.exists():
            norm_stats = load_norm_stats(stats_path)
            print(f"Loaded norm stats from {stats_path}")
        else:
            print(f"WARNING: normalize=True but {stats_path} not found, using raw data")

    # load full 16-step trajectories for evaluation
    history_size = cfg.data.history_size
    max_rollout = cfg.eval.max_rollout_steps

    dataset = RFSpectralDataset(
        cfg.data.test_path,
        history_size=history_size,
        num_preds=16 - history_size,  # full trajectory
        norm_stats=norm_stats,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # collect metrics
    onestep_errors = []
    rollout_errors = {t: [] for t in range(1, max_rollout + 1)}
    surprise_scores = []

    start_time = time.time()

    for batch in loader:
        obs = batch["observations"].to(device)  # (B, T, 2, 256, 51)
        T = obs.size(1)

        # encode all timesteps
        info = {"observations": obs}
        info = model.encode_rf(info)
        all_emb = info["emb"]  # (B, T, D)

        # one-step prediction from context
        ctx = all_emb[:, :history_size]
        pred_one = model.predict(ctx)[:, -1]  # (B, D)
        target_one = all_emb[:, history_size]  # (B, D)
        err = (pred_one - target_one).pow(2).mean(dim=-1)  # (B,)
        onestep_errors.append(err.cpu())

        # multi-step rollout
        n_steps = min(max_rollout, T - history_size)
        rolled = model.rollout_unconditional(
            ctx.clone(), n_steps=n_steps, history_size=history_size,
        )  # (B, history_size + n_steps, D)

        traj_surprise = []
        for t in range(1, n_steps + 1):
            pred_t = rolled[:, history_size + t - 1]  # (B, D)
            target_t = all_emb[:, history_size + t - 1]  # (B, D)
            err_t = (pred_t - target_t).pow(2).mean(dim=-1)  # (B,)
            rollout_errors[t].append(err_t.cpu())
            traj_surprise.append(err_t.cpu())

        # surprise: mean rollout error across steps per trajectory
        surprise = torch.stack(traj_surprise, dim=1).mean(dim=1)  # (B,)
        surprise_scores.append(surprise)

    elapsed = time.time() - start_time

    # aggregate
    onestep_mse = torch.cat(onestep_errors).mean().item()
    rollout_curve = {}
    for t in range(1, max_rollout + 1):
        if rollout_errors[t]:
            rollout_curve[t] = torch.cat(rollout_errors[t]).mean().item()
    all_surprise = torch.cat(surprise_scores)
    surprise_mean = all_surprise.mean().item()
    surprise_std = all_surprise.std().item()

    # print results
    print(f"\n{'='*50}")
    print(f"RF-LeWM Results ({cfg.policy})")
    print(f"{'='*50}")
    print(f"One-step prediction MSE: {onestep_mse:.6f}")
    print(f"\nRollout error by horizon:")
    for t, err in rollout_curve.items():
        print(f"  step {t:2d}: {err:.6f}")
    print(f"\nSurprise score: {surprise_mean:.6f} +/- {surprise_std:.6f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*50}\n")

    # save results
    results_path = Path(swm.data.utils.get_cache_dir()) / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"one_step_mse: {onestep_mse}\n")
        f.write(f"rollout_curve: {rollout_curve}\n")
        f.write(f"surprise_mean: {surprise_mean}\n")
        f.write(f"surprise_std: {surprise_std}\n")
        f.write(f"time: {elapsed}\n")


if __name__ == "__main__":
    run()
