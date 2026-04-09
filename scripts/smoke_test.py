"""Smoke test for RF-LeWM: validates dataset, model, forward pass, and training step.

Run before committing to real training:
    python smoke_test.py --data_path /path/to/train.h5

Checks:
  1. Dataset loads and shapes are correct
  2. Normalization stats compute and apply without NaN/Inf
  3. Encoder produces finite embeddings of correct shape
  4. Full forward pass produces finite loss
  5. Backward pass produces finite gradients
  6. One optimizer step doesn't produce NaN
  7. SIGReg loss is finite and positive
  8. Predictor output differs from input (not pure persistence)
"""

import argparse
import sys
from pathlib import Path

import torch

from dataset import RFSpectralDataset, compute_norm_stats
from encoder import SpectrogramViT
from jepa import JEPA
from module import ARPredictor, Block, MLP, SIGReg


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    if not condition:
        return False
    return True


def run(data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    all_pass = True

    # --- 1. Dataset ---
    print("=== Dataset ===")
    stats = compute_norm_stats(data_path)
    all_pass &= check("norm stats computed",
        len(stats["mean"]) == 2 and len(stats["std"]) == 2,
        f"mean={stats['mean']}, std={stats['std']}")
    all_pass &= check("std > 0",
        all(s > 1e-8 for s in stats["std"]))

    ds = RFSpectralDataset(data_path, history_size=3, num_preds=1, norm_stats=stats)
    all_pass &= check("dataset length > 0", len(ds) > 0, f"len={len(ds)}")

    sample = ds[0]
    obs = sample["observations"]
    seq_len, n_channels, freq_bins, time_bins = obs.shape
    all_pass &= check("sample dims", obs.ndim == 4 and n_channels == 2,
        f"got {obs.shape} (T={seq_len}, C={n_channels}, F={freq_bins}, T_bins={time_bins})")
    all_pass &= check("sample dtype", obs.dtype == torch.float32)
    all_pass &= check("no NaN in sample", not torch.isnan(obs).any().item())
    all_pass &= check("no Inf in sample", not torch.isinf(obs).any().item())
    all_pass &= check("normalized (mean near 0)",
        abs(obs.mean().item()) < 5.0,
        f"mean={obs.mean().item():.4f}")

    # batch via dataloader
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch_obs = batch["observations"]
    all_pass &= check("batch shape",
        batch_obs.shape == (4, seq_len, n_channels, freq_bins, time_bins),
        f"got {batch_obs.shape}")

    # --- 2. Encoder ---
    print("\n=== Encoder ===")
    hidden_dim = 192
    embed_dim = 192
    patch_freq = 16
    patch_time = 3

    # validate patch divisibility
    assert freq_bins % patch_freq == 0, f"freq_bins {freq_bins} not divisible by patch_freq {patch_freq}"
    assert time_bins % patch_time == 0, f"time_bins {time_bins} not divisible by patch_time {patch_time}"
    n_patches = (freq_bins // patch_freq) * (time_bins // patch_time)
    print(f"  Data shape: ({n_channels}, {freq_bins}, {time_bins})")
    print(f"  Patches: {freq_bins // patch_freq} x {time_bins // patch_time} = {n_patches} + CLS = {n_patches + 1} tokens")

    encoder = SpectrogramViT(
        freq_bins=freq_bins, time_bins=time_bins,
        patch_freq=patch_freq, patch_time=patch_time,
        hidden_dim=hidden_dim, depth=4, heads=3, mlp_dim=768,
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    all_pass &= check("encoder created", True, f"{n_params:,} params")

    test_input = torch.randn(2, n_channels, freq_bins, time_bins, device=device)
    enc_out = encoder(test_input)
    all_pass &= check("encoder output shape", enc_out.shape == (2, hidden_dim),
        f"got {enc_out.shape}")
    all_pass &= check("encoder output finite", torch.isfinite(enc_out).all().item())
    all_pass &= check("encoder output not constant",
        enc_out.std().item() > 1e-6,
        f"std={enc_out.std().item():.6f}")

    # --- 3. Full model ---
    print("\n=== Full Model ===")
    predictor = ARPredictor(
        num_frames=3, input_dim=embed_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=2, heads=4, mlp_dim=512,
        dim_head=64, block_class=Block,
    ).to(device)

    projector = MLP(hidden_dim, 2048, embed_dim, norm_fn=torch.nn.BatchNorm1d).to(device)
    pred_proj = MLP(hidden_dim, 2048, embed_dim, norm_fn=torch.nn.BatchNorm1d).to(device)

    model = JEPA(
        encoder=encoder, predictor=predictor,
        projector=projector, pred_proj=pred_proj,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    all_pass &= check("model created", True, f"{total_params:,} total params")

    # --- 4. Forward pass ---
    print("\n=== Forward Pass ===")
    batch_dev = {"observations": batch_obs.to(device)}
    info = model.encode_rf(batch_dev)
    emb = info["emb"]
    all_pass &= check("embedding shape", emb.shape == (4, 4, embed_dim),
        f"got {emb.shape}")
    all_pass &= check("embedding finite", torch.isfinite(emb).all().item())

    ctx = emb[:, :3]
    tgt = emb[:, 1:]
    pred = model.predict(ctx)
    all_pass &= check("prediction shape", pred.shape == tgt.shape,
        f"pred={pred.shape}, tgt={tgt.shape}")
    all_pass &= check("prediction finite", torch.isfinite(pred).all().item())

    # persistence check: prediction should not be identical to last context frame
    last_ctx = ctx[:, -1:]
    pred_last = pred[:, -1:]
    persistence_diff = (pred_last - last_ctx).abs().mean().item()
    all_pass &= check("not pure persistence",
        persistence_diff > 1e-6,
        f"diff from last ctx={persistence_diff:.6f}")

    # --- 5. Loss ---
    print("\n=== Loss ===")
    pred_loss = (pred - tgt).pow(2).mean()
    all_pass &= check("pred_loss finite", torch.isfinite(pred_loss).item(),
        f"value={pred_loss.item():.6f}")

    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    sigreg_loss = sigreg(emb.transpose(0, 1))
    all_pass &= check("sigreg_loss finite", torch.isfinite(sigreg_loss).item(),
        f"value={sigreg_loss.item():.6f}")
    all_pass &= check("sigreg_loss positive", sigreg_loss.item() > 0)

    total_loss = pred_loss + 0.09 * sigreg_loss
    all_pass &= check("total loss finite", torch.isfinite(total_loss).item(),
        f"value={total_loss.item():.6f}")

    # --- 6. Backward ---
    print("\n=== Backward ===")
    total_loss.backward()
    grad_norms = []
    has_nan_grad = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms.append(gn)
            if not torch.isfinite(p.grad).all():
                has_nan_grad = True
    all_pass &= check("gradients computed", len(grad_norms) > 0,
        f"{len(grad_norms)} param groups")
    all_pass &= check("no NaN/Inf gradients", not has_nan_grad)
    all_pass &= check("gradients not all zero",
        max(grad_norms) > 1e-10,
        f"max grad norm={max(grad_norms):.6f}")

    # --- 7. Optimizer step ---
    print("\n=== Optimizer Step ===")
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    opt.step()
    opt.zero_grad()

    # check weights didn't go NaN
    has_nan_weight = False
    for p in model.parameters():
        if not torch.isfinite(p).all():
            has_nan_weight = True
            break
    all_pass &= check("weights finite after step", not has_nan_weight)

    # --- 8. Rollout ---
    print("\n=== Rollout ===")
    with torch.no_grad():
        ctx_emb = emb[:, :3].clone()
        rolled = model.rollout_unconditional(ctx_emb, n_steps=5, history_size=3)
    all_pass &= check("rollout shape", rolled.shape == (4, 8, embed_dim),
        f"got {rolled.shape}")
    all_pass &= check("rollout finite", torch.isfinite(rolled).all().item())

    # check rollout doesn't collapse to constant
    step_diffs = []
    for t in range(3, rolled.shape[1] - 1):
        d = (rolled[:, t+1] - rolled[:, t]).abs().mean().item()
        step_diffs.append(d)
    all_pass &= check("rollout not collapsed",
        max(step_diffs) > 1e-8,
        f"max step diff={max(step_diffs):.6f}")

    # --- Summary ---
    print(f"\n{'='*50}")
    if all_pass:
        print("ALL CHECKS PASSED -- ready for training")
    else:
        print("SOME CHECKS FAILED -- investigate before training")
    print(f"{'='*50}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to train.h5")
    args = parser.parse_args()
    sys.exit(run(args.data_path))
