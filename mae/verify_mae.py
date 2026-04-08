"""MAE verification script -- run all checks before proceeding to bridge training."""

import sys
import json
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from mae import build_mae


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load norm stats
    with open("mae/cache/norm_stats.json") as f:
        stats = json.load(f)
    norm_min, norm_max = stats["min"], stats["max"]
    scale = norm_max - norm_min

    # Load checkpoint
    model = build_mae().to(device)
    state = torch.load("mae/mae_best.ckpt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Load val frames from different trajectories
    with h5py.File("decoder/cache/logmag_val.h5", "r") as f:
        raw_frames = torch.tensor(f["logmag"][[0, 100, 500, 1000], 0]).float()  # (4, 256, 51)

    # Normalize
    frames_norm = ((raw_frames - norm_min) / scale).clamp(0, 1)
    frames_norm = frames_norm.unsqueeze(1).to(device)  # (4, 1, 256, 51)

    # ===== STEP 2: Identity check =====
    print("=" * 60)
    print("STEP 2: IDENTITY CHECK")
    print("=" * 60)
    with torch.no_grad():
        recon = model.reconstruct(frames_norm)  # (4, 256, 51)

    # Match shapes for comparison
    recon_4d = recon.unsqueeze(1).clamp(0, 1)  # (4, 1, 256, 51)

    identical = torch.allclose(frames_norm, recon_4d, atol=1e-4)
    max_diff = (frames_norm - recon_4d).abs().max().item()
    mean_diff = (frames_norm - recon_4d).abs().mean().item()

    print(f"Input shape:  {frames_norm.shape}")
    print(f"Recon shape:  {recon.shape}")
    print(f"Are identical (atol=1e-4): {identical}")
    print(f"Max absolute difference:   {max_diff:.6f}")
    print(f"Mean absolute difference:  {mean_diff:.6f}")
    print()
    for i in range(4):
        inp_mean = frames_norm[i].mean().item()
        inp_std = frames_norm[i].std().item()
        rec_mean = recon_4d[i].mean().item()
        rec_std = recon_4d[i].std().item()
        print(f"Frame {i}: input mean={inp_mean:.4f} std={inp_std:.4f} | "
              f"recon mean={rec_mean:.4f} std={rec_std:.4f}")

    if identical:
        print("\nFAIL: Reconstruction is identical to input -- this is a bug!")
        return
    if max_diff < 0.001:
        print("\nWARNING: Max diff < 0.001 -- suspiciously close to identity")
    else:
        print(f"\nPASS: Max diff = {max_diff:.4f} (> 0.05 threshold)")

    # ===== STEP 3: Normalization sanity check =====
    print()
    print("=" * 60)
    print("STEP 3: NORMALIZATION SANITY CHECK")
    print("=" * 60)
    print(f"Norm min: {norm_min:.4f}, Norm max: {norm_max:.4f}, Scale: {scale:.4f}")
    print(f"Normalized input mean: {frames_norm.mean().item():.4f}")
    print(f"Normalized input std:  {frames_norm.std().item():.4f}")
    frac_gt09 = (frames_norm > 0.9).float().mean().item()
    frac_gt05 = (frames_norm > 0.5).float().mean().item()
    frac_lt01 = (frames_norm < 0.1).float().mean().item()
    print(f"Fraction of pixels > 0.9: {frac_gt09:.4f}")
    print(f"Fraction of pixels > 0.5: {frac_gt05:.4f}")
    print(f"Fraction of pixels < 0.1: {frac_lt01:.4f}")

    # Also check the reconstruction stats
    print(f"\nRecon mean: {recon_4d.mean().item():.4f}")
    print(f"Recon std:  {recon_4d.std().item():.4f}")

    if frames_norm.std().item() < 0.15:
        print("\nWARNING: Input std < 0.15 -- normalization may be collapsing dynamic range")
    else:
        print(f"\nPASS: Input std = {frames_norm.std().item():.4f} (> 0.15 threshold)")

    if frac_gt09 > 0.5:
        print("WARNING: >50% of pixels above 0.9 -- near-constant distribution")

    # ===== STEP 4: Visual inspection =====
    print()
    print("=" * 60)
    print("STEP 4: VISUAL INSPECTION")
    print("=" * 60)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 4, figsize=(20, 12))

        for row in range(3):
            inp = frames_norm[row, 0].cpu().numpy()
            rec = recon_4d[row, 0].cpu().numpy()
            err = np.abs(inp - rec)

            axes[row, 0].imshow(inp, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
            axes[row, 0].set_title(f"Frame {row}: Input")

            axes[row, 1].imshow(rec, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
            axes[row, 1].set_title(f"Frame {row}: Reconstruction")

            axes[row, 2].imshow(err, aspect="auto", origin="lower", cmap="hot")
            axes[row, 2].set_title(f"Frame {row}: Error (max={err.max():.3f})")

            axes[row, 3].hist(err.flatten(), bins=50)
            axes[row, 3].set_title(f"Frame {row}: Error distribution")

        plt.tight_layout()
        out_path = "mae/verification_visual.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Visualization failed: {e}")

    # ===== STEP 5: Independent SSIM recomputation =====
    print()
    print("=" * 60)
    print("STEP 5: INDEPENDENT SSIM RECOMPUTATION")
    print("=" * 60)
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # Load 200 val frames from diverse trajectories
    with h5py.File("decoder/cache/logmag_val.h5", "r") as f:
        # Sample from different trajectory indices
        indices = np.linspace(0, f["logmag"].shape[0] - 1, 200, dtype=int)
        val_raw = torch.tensor(f["logmag"][indices, 0]).float()  # (200, 256, 51)

    val_norm = ((val_raw - norm_min) / scale).clamp(0, 1).unsqueeze(1).to(device)

    with torch.no_grad():
        # Process in batches to avoid OOM
        all_recon = []
        for i in range(0, 200, 50):
            batch = val_norm[i:i+50]
            r = model.reconstruct(batch).unsqueeze(1).clamp(0, 1)
            all_recon.append(r)
        val_recon = torch.cat(all_recon, dim=0)

    ssim_score = ssim_metric(val_recon, val_norm)
    mse_score = ((val_recon - val_norm) ** 2).mean()

    print(f"SSIM (200 val frames): {ssim_score.item():.4f}")
    print(f"MSE  (200 val frames): {mse_score.item():.6f}")
    print(f"Recon mean: {val_recon.mean().item():.4f}  std: {val_recon.std().item():.4f}")
    print(f"Input mean: {val_norm.mean().item():.4f}   std: {val_norm.std().item():.4f}")

    if abs(ssim_score.item() - 0.9678) > 0.05:
        print(f"\nWARNING: Independent SSIM ({ssim_score.item():.4f}) differs from training ({0.9678}) by > 0.05")
    else:
        print(f"\nPASS: Independent SSIM ({ssim_score.item():.4f}) consistent with training (0.9678)")

    # ===== STEP 6: Signal structure test =====
    print()
    print("=" * 60)
    print("STEP 6: SIGNAL STRUCTURE TEST")
    print("=" * 60)

    # Find trajectories with high vs low temporal variance
    with h5py.File("decoder/cache/logmag_val.h5", "r") as f:
        variances = []
        for i in range(min(50, f["logmag"].shape[0])):
            traj = f["logmag"][i]  # (16, 256, 51)
            var = np.var(traj, axis=0).mean()
            variances.append((i, var))
        variances.sort(key=lambda x: -x[1])
        high_var_idx = variances[0][0]
        low_var_idx = variances[-1][0]

        print(f"High-variance trajectory: idx={high_var_idx} (temporal var={variances[0][1]:.4f})")
        print(f"Low-variance trajectory:  idx={low_var_idx} (temporal var={variances[-1][1]:.4f})")

        # Load frames from high-variance trajectory: early vs late
        traj = torch.tensor(f["logmag"][high_var_idx]).float()  # (16, 256, 51)

    frame_early = ((traj[0] - norm_min) / scale).clamp(0, 1).unsqueeze(0).unsqueeze(0).to(device)
    frame_late = ((traj[8] - norm_min) / scale).clamp(0, 1).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        recon_early = model.reconstruct(frame_early).unsqueeze(1).clamp(0, 1)
        recon_late = model.reconstruct(frame_late).unsqueeze(1).clamp(0, 1)

    # Check if reconstruction captures the structural difference
    input_diff = (frame_late - frame_early).abs().mean().item()
    recon_diff = (recon_late - recon_early).abs().mean().item()

    print(f"\nInput difference (frame 0 vs frame 8):  {input_diff:.4f}")
    print(f"Recon difference (frame 0 vs frame 8):  {recon_diff:.4f}")
    print(f"Ratio (recon_diff / input_diff):         {recon_diff / max(input_diff, 1e-8):.4f}")

    if recon_diff / max(input_diff, 1e-8) < 0.3:
        print("\nWARNING: Reconstruction doesn't capture temporal structure change")
    else:
        print("\nPASS: Reconstruction preserves temporal structure differences")

    # Save signal structure visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for col, (frame, rec, label) in enumerate([
            (frame_early, recon_early, "Frame 0 (early)"),
            (frame_late, recon_late, "Frame 8 (late)"),
        ]):
            inp_np = frame[0, 0].cpu().numpy()
            rec_np = rec[0, 0].cpu().numpy()

            axes[0, col].imshow(inp_np, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
            axes[0, col].set_title(f"Input: {label}")
            axes[1, col].imshow(rec_np, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
            axes[1, col].set_title(f"Recon: {label}")

        # Difference maps
        diff_input = np.abs(frame_early[0, 0].cpu().numpy() - frame_late[0, 0].cpu().numpy())
        diff_recon = np.abs(recon_early[0, 0].cpu().numpy() - recon_late[0, 0].cpu().numpy())
        axes[0, 2].imshow(diff_input, aspect="auto", origin="lower", cmap="hot")
        axes[0, 2].set_title("Input diff (frame 0 vs 8)")
        axes[1, 2].imshow(diff_recon, aspect="auto", origin="lower", cmap="hot")
        axes[1, 2].set_title("Recon diff (frame 0 vs 8)")

        plt.tight_layout()
        out_path = "mae/verification_signal_structure.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")
    except Exception as e:
        print(f"Visualization failed: {e}")

    # ===== STEP 7: Verdict =====
    print()
    print("=" * 60)
    print("STEP 7: VERDICT")
    print("=" * 60)

    issues = []
    if identical:
        issues.append("CRITICAL: reconstruction identical to input")
    if max_diff < 0.001:
        issues.append("reconstruction suspiciously close to input")
    if frames_norm.std().item() < 0.15:
        issues.append(f"input std too low ({frames_norm.std().item():.4f})")
    if frac_gt09 > 0.5:
        issues.append(f">{frac_gt09*100:.0f}% pixels above 0.9")
    if abs(ssim_score.item() - 0.9678) > 0.05:
        issues.append(f"SSIM mismatch: independent={ssim_score.item():.4f} vs training=0.9678")
    if recon_diff / max(input_diff, 1e-8) < 0.3:
        issues.append("reconstruction doesn't capture temporal changes")

    if not issues:
        print("VERDICT: VERIFIED")
        print("All checks passed. Proceed to bridge training.")
    elif any("CRITICAL" in i for i in issues):
        print("VERDICT: BUG FOUND")
        for issue in issues:
            print(f"  - {issue}")
        print("Do NOT proceed to bridge training.")
    else:
        print("VERDICT: PARTIALLY VERIFIED")
        for issue in issues:
            print(f"  - {issue}")
        print("Proceed to bridge training with caveats noted above.")


if __name__ == "__main__":
    main()
