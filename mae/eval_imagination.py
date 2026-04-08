"""Stage 5: Quality gate -- full test-set assessment of the imagination pipeline.

Reports MAE reconstruction quality, bridge fidelity, imagined future quality,
surprise detection rates, and per-regime SSIM breakdown.

Usage:
    python mae/eval_imagination.py
    python mae/eval_imagination.py --test_data /path/to/test.h5
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from mae import build_mae
from train_bridge import LatentBridge
from imagine import RFWorldModelImagination
from perturbations import noise_burst, signal_injection, signal_dropout, frequency_shift, temporal_reversal


# ---------------------------------------------------------------------------
# Datasets for batch assessment
# ---------------------------------------------------------------------------

class LogMagFrameDataset(Dataset):
    """Single frames for MAE direct reconstruction assessment."""

    def __init__(self, h5_path, vmin, vmax):
        with h5py.File(h5_path, "r") as f:
            data = f["logmag"][()]
        N, T = data.shape[:2]
        self.data = data.reshape(N * T, 256, 51).astype(np.float32)
        self.vmin = vmin
        self.scale = max(vmax - vmin, 1e-8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame = self.data[idx]
        frame = np.clip((frame - self.vmin) / self.scale, 0.0, 1.0)
        return torch.from_numpy(frame).unsqueeze(0)  # (1, 256, 51)


class PairedDataset(Dataset):
    """Paired (embedding, logmag_frame) for bridge assessment."""

    def __init__(self, emb_h5, logmag_h5, vmin, vmax):
        with h5py.File(emb_h5, "r") as f:
            emb = f["embeddings"][()]
        with h5py.File(logmag_h5, "r") as f:
            logmag = f["logmag"][()]
        N, T = emb.shape[:2]
        self.emb = emb.reshape(N * T, -1).astype(np.float32)
        logmag_flat = logmag.reshape(N * T, 256, 51).astype(np.float32)
        scale = max(vmax - vmin, 1e-8)
        self.frames = np.clip((logmag_flat - vmin) / scale, 0.0, 1.0)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.emb[idx]),
            torch.from_numpy(self.frames[idx]).unsqueeze(0),
        )


# ---------------------------------------------------------------------------
# Main assessment
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    mae_dir = Path(args.mae_dir)

    # Load norm stats
    with open(mae_dir / "cache" / "norm_stats.json") as f:
        norm = json.load(f)
    vmin, vmax = norm["min"], norm["max"]

    # Load MAE
    mae_model = build_mae().to(device)
    mae_model.load_state_dict(torch.load(mae_dir / "mae_best.ckpt", map_location=device, weights_only=True))
    mae_model.requires_grad_(False)

    # Load bridge
    bridge = LatentBridge(wm_dim=192, mae_dim=256, num_patches=272).to(device)
    bridge.load_state_dict(torch.load(mae_dir / "bridge_best.ckpt", map_location=device, weights_only=True))
    bridge.requires_grad_(False)

    # ===================================================================
    # Section 1: MAE direct reconstruction
    # ===================================================================
    print("\n--- MAE Direct Reconstruction ---")
    test_frames_ds = LogMagFrameDataset(str(cache_dir / "logmag_test.h5"), vmin, vmax)
    frame_loader = DataLoader(test_frames_ds, batch_size=512, num_workers=4, pin_memory=True)

    from torchmetrics.image import StructuralSimilarityIndexMeasure
    mae_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mae_mse_total = 0.0
    mae_n = 0

    mae_model.eval()
    with torch.no_grad():
        for batch in frame_loader:
            batch = batch.to(device)
            recon = mae_model.reconstruct(batch).unsqueeze(1).clamp(0, 1)
            mae_ssim_metric.update(recon, batch)
            mae_mse_total += torch.nn.functional.mse_loss(recon, batch).item()
            mae_n += 1

    mae_ssim = mae_ssim_metric.compute().item()
    mae_mse = mae_mse_total / mae_n
    print(f"  SSIM: {mae_ssim:.3f}  MSE: {mae_mse:.5f}")

    # ===================================================================
    # Section 2: Bridge reconstruction
    # ===================================================================
    print("\n--- Bridge Reconstruction ---")
    paired_ds = PairedDataset(
        str(cache_dir / "embeddings_test.h5"),
        str(cache_dir / "logmag_test.h5"),
        vmin, vmax,
    )
    paired_loader = DataLoader(paired_ds, batch_size=512, num_workers=4, pin_memory=True)

    bridge_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    bridge_mse_total = 0.0
    bridge_n = 0

    bridge.eval()
    with torch.no_grad():
        for emb_batch, frame_batch in paired_loader:
            emb_batch = emb_batch.to(device)
            frame_batch = frame_batch.to(device)
            tokens = bridge(emb_batch)
            recon = mae_model.decode(tokens).unsqueeze(1).clamp(0, 1)
            bridge_ssim_metric.update(recon, frame_batch)
            bridge_mse_total += torch.nn.functional.mse_loss(recon, frame_batch).item()
            bridge_n += 1

    bridge_ssim = bridge_ssim_metric.compute().item()
    bridge_mse = bridge_mse_total / bridge_n
    bridge_fidelity = bridge_ssim / max(mae_ssim, 1e-8)
    print(f"  SSIM: {bridge_ssim:.3f}  MSE: {bridge_mse:.5f}  Fidelity: {bridge_fidelity:.2f}x")

    # ===================================================================
    # Section 3: Imagined future (predicted WM embedding -> bridge -> MAE decoder)
    # ===================================================================
    print("\n--- Imagined Future ---")

    # Load imagination pipeline
    pipeline = RFWorldModelImagination(
        wm_checkpoint=args.wm_ckpt,
        mae_checkpoint=str(mae_dir / "mae_best.ckpt"),
        bridge_checkpoint=str(mae_dir / "bridge_best.ckpt"),
        norm_stats_path=args.wm_norm_stats,
        mae_norm_stats_path=str(mae_dir / "cache" / "norm_stats.json"),
        device=str(device),
    )

    # Load test trajectories
    with h5py.File(args.test_data, "r") as f:
        test_obs = torch.from_numpy(f["observations"][()]).float()  # [N, 16, 256, 51, 2]
        source_ids = None
        if "source_ids" in f:
            source_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in f["source_ids"][()]]

    N = test_obs.shape[0]
    context_len = 4

    # Load scene metadata for regime breakdown
    regime_map = {}
    metadata_path = Path(args.metadata_path)
    if metadata_path.exists() and source_ids is not None:
        with open(metadata_path) as f:
            scene_meta = json.load(f)
        for i, sid in enumerate(source_ids):
            regime_map[i] = scene_meta.get(sid, {}).get("regime", "unknown")

    # Batch process trajectories
    imagined_ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    imagined_mse_total = 0.0
    imagined_n = 0
    regime_ssim_accum = defaultdict(list)

    print(f"  Processing {N} trajectories...")
    t0 = time.time()

    for i in range(N):
        obs_i = test_obs[i]  # (16, 256, 51, 2)
        result = pipeline.imagine(obs_i, context_len=context_len)

        gt = result["ground_truth_spectrograms"][context_len:]  # (12, 256, 51)
        imagined = result["imagined_spectrograms"]              # (12, 256, 51)

        # SSIM computation
        gt_4d = gt.unsqueeze(1).to(device)
        im_4d = imagined.unsqueeze(1).clamp(0, 1).to(device)

        imagined_ssim_metric.update(im_4d, gt_4d)
        imagined_mse_total += torch.nn.functional.mse_loss(im_4d, gt_4d).item()
        imagined_n += 1

        # Per-regime SSIM
        regime = regime_map.get(i, "unknown")
        local_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        local_ssim.update(im_4d, gt_4d)
        regime_ssim_accum[regime].append(local_ssim.compute().item())

        if (i + 1) % 500 == 0 or i == N - 1:
            elapsed = time.time() - t0
            print(f"    {i+1}/{N} ({elapsed:.0f}s)")

    imagined_ssim = imagined_ssim_metric.compute().item()
    imagined_mse = imagined_mse_total / max(imagined_n, 1)
    imagined_fidelity = imagined_ssim / max(mae_ssim, 1e-8)

    print(f"  SSIM: {imagined_ssim:.3f}  MSE: {imagined_mse:.5f}  Fidelity: {imagined_fidelity:.2f}x")

    # ===================================================================
    # Section 4: Surprise detection
    # ===================================================================
    print("\n--- Surprise Detection ---")

    perturbation_fns = {
        "Noise burst": lambda obs, t: noise_burst(obs, t, intensity=3.0),
        "Signal injection": lambda obs, t: signal_injection(obs, t, freq_center=128, bandwidth=10, power=1.0, duration=4),
        "Signal dropout": lambda obs, t: signal_dropout(obs, t, freq_range=(50, 200), duration=4),
        "Frequency shift": lambda obs, t: frequency_shift(obs, t, shift_bins=20),
        "Temporal reversal": lambda obs, t: temporal_reversal(obs, t, min(t + 4, 16)),
    }

    # Sample subset for perturbation testing
    n_perturb_samples = min(args.n_perturb, N)
    perturb_indices = np.random.RandomState(42).choice(N, n_perturb_samples, replace=False)

    detection_results = {}
    for pname, pfn in perturbation_fns.items():
        detections = 0
        surprise_ratios = []

        for idx in perturb_indices:
            obs_i = test_obs[idx]
            result = pipeline.imagine_perturbed(obs_i, pfn, perturb_at_step=8, context_len=context_len)

            if result["detection"]:
                detections += 1

            # Surprise ratio: perturbed mean surprise / unperturbed mean surprise
            unp_mean = result["unperturbed"]["surprise_scores"].mean().item()
            p_mean = result["perturbed"]["surprise_scores"].mean().item()
            ratio = p_mean / max(unp_mean, 1e-8)
            surprise_ratios.append(ratio)

        det_rate = 100.0 * detections / n_perturb_samples
        mean_ratio = np.mean(surprise_ratios)
        detection_results[pname] = {"ratio": mean_ratio, "rate": det_rate}
        print(f"  {pname:<25s} {mean_ratio:.2f}x surprise ratio  ({det_rate:.1f}% detection)")

    # ===================================================================
    # Final report
    # ===================================================================
    print()
    print("=" * 60)
    print("=== IMAGINATION QUALITY REPORT ===")
    print("=" * 60)

    print(f"\nMAE Reconstruction (direct):")
    print(f"  SSIM:     {mae_ssim:.3f}")
    print(f"  MSE:      {mae_mse:.5f}")

    print(f"\nBridge Reconstruction (wm embedding -> MAE decoder):")
    print(f"  SSIM:     {bridge_ssim:.3f}")
    print(f"  MSE:      {bridge_mse:.5f}")
    print(f"  Fidelity: {bridge_fidelity:.2f}x vs direct MAE")

    print(f"\nImagined Future (predicted wm embedding -> bridge -> MAE decoder):")
    print(f"  SSIM:     {imagined_ssim:.3f}  (vs ground truth)")
    print(f"  MSE:      {imagined_mse:.5f}")
    print(f"  Fidelity: {imagined_fidelity:.2f}x vs direct MAE")

    print(f"\nSurprise Detection (imagined future):")
    for pname, res in detection_results.items():
        print(f"  {pname + ':':<25s} {res['ratio']:.2f}x surprise ratio  ({res['rate']:.1f}% detection)")

    regime_order = [
        "quiet", "dense", "bursty", "ramp_up", "interference_event",
        "correlated_alternating", "correlated_leader_follower", "random",
    ]
    print(f"\nPer-regime SSIM (imagined future):")
    for regime in regime_order:
        vals = regime_ssim_accum.get(regime, [])
        if vals:
            mean_ssim = np.mean(vals)
            print(f"  {regime + ':':<35s} {mean_ssim:.3f}")
        else:
            print(f"  {regime + ':':<35s} (no samples)")

    # Handle extra regimes not in the standard list
    for regime in sorted(regime_ssim_accum.keys()):
        if regime not in regime_order and regime != "unknown":
            vals = regime_ssim_accum[regime]
            print(f"  {regime + ':':<35s} {np.mean(vals):.3f}")

    # Save results to JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    report = {
        "mae_direct": {"ssim": mae_ssim, "mse": mae_mse},
        "bridge": {"ssim": bridge_ssim, "mse": bridge_mse, "fidelity": bridge_fidelity},
        "imagined": {"ssim": imagined_ssim, "mse": imagined_mse, "fidelity": imagined_fidelity},
        "detection": {k: {"ratio": v["ratio"], "rate": v["rate"]} for k, v in detection_results.items()},
        "per_regime_ssim": {r: float(np.mean(v)) for r, v in regime_ssim_accum.items() if v},
    }
    out_path = results_dir / "imagination_quality.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Imagination quality gate")
    parser.add_argument("--cache_dir", default="decoder/cache")
    parser.add_argument("--mae_dir", default="mae")
    parser.add_argument("--test_data", default="/workspace/data/test.h5")
    parser.add_argument("--wm_ckpt", default="/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt")
    parser.add_argument("--wm_norm_stats", default="/workspace/data/norm_stats.json")
    parser.add_argument("--metadata_path", default="scene_metadata.json")
    parser.add_argument("--n_perturb", type=int, default=200, help="Trajectories for perturbation testing")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
