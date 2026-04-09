"""Stage 4: Imagination pipeline -- visualize world model predictions in pixel space.

The pipeline chain:
  raw obs -> WM encoder -> embeddings -> WM predictor (rollout) -> predicted embeddings
  predicted embeddings -> bridge -> MAE patch tokens -> MAE decoder -> imagined spectrogram

Usage:
    python mae/imagine.py                                    # smoke test
    python mae/imagine.py --wm_ckpt /path/to/model.ckpt     # custom checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_bridge import LatentBridge
from perturbations import noise_burst, signal_injection, signal_dropout, frequency_shift, temporal_reversal
from dataset import load_norm_stats


class RFWorldModelImagination:
    """End-to-end pipeline for visualizing world model predictions."""

    def __init__(self, wm_checkpoint, bridge_checkpoint,
                 norm_stats_path, mae_norm_stats_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load world model (serialized full object)
        print(f"Loading world model from {wm_checkpoint}...")
        self.wm = torch.load(wm_checkpoint, map_location=self.device, weights_only=False)
        self.wm.to(self.device)
        self.wm.requires_grad_(False)

        # Load WM normalization stats (for raw observations)
        wm_stats = load_norm_stats(norm_stats_path)
        self.wm_mean = torch.tensor(wm_stats["mean"], dtype=torch.float32).view(1, 2, 1, 1).to(self.device)
        self.wm_std = torch.tensor(wm_stats["std"], dtype=torch.float32).view(1, 2, 1, 1).to(self.device)

        # Load bridge (per-patch MLP, direct pixel output)
        print(f"Loading bridge from {bridge_checkpoint}...")
        self.bridge = LatentBridge().to(self.device)
        self.bridge.load_state_dict(torch.load(bridge_checkpoint, map_location=self.device, weights_only=True))
        self.bridge.requires_grad_(False)

        # Load normalization stats for log-magnitude
        with open(mae_norm_stats_path) as f:
            mae_stats = json.load(f)
        self.logmag_min = mae_stats["min"]
        self.logmag_max = mae_stats["max"]
        self.logmag_scale = max(self.logmag_max - self.logmag_min, 1e-8)

        print("Imagination pipeline ready.")

    def _obs_to_logmag(self, obs):
        """Convert raw observations [T, 256, 51, 2] to log-magnitude [T, 256, 51]."""
        real, imag = obs[..., 0], obs[..., 1]
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
        logmag = torch.log(mag + 1e-6)
        return logmag

    def _normalize_logmag(self, logmag):
        """Normalize log-magnitude to [0, 1]."""
        return ((logmag - self.logmag_min) / self.logmag_scale).clamp(0, 1)

    def _encode_obs(self, obs):
        """Encode raw observations through the world model.

        Args:
            obs: (T, 256, 51, 2) raw observations
        Returns:
            embeddings: (T, 192)
        """
        # Reshape: (T, 256, 51, 2) -> (T, 2, 256, 51)
        x = obs.permute(0, 3, 1, 2).float().to(self.device)
        # Normalize for WM
        x = (x - self.wm_mean) / self.wm_std
        # Encode
        with torch.no_grad():
            emb = self.wm.encoder(x)        # (T, 192)
            emb = self.wm.projector(emb)     # (T, 192)
        return emb

    def _patches_to_spectrogram(self, patches):
        """Convert WM patch tokens to spectrogram via bridge.

        Args:
            patches: (N, 272, 192) WM encoder patch tokens
        Returns:
            spectrograms: (N, 256, 51) normalized [0, 1]
        """
        with torch.no_grad():
            specs = self.bridge(patches)  # (N, 256, 51)
        return specs.clamp(0, 1)

    def _encode_obs_patches(self, obs):
        """Encode raw observations to patch-level tokens.

        Args:
            obs: (T, 256, 51, 2) raw observations
        Returns:
            patches: (T, 272, 192)
        """
        x = obs.permute(0, 3, 1, 2).float().to(self.device)
        x = (x - self.wm_mean) / self.wm_std
        with torch.no_grad():
            p = self.wm.encoder.forward_patches(x)  # (T, 16, 17, 192)
            p = p.reshape(x.shape[0], -1, p.shape[-1])  # (T, 272, 192)
        return p

    @torch.no_grad()
    def imagine(self, observations, context_len=4):
        """Run imagination pipeline on a single trajectory.

        Args:
            observations: (T, 256, 51, 2) full trajectory, real/imag
            context_len: number of frames for predictor context

        Returns dict with:
            ground_truth_spectrograms: (T, 256, 51) log-magnitude, normalized [0,1]
            imagined_spectrograms: (T-context_len, 256, 51) model's imagined future
            surprise_scores: (T-context_len,) per-step surprise
            wm_embeddings: (T, 192) actual encoded embeddings
            wm_predicted_embeddings: (T-context_len, 192) predicted embeddings
        """
        obs = observations.float().to(self.device)

        # Ground truth log-magnitude spectrograms
        logmag = self._obs_to_logmag(obs)
        gt_specs = self._normalize_logmag(logmag)  # (T, 256, 51)

        # Encode all frames
        wm_emb = self._encode_obs(obs)  # (T, 192)

        # Rollout from context
        T = obs.shape[0]
        n_steps = T - context_len
        ctx = wm_emb[:context_len].unsqueeze(0)  # (1, ctx_len, 192)
        rolled = self.wm.rollout_unconditional(ctx, n_steps=n_steps, history_size=context_len)
        # rolled: (1, ctx_len + n_steps, 192)
        pred_emb = rolled[0, context_len:]  # (n_steps, 192)

        # Surprise: MSE between predicted and actual embeddings
        actual_future = wm_emb[context_len:]  # (n_steps, 192)
        surprise = (pred_emb - actual_future).pow(2).mean(dim=-1)  # (n_steps,)

        # Imagined spectrograms: broadcast predicted embedding to all patch positions
        pred_patches = pred_emb.unsqueeze(1).expand(-1, 272, -1)  # (n_steps, 272, 192)
        imagined = self._patches_to_spectrogram(pred_patches)  # (n_steps, 256, 51)

        return {
            "ground_truth_spectrograms": gt_specs.cpu(),
            "imagined_spectrograms": imagined.cpu(),
            "surprise_scores": surprise.cpu(),
            "wm_embeddings": wm_emb.cpu(),
            "wm_predicted_embeddings": pred_emb.cpu(),
        }

    @torch.no_grad()
    def imagine_perturbed(self, observations, perturbation_fn, perturb_at_step, context_len=4):
        """Compare unperturbed vs perturbed imagination.

        Args:
            observations: (T, 256, 51, 2) raw trajectory
            perturbation_fn: callable(obs, timestep) -> perturbed obs
            perturb_at_step: timestep at which to apply perturbation
            context_len: predictor context length

        Returns dict with:
            unperturbed: output of imagine(observations)
            perturbed: output of imagine(perturbed_observations)
            surprise_delta: per-step difference in surprise scores
            detection: bool -- True if max surprise_delta > 1.3x baseline std
        """
        # Unperturbed
        unperturbed = self.imagine(observations, context_len=context_len)

        # Perturbed
        perturbed_obs = perturbation_fn(observations, perturb_at_step)
        perturbed = self.imagine(perturbed_obs, context_len=context_len)

        # Surprise delta
        surprise_delta = perturbed["surprise_scores"] - unperturbed["surprise_scores"]

        # Detection criterion: max delta > 1.3x std of unperturbed surprise
        baseline_std = unperturbed["surprise_scores"].std().item()
        max_delta = surprise_delta.abs().max().item()
        detected = max_delta > 1.3 * baseline_std if baseline_std > 1e-8 else max_delta > 1e-6

        return {
            "unperturbed": unperturbed,
            "perturbed": perturbed,
            "surprise_delta": surprise_delta,
            "detection": detected,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Imagination Smoke Test (device: {device}) ===\n")

    # Load pipeline
    pipeline = RFWorldModelImagination(
        wm_checkpoint=args.wm_ckpt,
        bridge_checkpoint=args.bridge_ckpt,
        norm_stats_path=args.wm_norm_stats,
        mae_norm_stats_path=args.mae_norm_stats,
        device=device,
    )

    # Load one test trajectory
    print("\nLoading test trajectory...")
    with h5py.File(args.test_data, "r") as f:
        obs = torch.from_numpy(f["observations"][0]).float()  # (16, 256, 51, 2)
    print(f"Trajectory shape: {obs.shape}")

    # Run imagine
    print("\nRunning imagine()...")
    result = pipeline.imagine(obs, context_len=4)
    print(f"  ground_truth_spectrograms: {result['ground_truth_spectrograms'].shape}")
    print(f"  imagined_spectrograms:     {result['imagined_spectrograms'].shape}")
    print(f"  surprise_scores:           {result['surprise_scores'].shape}")
    print(f"  wm_embeddings:             {result['wm_embeddings'].shape}")
    print(f"  wm_predicted_embeddings:   {result['wm_predicted_embeddings'].shape}")
    surprise = result["surprise_scores"]
    print(f"  surprise mean={surprise.mean():.4f} std={surprise.std():.4f}")

    # Run imagine_perturbed with noise burst at step 8
    print("\nRunning imagine_perturbed() with noise_burst at step 8...")
    perturb_result = pipeline.imagine_perturbed(
        obs,
        perturbation_fn=lambda o, t: noise_burst(o, t, intensity=3.0),
        perturb_at_step=8,
        context_len=4,
    )
    print(f"  detection: {perturb_result['detection']}")
    print(f"  surprise_delta max: {perturb_result['surprise_delta'].abs().max():.4f}")
    print(f"  surprise_delta mean: {perturb_result['surprise_delta'].mean():.4f}")

    # Save side-by-side visualization
    print("\nGenerating visualization...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt = result["ground_truth_spectrograms"]
        imagined = result["imagined_spectrograms"]

        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        fig.suptitle("Ground Truth (top) vs Imagined (bottom)", fontsize=16)

        # Show 6 future timesteps (after context)
        for i in range(min(6, imagined.shape[0])):
            step = i
            # Ground truth (corresponding future frame)
            axes[0, i].imshow(gt[4 + step].numpy(), aspect="auto", origin="lower", cmap="viridis")
            axes[0, i].set_title(f"GT t={4+step}")
            axes[0, i].axis("off")
            # Imagined
            axes[1, i].imshow(imagined[step].numpy(), aspect="auto", origin="lower", cmap="viridis")
            axes[1, i].set_title(f"Imagined t={4+step}\nsurprise={surprise[step]:.3f}")
            axes[1, i].axis("off")

        plt.tight_layout()
        out_path = Path(args.mae_dir) / "smoke_test_imagination.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization to {out_path}")
    except ImportError:
        print("matplotlib not available, skipping visualization")

    print("\n=== SMOKE TEST PASSED ===")


def main():
    parser = argparse.ArgumentParser(description="Imagination pipeline smoke test")
    parser.add_argument("--wm_ckpt", default="/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt")
    parser.add_argument("--bridge_ckpt", default="mae/bridge_best.ckpt")
    parser.add_argument("--wm_norm_stats", default="/workspace/data/norm_stats.json")
    parser.add_argument("--mae_norm_stats", default="mae/cache/norm_stats.json")
    parser.add_argument("--test_data", default="/workspace/data/test.h5")
    parser.add_argument("--mae_dir", default="mae")
    args = parser.parse_args()
    smoke_test(args)


if __name__ == "__main__":
    main()
