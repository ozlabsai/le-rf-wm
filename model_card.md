---
license: mit
tags:
  - world-model
  - jepa
  - rf
  - spectrogram
  - spectral-dynamics
  - pytorch
datasets:
  - OzLabs/rf-spectral-trajectories
library_name: pytorch
pipeline_tag: other
---

# RF-LeWM v0

A **Joint-Embedding Predictive Architecture (JEPA)** world model for RF spectral environments, adapted from [LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes et al., 2026).

RF-LeWM learns latent dynamics of RF spectral scenes from STFT spectrogram trajectories — predicting how the spectral environment evolves over time without requiring agent actions or interactive environments.

## Model Description

- **Architecture:** SpectrogramViT encoder (12-layer, 192-dim, 272 patches via 16x3 Conv2d) + 6-layer unconditional AR transformer predictor
- **Parameters:** 16.4M total (5.4M encoder, 9.5M predictor, 1.6M projectors)
- **Training objective:** L2-normalized residual prediction loss + SIGReg Gaussian regularizer + VICReg variance floor
- **Input:** Complex STFT spectrograms `(2, 256, 51)` — real/imaginary channels, 256 frequency bins, 51 time bins
- **Output:** 192-dimensional latent embeddings per timestep

## Key Design Choices

| Choice | Why |
|--------|-----|
| Mean pooling (not CLS token) | CLS token has zero variance across batch at random init |
| LayerNorm projectors (not BatchNorm) | BN running stats diverge with randomly-initialized encoders |
| Residual prediction (predict delta-z, not z) | Removes scene-specific memorization shortcut |
| L2-normalized loss | Prevents magnitude shrinkage shortcut |
| Stop-gradient on targets | Prevents encoder collapse |
| VICReg variance loss | Ensures embedding diversity with proper gradient at collapse |

## Training

Trained for 100 epochs on [OzLabs/rf-spectral-trajectories](https://huggingface.co/datasets/OzLabs/rf-spectral-trajectories) (13,841 training trajectories, 16 timesteps each at 80ms resolution).

- Optimizer: AdamW, lr=5e-5, weight_decay=1e-3
- Precision: bf16 mixed precision
- Batch size: 128
- Hardware: 1x NVIDIA A100-SXM4-80GB
- Training time: ~6 hours
- WandB: [training logs](https://wandb.ai/guy-na8/lewm-rf/runs/6abzuaa0)

## Results

Compared against trivial baselines on the test set (2,999 trajectories from held-out scenes):

### MSE (embedding space, lower is better)

| Method | 1-step | 12-step | vs Copy-last |
|--------|--------|---------|-------------|
| **RF-LeWM** | **1.289** | **1.656** | **+42.6%** |
| Copy-last | 2.245 | 2.304 | baseline |
| Mean-context | 1.501 | 1.618 | +33.1% |

### Cosine Similarity (directional prediction, higher is better)

| Method | CosSim |
|--------|--------|
| **RF-LeWM** | **0.076** |
| Mean-context | 0.056 |
| Copy-last | 0.038 |

RF-LeWM achieves the highest cosine similarity of all methods, demonstrating learned temporal dynamics that generalize to unseen RF scenes. Beats copy-last on MSE in 58/60 test scenes.

Rollout error degrades gracefully over 12 autoregressive steps (1.29 to 1.66), indicating stable latent dynamics without collapse.

## Usage

```python
import torch
import stable_worldmodel as swm

# Load model
model = swm.policy.AutoCostModel("RF-LeWM-v0/lewm_rf_epoch_99")
model.requires_grad_(False)

# Encode RF spectrograms
# obs: (batch, timesteps, 2, 256, 51) — normalized complex STFT
info = {"observations": obs}
info = model.encode_rf(info)
emb = info["emb"]  # (batch, timesteps, 192)

# Predict future embeddings
ctx = emb[:, :3]  # 3-frame context
pred = model.predict(ctx)  # predicted embeddings

# Autoregressive rollout
rolled = model.rollout_unconditional(ctx, n_steps=12, history_size=3)
```

## Limitations

- Operates in latent space only — no decoder to reconstruct spectrograms
- Cosine similarity is modest (0.076) — the model captures temporal dynamics but has room for improvement
- Trained on a specific set of RF scenes; generalization to very different RF environments is untested
- Requires `stable-worldmodel` and `stable-pretraining` packages for loading

## Citation

```bibtex
@misc{ozlabs2026rflewm,
  title={RF-LeWM: JEPA World Model for RF Spectral Environments},
  author={OzLabs},
  year={2026},
  url={https://huggingface.co/OzLabs/RF-LeWM-v0}
}
```

Based on:
```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End JEPA from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Repository

Source code: [github.com/ozlabsai/le-rf-wm](https://github.com/ozlabsai/le-rf-wm)
