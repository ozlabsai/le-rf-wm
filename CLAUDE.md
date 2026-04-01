# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of LeWorldModel (LeWM) adapted for RF spectral environments. The original LeWM is a JEPA world model for robotics; this fork adds an **RF pipeline** with a custom spectrogram ViT encoder and unconditional predictor for learning dynamics of RF spectral scenes.

Two parallel pipelines exist:
- **Robotics** (original): `train.py` / `eval.py` — action-conditioned, HF ViT encoder, MPC planning
- **RF** (new): `train_rf.py` / `eval_rf.py` — unconditional, SpectrogramViT encoder, prediction/rollout/surprise eval

## Setup

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

Datasets are HDF5 files stored under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`).

## Commands

### RF Training
```bash
python train_rf.py                            # train RF-LeWM (default config)
python train_rf.py trainer.max_epochs=50      # override via Hydra
```
Config: `config/train/lewm_rf.yaml`. Data paths in `config/train/data/rf.yaml`. Set WandB in the config.

### RF Evaluation
```bash
python eval_rf.py policy=lewm_rf              # evaluate checkpoint
python eval_baselines.py --data_path /path/to/test.h5 --model_policy lewm_rf_epoch_99
python eval_diagnostics.py --data_path /path/to/test.h5 --model_policy lewm_rf_epoch_99
```
`eval_rf.py` — rollout error and surprise scores. `eval_baselines.py` — comparison against copy-last/mean/zero baselines. `eval_diagnostics.py` — embedding space analysis, cosine similarity, per-regime breakdown.

### Robotics Training (original)
```bash
python train.py data=pusht                    # train on PushT
python train.py data=tworoom                  # train on TwoRoom
```
Config: `config/train/lewm.yaml`.

### Robotics Evaluation (original)
```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```
Policy path is relative to `$STABLEWM_HOME`, without `_object.ckpt` suffix.

## Architecture

**Core model** (`jepa.py`): `JEPA` class — shared by both pipelines. Key methods:
- `encode()` — robotics: ViT encodes pixels, action encoder embeds actions
- `encode_rf()` — RF: SpectrogramViT encodes STFT observations (no actions)
- `predict(emb, act_emb=None)` — AR predictor, unconditional when act_emb is None
- `rollout()` — action-conditioned multi-step rollout (robotics)
- `rollout_unconditional()` — autoregressive rollout without actions (RF)

**RF encoder** (`encoder.py`): `SpectrogramViT` — custom ViT for `(2, 256, 51)` spectrograms. Conv2d patch embedding with `(16, 3)` kernel → 272 patches. Mean-pooled patch embeddings (NOT CLS token). Separate learned frequency/time positional embeddings. Uses `Block` from module.py.

**RF dataset** (`dataset.py`): `RFSpectralDataset` — reads HDF5 directly, supports preloading entire dataset into RAM. Slices 16-step trajectories into subsequences. Global per-channel normalization from train split.

**Modules** (`module.py`): Shared building blocks:
- `ARPredictor` — supports both `ConditionalBlock` (robotics) and `Block` (RF) via `block_class` param
- `SIGReg` — Epps-Pulley Gaussian regularizer (domain-agnostic)
- `Block` / `ConditionalBlock` — standard vs AdaLN-zero transformer blocks
- `MLP` — projector heads (LayerNorm for RF, BatchNorm1d for robotics)

**RF Training** (`train_rf.py`): Key differences from original:
- **Residual prediction**: predicts `Δz = z_{t+k} - z_t` not absolute `z_{t+k}`
- **L2-normalized loss**: `MSE(normalize(pred_delta), normalize(tgt_delta))` — cosine-aware
- **Stop-gradient on targets**: `tgt_emb.detach()` prevents collapse
- **VICReg variance loss**: `sqrt(var + eps)` hinge prevents embedding collapse
- **SIGReg warmup schedule**: linear ramp from 0 → 0.05 over 20 epochs
- Loss: `pred_loss + sigreg_weight * sigreg_loss + variance_weight * var_loss`

**Evaluation**: `eval_rf.py` measures rollout error and surprise. `eval_baselines.py` compares against trivial baselines. `eval_diagnostics.py` provides cosine similarity, per-regime breakdown, and embedding statistics.

## Key Design Decisions

- **Mean pooling, not CLS token**: CLS token has zero variance across batch at random init (it's a shared parameter). Mean pooling preserves per-sample patch diversity through the transformer.
- **LayerNorm, not BatchNorm** in RF projectors: BN running stats diverge between train/eval with randomly-initialized encoders. LN has no running stats.
- **Residual prediction**: Predicting temporal change (Δz) removes scene-specific memorization shortcuts that absolute prediction (z) enables. The model must learn dynamics, not scene identity.
- **L2-normalized loss**: Prevents magnitude shrinkage shortcut where the predictor minimizes MSE by outputting low-magnitude vectors near zero.
- **No overlap between context and targets**: `tgt_emb = emb[:, ctx_len:]` ensures targets start after context ends. The original LeWM uses overlapping slices which work for action-conditioned prediction but create trivial shortcuts for unconditional prediction.
- Embeddings use `(B, T, D)` convention; projectors operate on flattened `(B*T, D)` via einops rearrange
- SIGReg operates on `(T, B, D)` transposed embeddings — note the transpose in forward functions
- Checkpoints are saved as serialized model objects (`torch.save(model, path)`) for `AutoCostModel` compatibility
- RF encoder uses factored positional embeddings: freq and time axes have separate learned embeddings broadcast-summed
- RF dataset is pre-split by source scene (train/val/test .h5 files) — no random splitting needed
