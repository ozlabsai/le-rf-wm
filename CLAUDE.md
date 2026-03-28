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
Config: `config/train/lewm_rf.yaml`. Data paths resolve via `$STABLEWM_HOME`. Set WandB in the config.

### RF Evaluation
```bash
python eval_rf.py policy=lewm_rf              # evaluate checkpoint
```
Outputs one-step MSE, rollout error curve, and surprise scores.

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

**RF encoder** (`encoder.py`): `SpectrogramViT` — custom ViT for `(2, 256, 51)` spectrograms. Conv2d patch embedding with `(16, 3)` kernel → 272 patches + CLS token. Separate learned frequency/time positional embeddings. Uses `Block` from module.py.

**RF dataset** (`dataset.py`): `RFSpectralDataset` — reads HDF5 directly, slices 16-step trajectories into subsequences. Returns `(T, 2, 256, 51)` float32 tensors.

**Modules** (`module.py`): Shared building blocks:
- `ARPredictor` — supports both `ConditionalBlock` (robotics) and `Block` (RF) via `block_class` param
- `SIGReg` — Epps-Pulley Gaussian regularizer (domain-agnostic)
- `Block` / `ConditionalBlock` — standard vs AdaLN-zero transformer blocks
- `MLP` — projector heads with BatchNorm1d

**Training**: Both `train.py` (robotics) and `train_rf.py` (RF) use the same pattern: `pred_loss + lambd * sigreg_loss`, wrapped in `spt.Module` + `spt.Manager`.

**Evaluation**: `eval.py` (robotics) uses MPC planning. `eval_rf.py` measures one-step prediction MSE, multi-step rollout error, and surprise scores.

## Key Design Decisions

- Device-agnostic: uses `proj.device` / auto-detect for tensor creation, not hardcoded `cuda`
- Embeddings use `(B, T, D)` convention; projectors operate on flattened `(B*T, D)` via einops rearrange
- SIGReg operates on `(T, B, D)` transposed embeddings — note the transpose in forward functions
- Checkpoints are saved as serialized model objects (`torch.save(model, path)`) for `AutoCostModel` compatibility
- RF encoder uses factored positional embeddings: freq and time axes have separate learned embeddings broadcast-summed, reflecting their different physical semantics
- `ARPredictor.block_class` parameter controls conditional vs unconditional mode without code duplication
- RF dataset is pre-split by source scene (train/val/test .h5 files) — no random splitting needed
