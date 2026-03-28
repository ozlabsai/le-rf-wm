# le-rf-wm — RF-LeWM

Fork of [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm): a **Joint-Embedding Predictive Architecture** extended with an **RF spectral pipeline**—`SpectrogramViT` encoder, HDF5 trajectory data, and unconditional latent prediction for RF scene dynamics. The original **robotics** path (pixels, actions, MPC) remains in `train.py` / `eval.py`.

Upstream paper: [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/pdf/2603.19312v1) (Maes et al., 2026).

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Install

Uses [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for training utilities and (for robotics) environments/planning.

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install -e .
# Robotics + envs (optional):
uv pip install "stable-worldmodel[train,env]"
```

Core deps are listed in `pyproject.toml` (`torch`, `lightning`, `hydra-core`, `h5py`, `einops`, etc.).

## RF data

Spectrogram trajectories are **HDF5** files (train / val / test). Schema, shapes, and splits are documented in [DATASET.md](DATASET.md).

Point the configs at your files:

- `config/train/data/rf.yaml` — `train_path`, `val_path`
- `config/eval/rf.yaml` — `data.test_path`

Robotics datasets from the upstream project still go under `$STABLEWM_HOME` (default `~/.stable-wm/`); see upstream README / Hugging Face collections for `.h5` layout.

## RF training

```bash
python train_rf.py
python train_rf.py trainer.max_epochs=50
```

Hydra defaults: `config/train/lewm_rf.yaml`. Set WandB `entity` / `project` under the `wandb:` block there (or override on the CLI).

Serialized model dumps (`*_object.ckpt`) and Lightning weight checkpoints are written under `stable_worldmodel`’s cache root (usually `$STABLEWM_HOME`) with a per-run subdirectory from Hydra (`lewm_rf.yaml` `subdir`, default `${hydra:job.id}`).

## RF evaluation

```bash
python eval_rf.py policy=lewm_rf
```

`policy` is the checkpoint name **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix. Eval reports one-step prediction error, rollout error, and surprise-style metrics (see `eval_rf.py`).

## Robotics (upstream)

```bash
python train.py data=pusht
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

Configs: `config/train/lewm.yaml`, `config/eval/`.

## Code map

| Area | Files |
|------|--------|
| Shared JEPA | `jepa.py` — `encode`, `encode_rf`, `predict`, `rollout`, `rollout_unconditional` |
| RF encoder | `encoder.py` — `SpectrogramViT` |
| RF data | `dataset.py` — `RFSpectralDataset` |
| Blocks / predictor / SIGReg | `module.py` |
| RF train / eval entrypoints | `train_rf.py`, `eval_rf.py` |

Design notes for contributors: see [CLAUDE.md](CLAUDE.md).

## License

MIT — see [LICENSE](LICENSE) (upstream copyright).
