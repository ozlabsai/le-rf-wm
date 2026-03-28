"""Training script for RF-LeWM (unconditional JEPA on RF spectrograms)."""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from dataset import RFSpectralDataset, compute_norm_stats, save_norm_stats
from encoder import SpectrogramViT
from jepa import JEPA
from module import ARPredictor, Block, MLP, SIGReg
from utils import ModelObjectCallBack

# Workaround: stable_pretraining 0.1.4 on_train_start calls len() on
# self.optimizers(), which Lightning unwraps to a bare object when there's
# only one optimizer. Patch it to handle both cases.
_orig_on_train_start = spt.Module.on_train_start

def _patched_on_train_start(self):
    optimizers = self.optimizers()
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    # rebind so the rest of the method sees a list
    self.optimizers = lambda: optimizers
    _orig_on_train_start(self)

spt.Module.on_train_start = _patched_on_train_start


def rf_forward(self, batch, stage, cfg):
    """Encode RF spectrograms, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    output = self.model.encode_rf(batch)

    emb = output["emb"]  # (B, T, D)

    ctx_emb = emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb)  # unconditional

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm_rf")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # compute or load normalization stats from train split
    norm_stats = None
    if cfg.data.get("normalize", True):
        stats_path = Path(cfg.data.train_path).parent / "norm_stats.json"
        if stats_path.exists():
            print(f"Loading norm stats from {stats_path}")
            from dataset import load_norm_stats
            norm_stats = load_norm_stats(stats_path)
        else:
            print(f"Computing norm stats from {cfg.data.train_path} ...")
            norm_stats = compute_norm_stats(cfg.data.train_path)
            save_norm_stats(norm_stats, stats_path)
            print(f"Saved norm stats to {stats_path}")
        print(f"  mean: {norm_stats['mean']}")
        print(f"  std:  {norm_stats['std']}")

    train_set = RFSpectralDataset(
        cfg.data.train_path,
        history_size=cfg.wm.history_size,
        num_preds=cfg.wm.num_preds,
        norm_stats=norm_stats,
    )
    val_set = RFSpectralDataset(
        cfg.data.val_path,
        history_size=cfg.wm.history_size,
        num_preds=cfg.wm.num_preds,
        norm_stats=norm_stats,
    )

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen,
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = SpectrogramViT(**cfg.encoder)

    hidden_dim = cfg.encoder.hidden_dim
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        block_class=Block,  # unconditional
        **cfg.predictor,
    )

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(rf_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()


if __name__ == "__main__":
    run()
