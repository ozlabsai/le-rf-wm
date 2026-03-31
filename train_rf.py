"""Training script for RF-LeWM (unconditional JEPA on RF spectrograms)."""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from einops import rearrange
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


def sigreg_weight_schedule(epoch, cfg_loss):
    """Compute current SIGReg weight based on epoch and schedule config."""
    schedule = cfg_loss.sigreg.get("schedule", "constant")

    if schedule == "constant":
        return cfg_loss.sigreg.weight

    if schedule == "linear":
        start = cfg_loss.sigreg.weight_start
        end = cfg_loss.sigreg.weight_end
        warmup = cfg_loss.sigreg.warmup_epochs
        if warmup <= 0 or epoch >= warmup:
            return end
        return start + (end - start) * (epoch / warmup)

    raise ValueError(f"Unknown sigreg schedule: {schedule}")


def variance_loss(emb, target_std=1.0, eps=1e-4):
    """VICReg-style variance regularizer.
    Penalizes embedding dimensions whose std across samples falls below target_std.
    Uses sqrt(var + eps) to ensure gradient is finite at collapse.
    emb: (N, D) — flattened embeddings
    """
    var = emb.var(dim=0)          # (D,) variance per dimension
    std = (var + eps).sqrt()      # finite gradient even when var=0
    return torch.relu(target_std - std).mean()


def rf_forward(self, batch, stage, cfg):
    """Encode RF spectrograms, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    epoch = self.current_epoch
    lambd = sigreg_weight_schedule(epoch, cfg.loss)
    var_weight = cfg.loss.get("variance_weight", 1.0)

    output = self.model.encode_rf(batch)

    emb = output["emb"]  # (B, T, D) where T = ctx_len + n_preds

    ctx_emb = emb[:, :ctx_len]                     # (B, ctx_len, D)
    tgt_emb = emb[:, ctx_len:].detach()             # (B, n_preds, D) — stop gradient on targets
    pred_emb = self.model.predict(ctx_emb)          # (B, ctx_len, D)

    # Match prediction and target lengths (predictor outputs ctx_len frames)
    n_match = min(n_preds, ctx_len)
    pred_emb = pred_emb[:, -n_match:]               # last n_match predictions
    tgt_emb = tgt_emb[:, :n_match]                   # first n_match targets

    # Residual prediction: predict change from last context frame
    anchor = ctx_emb[:, -1:].detach()                # (B, 1, D)
    tgt_delta = tgt_emb - anchor                     # (B, n_match, D)
    pred_delta = pred_emb - anchor                   # (B, n_match, D)

    # Flatten embeddings across batch and time for variance computation
    emb_flat = rearrange(emb, "b t d -> (b t) d")

    # L2-normalize residuals before loss — direction of change, not magnitude
    pred_norm = F.normalize(pred_delta, dim=-1)
    tgt_norm = F.normalize(tgt_delta, dim=-1)

    # LeWM loss + variance regularizer
    output["pred_loss"] = (pred_norm - tgt_norm).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["var_loss"] = variance_loss(emb_flat)
    output["loss"] = (output["pred_loss"]
                      + lambd * output["sigreg_loss"]
                      + var_weight * output["var_loss"])

    # Embedding std diagnostics
    with torch.no_grad():
        dim_std = emb_flat.std(dim=0)
        output["emb_std_mean"] = dim_std.mean()
        output["emb_std_min"] = dim_std.min()
        output["emb_std_max"] = dim_std.max()

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    losses_dict[f"{stage}/sigreg_weight"] = lambd
    losses_dict[f"{stage}/emb_std_mean"] = output["emb_std_mean"]
    losses_dict[f"{stage}/emb_std_min"] = output["emb_std_min"]
    losses_dict[f"{stage}/emb_std_max"] = output["emb_std_max"]
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
        norm_fn=torch.nn.LayerNorm,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.LayerNorm,
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
