"""Stage 4: Fine-tune decoder on predicted embeddings."""

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

sys.path.insert(0, str(Path(__file__).parent.parent))
from decoder.decoder import SpectrogramDecoder

CACHE = Path(__file__).parent / "cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt"
NORM_STATS = "/workspace/data/norm_stats.json"
HISTORY = 3
EPOCHS = 10
BATCH = 512
LR = 1e-4


class PredEmbLogmagDataset(Dataset):
    def __init__(self, emb_path, logmag_path, start_step=3):
        with h5py.File(emb_path, "r") as f:
            self.emb = torch.from_numpy(f["embeddings"][()]).float()  # [N, steps, 192]
        with h5py.File(logmag_path, "r") as f:
            self.logmag = torch.from_numpy(f["logmag"][:, start_step:start_step+self.emb.shape[1]]).float()
        self.N, self.T = self.emb.shape[:2]

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, idx):
        n, t = divmod(idx, self.T)
        return self.emb[n, t], self.logmag[n, t]


def generate_predicted_embeddings(split="train"):
    """Run predictor autoregressively and cache predicted embeddings."""
    dst = CACHE / f"predicted_embeddings_{split}.h5"
    if dst.exists():
        print(f"  {split}: already cached")
        return

    from dataset import load_norm_stats

    print(f"  Loading model...")
    model = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.requires_grad_(False)

    with h5py.File(CACHE / f"embeddings_{split}.h5", "r") as f:
        all_emb = torch.from_numpy(f["embeddings"][()]).float()  # [N, 16, 192]

    N = all_emb.shape[0]
    n_pred_steps = 16 - HISTORY  # 13 steps
    pred_embs = torch.zeros(N, n_pred_steps, 192)

    print(f"  Generating predicted embeddings for {N} trajectories...")
    with torch.no_grad():
        for i in range(0, N, 64):
            end = min(i + 64, N)
            batch_emb = all_emb[i:end]  # [B, 16, 192]
            ctx = batch_emb[:, :HISTORY]  # [B, 3, 192]
            rolled = model.rollout_unconditional(ctx.clone(), n_steps=n_pred_steps, history_size=HISTORY)
            pred_embs[i:end] = rolled[:, HISTORY:]  # [B, 13, 192]
            if i % 640 == 0:
                print(f"    {end}/{N}")

    with h5py.File(dst, "w") as f:
        f.create_dataset("embeddings", data=pred_embs.numpy(), dtype="float32")
    print(f"  Saved {pred_embs.shape} to {dst}")


def compute_ssim_loader(model, loader, data_range):
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=data_range).to(DEVICE)
    with torch.no_grad():
        for emb, target in loader:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            pred = model(emb)
            ssim_fn.update(pred.unsqueeze(1), target.unsqueeze(1))
    return ssim_fn.compute().item()


if __name__ == "__main__":
    print("=== Stage 4: Fine-tuning on predicted embeddings ===")

    # Generate predicted embeddings
    for split in ["train", "val"]:
        generate_predicted_embeddings(split)

    # Log-mag range
    with h5py.File(CACHE / "logmag_train.h5", "r") as f:
        subset = f["logmag"][:500].astype(np.float32)
        lm_min, lm_max = float(subset.min()), float(subset.max())
    data_range = lm_max - lm_min

    # Datasets
    train_ds = PredEmbLogmagDataset(CACHE / "predicted_embeddings_train.h5", CACHE / "logmag_train.h5", start_step=HISTORY)
    val_ds = PredEmbLogmagDataset(CACHE / "predicted_embeddings_val.h5", CACHE / "logmag_val.h5", start_step=HISTORY)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    # Load base decoder
    base_path = Path(__file__).parent / "decoder_best.ckpt"
    model = SpectrogramDecoder(embed_dim=192).to(DEVICE)
    model.load_state_dict(torch.load(base_path, weights_only=True))

    # Measure base decoder on predicted embeddings
    model.train(False)
    base_ssim = compute_ssim_loader(model, val_loader, data_range)
    print(f"Base decoder SSIM on predicted embeddings: {base_ssim:.4f}")

    # Fine-tune
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ft_path = Path(__file__).parent / "decoder_finetuned.ckpt"
    best_ssim = base_ssim

    for epoch in range(EPOCHS):
        model.train(True)
        total_loss = 0
        for emb, target in train_loader:
            emb, target = emb.to(DEVICE), target.to(DEVICE)
            pred = model(emb)
            loss = (pred - target).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.train(False)
        ssim = compute_ssim_loader(model, val_loader, data_range)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.5f}  val_SSIM={ssim:.4f}")
        if ssim > best_ssim:
            best_ssim = ssim
            torch.save(model.state_dict(), ft_path)

    # Final summary
    model.load_state_dict(torch.load(ft_path, weights_only=True))
    model.train(False)
    ft_ssim = compute_ssim_loader(model, val_loader, data_range)

    delta = ft_ssim - base_ssim
    print(f"\n{'='*50}")
    print(f"=== FINE-TUNE SUMMARY ===")
    print(f"SSIM on predicted emb (base decoder):       {base_ssim:.4f}")
    print(f"SSIM on predicted emb (fine-tuned decoder):  {ft_ssim:.4f}")
    print(f"Delta:                                       {delta:+.4f}")
    print(f"{'='*50}")
