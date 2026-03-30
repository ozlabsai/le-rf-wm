"""Audit prediction wiring: verify offset, check for trivial shortcuts.

Run: python audit_wiring.py --data_path /path/to/train.h5

This script checks:
1. Are context and target indices actually offset?
2. How similar are embeddings at different time offsets?
3. Can the predictor trivially solve the task from random init?
4. Does the encoder differentiate between timesteps at all?
"""

import argparse
import torch
from dataset import RFSpectralDataset, compute_norm_stats
from encoder import SpectrogramViT
from jepa import JEPA
from module import ARPredictor, Block, MLP


def run(data_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data with num_preds=2 (matching current config)
    stats = compute_norm_stats(data_path)
    ds = RFSpectralDataset(data_path, history_size=3, num_preds=2, norm_stats=stats)

    print(f"Dataset: {len(ds)} samples, seq_len={ds.seq_len}")
    sample = ds[0]
    obs = sample["observations"]
    print(f"Sample shape: {obs.shape}")  # Should be (5, 2, 256, 51)

    # Build model (random init)
    hidden_dim = 192
    encoder = SpectrogramViT(hidden_dim=hidden_dim, depth=4, heads=3, mlp_dim=768).to(device)
    predictor = ARPredictor(
        num_frames=3, input_dim=hidden_dim, hidden_dim=hidden_dim,
        output_dim=hidden_dim, depth=2, heads=4, mlp_dim=512,
        dim_head=64, block_class=Block,
    ).to(device)
    projector = MLP(hidden_dim, 2048, hidden_dim, norm_fn=torch.nn.LayerNorm).to(device)
    pred_proj = MLP(hidden_dim, 2048, hidden_dim, norm_fn=torch.nn.LayerNorm).to(device)
    model = JEPA(encoder=encoder, predictor=predictor, projector=projector, pred_proj=pred_proj).to(device)

    # Get a batch
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    obs_batch = batch["observations"].to(device)  # (B, 5, 2, 256, 51)
    B, T = obs_batch.shape[:2]

    print(f"\n=== 1. Index Audit ===")
    print(f"history_size=3, num_preds=2, seq_len={T}")
    print(f"ctx_emb = emb[:, :3]  → timesteps [0, 1, 2]")
    print(f"tgt_emb = emb[:, 2:]  → timesteps [2, 3, 4]")
    print(f"Predictor input: 3 frames → output: 3 predictions")
    print(f"Prediction mapping: pred[0]→tgt[0]=frame2, pred[1]→tgt[1]=frame3, pred[2]→tgt[2]=frame4")
    print()
    print(f"WARNING: pred[0] predicts frame 2, and ctx includes frame 2!")
    print(f"  ctx = [frame0, frame1, frame2]")
    print(f"  tgt = [frame2, frame3, frame4]")
    print(f"  The first prediction target (frame2) is IN the context!")
    print(f"  This means the predictor can trivially copy ctx[-1] for position 0.")

    print(f"\n=== 2. Embedding Similarity at Init ===")
    with torch.no_grad():
        info = {"observations": obs_batch}
        info = model.encode_rf(info)
        emb = info["emb"]  # (B, 5, D)

    for i in range(T):
        for j in range(i+1, T):
            dist = (emb[:, i] - emb[:, j]).pow(2).mean().item()
            cos = torch.nn.functional.cosine_similarity(emb[:, i], emb[:, j], dim=-1).mean().item()
            print(f"  frame {i} vs frame {j}: MSE={dist:.6f}, cos_sim={cos:.4f}")

    print(f"\n=== 3. Prediction vs Target at Init ===")
    with torch.no_grad():
        ctx_emb = emb[:, :3]
        tgt_emb = emb[:, 2:]  # n_preds=2
        pred_emb = model.predict(ctx_emb)

    for t in range(3):
        err = (pred_emb[:, t] - tgt_emb[:, t]).pow(2).mean().item()
        # Also check: how close is pred to just copying last context frame?
        copy_err = (ctx_emb[:, -1] - tgt_emb[:, t]).pow(2).mean().item()
        print(f"  position {t}: pred_error={err:.6f}, copy_last_ctx_error={copy_err:.6f}")

    print(f"\n=== 4. The Overlap Problem ===")
    print(f"With history_size=3, num_preds=2:")
    print(f"  ctx  = emb[:, 0:3] = [t0, t1, t2]")
    print(f"  tgt  = emb[:, 2:5] = [t2, t3, t4]")
    print(f"  OVERLAP: t2 appears in BOTH ctx and tgt!")
    print(f"")
    print(f"With history_size=3, num_preds=1 (previous config):")
    print(f"  ctx  = emb[:, 0:3] = [t0, t1, t2]")
    print(f"  tgt  = emb[:, 1:4] = [t1, t2, t3]")
    print(f"  OVERLAP: t1 and t2 appear in BOTH ctx and tgt!")
    print(f"")
    print(f"This means the predictor can achieve near-zero loss on 2 out of 3")
    print(f"target positions just by copying context frames. The causal attention")
    print(f"mask in the transformer makes this trivial.")
    print(f"")
    print(f"FIX: tgt_emb should be emb[:, ctx_len:ctx_len+num_preds] (NO overlap)")
    print(f"  or equivalently: tgt_emb = emb[:, ctx_len:]")
    print(f"  ctx  = emb[:, 0:3] = [t0, t1, t2]")
    print(f"  tgt  = emb[:, 3:5] = [t3, t4]")
    print(f"  pred = predict(ctx)[:, -num_preds:] = last 2 outputs only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    run(args.data_path)
