"""FastAPI backend for RF-LeWM demo.

Serves trajectories, predictions, surprise scores, and perturbation experiments.
Runs on CPU — no GPU required for inference.

Usage:
    cd demo/backend
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.decomposition import PCA

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import load_norm_stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "demo" / "data"
TEST_H5_PATH = str(DATA_DIR / "test.h5")
NORM_STATS_PATH = str(DATA_DIR / "norm_stats.json")
CHECKPOINT_PATH = str(DATA_DIR / "lewm_rf_epoch_99_object.ckpt")
METADATA_PATH = str(PROJECT_ROOT / "scene_metadata.json")
NUM_PCA_SAMPLES = 300
HISTORY_SIZE = 3

app = FastAPI(title="RF-LeWM Demo API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
norm_mean = None
norm_std = None
test_data = None
source_ids = None
scene_meta = None
pca_model = None
pca_background = None


@app.on_event("startup")
def load_everything():
    global model, norm_mean, norm_std, test_data, source_ids, scene_meta
    global pca_model, pca_background

    print("Loading model...")
    model = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.requires_grad_(False)
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    print("Loading normalization stats...")
    stats = load_norm_stats(NORM_STATS_PATH)
    norm_mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 1, 2, 1, 1)
    norm_std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 1, 2, 1, 1)

    print("Loading test data...")
    with h5py.File(TEST_H5_PATH, "r") as f:
        raw = torch.from_numpy(f["observations"][()]).float()
        test_data = raw.permute(0, 1, 4, 2, 3)  # (N, 16, 2, 256, 51)
        test_data = (test_data - norm_mean) / norm_std
        ids_raw = f["source_ids"][()]
        source_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in ids_raw]
    print(f"  Loaded {test_data.shape[0]} trajectories")

    print("Loading scene metadata...")
    with open(METADATA_PATH) as f:
        scene_meta = json.load(f)

    print("Computing PCA projection...")
    n_pca = min(NUM_PCA_SAMPLES, test_data.shape[0])
    rng = np.random.RandomState(42)
    indices = rng.choice(test_data.shape[0], n_pca, replace=False)
    pca_embs = []
    with torch.no_grad():
        for i in indices:
            obs = test_data[i:i+1]
            obs_flat = rearrange(obs, "b t ... -> (b t) ...")
            emb = model.encoder(obs_flat)
            emb = model.projector(emb)
            pca_embs.append(emb.numpy())
    pca_embs = np.concatenate(pca_embs, axis=0)

    pca_model = PCA(n_components=2)
    pca_model.fit(pca_embs)

    pca_background = []
    projected = pca_model.transform(pca_embs)
    for i, idx in enumerate(indices):
        sid = source_ids[idx]
        meta = scene_meta.get(sid, {})
        points = projected[i * 16:(i + 1) * 16]
        pca_background.append({
            "scene_id": sid,
            "regime": meta.get("regime", "unknown"),
            "points": points.tolist(),
        })
    print(f"  PCA fitted on {pca_embs.shape[0]} embeddings")
    print("Startup complete!")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode_trajectory(obs):
    """Encode a trajectory, return embeddings + patch norms.
    obs: (1, T, 2, 256, 51)
    Returns: emb (T, D), patch_norms (T, n_freq, n_time)
    """
    obs_flat = rearrange(obs, "b t ... -> (b t) ...")
    with torch.no_grad():
        pooled, patch_norms = model.encoder(obs_flat, return_patch_norms=True)
        emb = model.projector(pooled)
    return emb, patch_norms


def compute_surprise(emb, max_steps=13):
    """Per-step surprise via autoregressive rollout."""
    T = emb.size(0)
    emb_b = emb.unsqueeze(0)
    ctx = emb_b[:, :HISTORY_SIZE]
    n_steps = min(max_steps, T - HISTORY_SIZE)

    with torch.no_grad():
        rolled = model.rollout_unconditional(ctx.clone(), n_steps=n_steps,
                                              history_size=HISTORY_SIZE)
    scores = []
    for t in range(n_steps):
        pred = rolled[0, HISTORY_SIZE + t]
        target = emb[HISTORY_SIZE + t]
        scores.append((pred - target).pow(2).mean().item())
    return scores


def apply_perturbation(obs, ptype, inject_step=8, strength=3.0):
    """Apply perturbation to trajectory observations."""
    p = obs.clone()
    F_bins = p.shape[3]

    if ptype == "noise_burst":
        end = min(inject_step + 3, p.shape[1])
        p[:, inject_step:end] += strength * torch.randn_like(p[:, inject_step:end])
    elif ptype == "signal_injection":
        band = slice(F_bins // 4, F_bins // 4 + 10)
        p[:, inject_step:, :, band, :] += strength
    elif ptype == "signal_dropout":
        band = slice(F_bins // 3, F_bins // 3 + 30)
        p[:, inject_step:, :, band, :] = 0
    elif ptype == "temporal_reversal":
        p[:, inject_step:] = p[:, inject_step:].flip(dims=[1])
    elif ptype == "frequency_shift":
        shift = 20
        p[:, inject_step:, :, shift:, :] = p[:, inject_step:, :, :-shift, :].clone()
        p[:, inject_step:, :, :shift, :] = 0
    return p


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    trajectory_id: int

class InjectRequest(BaseModel):
    trajectory_id: int
    perturbation_type: str = "noise_burst"
    inject_step: int = 8
    strength: float = 3.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/trajectory/{traj_id}")
def get_trajectory(traj_id: int):
    if traj_id < 0 or traj_id >= test_data.shape[0]:
        raise HTTPException(404, f"Trajectory {traj_id} not found")

    sid = source_ids[traj_id]
    meta = scene_meta.get(sid, {})
    obs = test_data[traj_id]  # (16, 2, 256, 51)
    # Compute magnitude from real/imag for visualization
    mag = (obs[:, 0].pow(2) + obs[:, 1].pow(2)).sqrt()  # (16, 256, 51)

    return {
        "trajectory_id": traj_id,
        "scene_id": sid,
        "regime": meta.get("regime", "unknown"),
        "snr_db": meta.get("snr_db", 0),
        "num_signals": meta.get("num_signals", 0),
        "num_timesteps": 16,
        "magnitude": mag.numpy().tolist(),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trajectory_id
    if tid < 0 or tid >= test_data.shape[0]:
        raise HTTPException(404, f"Trajectory {tid} not found")

    obs = test_data[tid:tid+1]
    emb, patch_norms = encode_trajectory(obs)
    surprise = compute_surprise(emb)

    # PCA projection
    emb_2d = pca_model.transform(emb.numpy()).tolist()

    # Predicted trajectory in PCA space
    emb_b = emb.unsqueeze(0)
    ctx = emb_b[:, :HISTORY_SIZE]
    with torch.no_grad():
        rolled = model.rollout_unconditional(ctx.clone(), n_steps=13,
                                              history_size=HISTORY_SIZE)
    pred_emb = rolled[0].numpy()
    pred_2d = pca_model.transform(pred_emb).tolist()

    # Delta cosine per step
    delta_cosines = []
    for t in range(HISTORY_SIZE, emb.shape[0]):
        anchor = emb[t-1:t]
        pred_t = torch.from_numpy(pred_emb[t:t+1])
        tgt_t = emb[t:t+1]
        if (tgt_t - anchor).norm() > 1e-6:
            cos = F.cosine_similarity(pred_t - anchor, tgt_t - anchor, dim=-1).item()
        else:
            cos = 0.0
        delta_cosines.append(cos)

    sid = source_ids[tid]
    meta = scene_meta.get(sid, {})

    return {
        "trajectory_id": tid,
        "scene_id": sid,
        "regime": meta.get("regime", "unknown"),
        "snr_db": meta.get("snr_db", 0),
        "surprise_scores": surprise,
        "delta_cosines": delta_cosines,
        "pca_actual": emb_2d,
        "pca_predicted": pred_2d,
        "patch_norms": patch_norms.numpy().tolist(),
    }


@app.post("/inject")
def inject(req: InjectRequest):
    tid = req.trajectory_id
    if tid < 0 or tid >= test_data.shape[0]:
        raise HTTPException(404, f"Trajectory {tid} not found")

    valid = ["noise_burst", "signal_injection", "signal_dropout",
             "temporal_reversal", "frequency_shift"]
    if req.perturbation_type not in valid:
        raise HTTPException(400, f"Invalid type. Valid: {valid}")

    obs = test_data[tid:tid+1]

    emb_normal, _ = encode_trajectory(obs)
    surprise_normal = compute_surprise(emb_normal)

    obs_perturbed = apply_perturbation(obs, req.perturbation_type,
                                        req.inject_step, req.strength)
    emb_perturbed, _ = encode_trajectory(obs_perturbed)
    surprise_perturbed = compute_surprise(emb_perturbed)

    mean_n = np.mean(surprise_normal) if surprise_normal else 1.0
    mean_p = np.mean(surprise_perturbed) if surprise_perturbed else 0.0
    ratio = mean_p / mean_n if mean_n > 0 else 0.0

    return {
        "trajectory_id": tid,
        "perturbation_type": req.perturbation_type,
        "inject_step": req.inject_step,
        "surprise_normal": surprise_normal,
        "surprise_perturbed": surprise_perturbed,
        "surprise_ratio": ratio,
        "detected": ratio > 1.2,
    }


@app.get("/pca_background")
def get_pca_background():
    return {"trajectories": pca_background}


@app.get("/regimes")
def get_regimes():
    """List all regimes with trajectory counts."""
    regime_counts = {}
    for sid in source_ids:
        r = scene_meta.get(sid, {}).get("regime", "unknown")
        regime_counts[r] = regime_counts.get(r, 0) + 1
    return {"regimes": regime_counts, "total": len(source_ids)}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None,
            "num_trajectories": test_data.shape[0] if test_data is not None else 0}
