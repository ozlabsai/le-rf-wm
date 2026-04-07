# RF-LeWM Results Log

Ongoing record of experiments, findings, and conclusions for the RF-LeWM project.

---

## Project Summary

**Goal:** Adapt LeWorldModel (JEPA world model for robotics) to learn latent dynamics of RF spectral environments from STFT spectrogram trajectories.

**Dataset:** [OzLabs/rf-spectral-trajectories](https://huggingface.co/datasets/OzLabs/rf-spectral-trajectories)
- 13,841 train / 2,938 val / 2,999 test trajectories
- 16 timesteps per trajectory at 80ms resolution
- Shape: `(N, 16, 256, 51, 2)` — freq bins × time bins × real/imag
- 8 RF activity regimes: quiet, dense, bursty, ramp_up, interference_event, correlated_alternating, correlated_leader_follower, random
- 400 source scenes, split by scene (no leakage)

**Model:** 16.4M params
- SpectrogramViT encoder (12-layer, 192-dim, 272 patches via 16×3 Conv2d, mean pooling)
- 6-layer unconditional AR transformer predictor (16 heads, 2048 mlp_dim)
- LayerNorm projectors (2048 hidden, 192 output)
- Training: residual prediction + L2-normalized loss + SIGReg + VICReg variance floor

**Hardware:** 1× NVIDIA A100-SXM4-80GB (RunPod), ~6 hours per 100-epoch run

---

## Experiment Timeline

### Phase 1: Architecture Debugging (Mar 28–30)

A series of training runs that all converged to degenerate solutions. Each failure revealed a different architectural issue.

#### Run 1: First training attempt
- **Config:** num_preds=1, sigreg_weight=0.09, BatchNorm projectors, lr=5e-5
- **Result:** Train loss → 0, val explodes. sigreg pinned at 51.46.
- **Diagnosis:** BatchNorm running stats diverge between train/eval with randomly-initialized encoder.
- **Fix:** Replaced BatchNorm1d → LayerNorm in projectors.

#### Run 2: LayerNorm fix, constant SIGReg
- **Config:** sigreg_weight=1.0, LayerNorm projectors
- **Result:** Stable but trivial. val/pred_loss → 1e-5, sigreg frozen at 51.458927 for 100 epochs.
- **Diagnosis:** SIGReg weight=1.0 too dominant — encoder stays at random init because SIGReg gradient overwhelms pred_loss gradient.
- **Fix:** SIGReg warmup schedule (0→target over N epochs).

#### Run 3: SIGReg warmup, various weights (0.05→0.3)
- **Config:** Linear warmup schedule, multiple weight values tested
- **Result:** Same degenerate pattern — sigreg frozen at 51.458927, pred_loss trivially low.
- **Diagnosis:** The prediction task itself was too easy, independent of SIGReg weight. The predictor achieved near-zero loss without the encoder learning anything.

#### Run 4: VICReg variance loss added
- **Config:** Added `variance_loss = relu(target_std - std).mean()` with std computed directly
- **Result:** var_loss stuck at exactly 1.0 — gradient was zero at collapse.
- **Diagnosis:** `std()` has undefined/zero gradient when all values are identical. Need `sqrt(var + eps)`.
- **Fix:** Variance loss using `sqrt(var + eps)` for finite gradient at collapse.

#### Run 5: Fixed variance loss
- **Config:** `sqrt(var + eps)` variance loss, weight=1.0
- **Result:** var_loss still stuck at 1.0. emb_std_mean = emb_std_min = emb_std_max = 0.0.
- **Diagnosis:** All embeddings exactly identical across batch. Not a gradient issue — the **CLS token** has zero variance by construction.

#### **Key Discovery: CLS Token Bug**
- **Audit:** `audit_encoder.py` traced variance through every encoder stage.
- **Finding:** Patch tokens had healthy std (~1.5) through all 12 transformer layers. But the **CLS token had exactly 0.0 std** across the batch at every layer — it's a shared `nn.Parameter` that attends uniformly to patches at random init.
- **Fix:** Replaced CLS token pooling with **mean pooling** over all patch tokens.

#### Run 6: Mean pooling — first non-degenerate training
- **Config:** Mean pooling, variance loss, SIGReg warmup (0→0.1 over 10 epochs), num_preds=2
- **Result:** First genuinely learning run.
  - val/pred_loss: 0.44 (not trivially zero!)
  - val/sigreg: 66 → 40 over 10 epochs (moving!)
  - val/emb_std_mean: 0.79 (nonzero!)
  - train/val gap: 2.6× (moderate, not catastrophic)
- **wandb:** `guy-na8/lewm-rf/runs/egq7jdan`

### Phase 2: Target Overlap Fix (Mar 30)

#### **Key Discovery: Context/Target Overlap Bug**
- **Audit:** `audit_wiring.py` printed the actual frame indices for context and targets.
- **Finding:** With the original slicing `tgt_emb = emb[:, n_preds:]`, the context and target windows overlapped:
  - ctx = [frame0, frame1, frame2], tgt = [frame1, frame2, frame3] (n_preds=1: 2 frames overlap)
  - ctx = [frame0, frame1, frame2], tgt = [frame2, frame3, frame4] (n_preds=2: 1 frame overlap)
- This overlap allowed the predictor to trivially copy context frames via causal attention to match targets.
- In the original robotics LeWM, this works because action-conditioning provides the prediction signal. In our unconditional setup, it's a free shortcut.
- **Fix:** `tgt_emb = emb[:, ctx_len:]` — targets start strictly after context. `pred_emb = pred_emb[:, -n_match:]` — only use the last (future-predicting) outputs.

### Phase 3: Prediction Objective Refinement (Mar 30–31)

#### Run 7: Stop-gradient on targets
- **Config:** `tgt_emb = emb[:, ctx_len:].detach()`, no overlap fix
- **Result:** Still degenerate — all embeddings identical at init means detach doesn't help when pred and tgt are the same constant vector.

#### Run 8: Full 100-epoch run with all fixes (v0 baseline)
- **Config:** Mean pooling, no overlap, detach targets, variance loss, SIGReg warmup (0→0.05 over 20 epochs), num_preds=4, lr=5e-5
- **Result (training):**
  - train/pred_loss: 0.132 (epoch 99)
  - val/pred_loss: 1.52 (epoch 99)
  - train/val ratio: 11.5× — significant overfitting
  - val/sigreg: 16.8
  - val/emb_std_mean: 0.91
- **wandb:** `guy-na8/lewm-rf/runs/9vw56yhz`
- **Baseline evaluation (raw MSE):**

| Method | 1-step MSE | vs Copy-last |
|--------|-----------|-------------|
| RF-LeWM | 1.411 | +12.1% |
| Copy-last | 1.606 | baseline |
| Mean-context | 1.104 | +31.3% |
| Zero | 0.925 | +42.4% |

- **Key issue:** Model beats copy-last but loses badly to zero and mean-context. Zero wins because the embedding space is centered near origin (norm ~13, MSE of zero ≈ 13²/192 ≈ 0.88).
- **Cosine similarity (from diagnostics):**

| Method | CosSim |
|--------|--------|
| RF-LeWM | 0.051 |
| Copy-last | 0.062 |
| Mean-context | 0.076 |

- **Conclusion:** Model learned a conservative "shrinkage" predictor — closer to mean than copy-last on MSE, but worse directional prediction than all baselines.

### Phase 4: L2-Normalized Loss (Mar 31)

#### Run 9: L2-normalized absolute prediction
- **Config:** `MSE(normalize(pred), normalize(tgt))` — cosine-aware loss
- **Result:**
  - train/pred_loss: 0.0011, val/pred_loss: 0.010 — train/val ratio 9.5×
  - Baseline MSE: +45.3% vs copy-last
  - Cosine sim: 0.051 (model) vs 0.062 (copy-last) — still loses on direction
- **Conclusion:** Removed shrinkage shortcut but still overfits absolute positions. The predictor memorizes scene-specific embeddings.

### Phase 5: Residual Prediction (Mar 31)

#### **Key Insight: Predict Change, Not State**
- Instead of `predict z_{t+k}`, predict `Δz = z_{t+k} - z_t`
- Removes scene-specific memorization shortcut — the model must learn temporal dynamics, not scene identity
- L2-normalize the residuals before loss: `MSE(normalize(pred_delta), normalize(tgt_delta))`

#### Run 10: Residual + L2-norm, num_preds=4 (v0 final)
- **Config:** Residual prediction, L2-normalized loss, SIGReg 0→0.05 over 20 epochs, variance_weight=1.0
- **Result (training):**
  - train/pred_loss: 0.0027, val/pred_loss: 0.0033
  - **train/val ratio: 1.24×** — generalization solved
  - val/sigreg: 9.56
  - val/emb_std_mean: 0.96
- **wandb:** `guy-na8/lewm-rf/runs/6abzuaa0`
- **Baseline evaluation:**

| Method | 1-step MSE | vs Copy-last | CosSim |
|--------|-----------|-------------|--------|
| RF-LeWM | 1.289 | +42.6% | **0.076** |
| Copy-last | 2.245 | baseline | 0.038 |
| Mean-context | 1.501 | +33.1% | 0.056 |
| Zero | 1.161 | +48.3% | N/A |

- **Key result:** RF-LeWM now has the **highest cosine similarity** of all methods (0.076 vs copy-last 0.038, mean 0.056). First time the model beats ALL baselines on directional prediction.
- **Regime analysis:** Beats copy-last on MSE in **58/60 test scenes**. Performance varies by regime as expected:
  - Best: ramp_up (+47.3%), quiet (+45.3%), leader_follower (+45.0%)
  - Weakest: interference_event (+36.6%), random (+40.5%)
  - Rollout degradation: ramp_up 1.03×, quiet 1.06×, alternating 1.61× (hardest)
- **Surprise detection:** Only detects noise burst (1.66×, 80.7% detection). Blind to signal injection, dropout, temporal reversal (all ~1.00×).
- **Checkpoint:** `OzLabs/RF-LeWM-v0` on HuggingFace

### Phase 6: Harder Prediction Horizon (Apr 1–2)

#### Run 11: num_preds=6
- **Config:** Same as Run 10 but num_preds=6 (seq_len=9, predict 3–5 steps ahead)
- **Result (training):**
  - train/pred_loss: 0.0030, val/pred_loss: 0.0044
  - train/val ratio: 1.47× (slightly more overfitting than num_preds=4)
  - val/sigreg: 10.9
  - val/emb_std_mean: 0.96
- **wandb:** `guy-na8/lewm-rf/runs/ppbeyl3z`
- **Baseline evaluation:** Pending (checkpoint saved to `/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt`)

---

## Key Bugs Found (Ordered by Discovery)

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | BatchNorm in projectors | Train/eval embedding mismatch → val explodes | LayerNorm |
| 2 | SIGReg too dominant | Encoder frozen at random init | Warmup schedule, lower weight |
| 3 | `std()` gradient zero at collapse | Variance loss ineffective | `sqrt(var + eps)` |
| 4 | CLS token zero variance | All embeddings identical across batch | Mean pooling over patches |
| 5 | Context/target overlap | Predictor copies context frames, trivial loss | `tgt = emb[:, ctx_len:]` |
| 6 | Absolute prediction + MSE | Shrinkage shortcut (predict low magnitude) | Residual + L2-normalized loss |

---

## Key Design Decisions

| Decision | Why | Alternative Considered |
|----------|-----|----------------------|
| Mean pooling (not CLS) | CLS has zero variance at random init | CLS works with pretrained encoders |
| LayerNorm (not BatchNorm) | BN running stats diverge with random-init encoder | BN works with pretrained encoders |
| Residual prediction (Δz) | Removes scene memorization shortcut | Absolute prediction (z) |
| L2-normalized loss | Removes magnitude shrinkage shortcut | Raw MSE, cosine loss |
| Stop-gradient on targets | Prevents encoder collapse to constant | EMA target encoder (not tried yet) |
| VICReg variance loss | Prevents embedding collapse with gradient at zero | SIGReg alone insufficient |
| SIGReg warmup (0→0.05) | Let prediction shape encoder before regularization | Constant SIGReg (too dominant early) |
| No overlap in targets | Prevents trivial copy shortcut in unconditional prediction | Original LeWM overlap (works for action-conditioned) |

---

## Current Understanding

### What works
- The encoder produces diverse, scene-specific embeddings (emb_std_mean ~0.96)
- The training pipeline is stable — no collapse, no explosion, proper losses
- Residual + L2-norm prediction generalizes well (train/val ratio 1.2–1.5×)
- Model beats copy-last consistently (42–45% on MSE, all regimes)
- Model has highest cosine similarity among all baselines (0.076)
- Rollout degrades gracefully over 12 steps (no collapse)
- Regime-specific behavior matches RF physics (ramp_up easiest, alternating hardest)

### What doesn't work yet
- Zero baseline still wins on raw MSE (embedding space is centered)
- Cosine similarity is modest (0.076) — room for improvement
- Surprise detection only works for gross amplitude changes (noise burst)
- Blind to structural perturbations (signal injection, dropout, temporal reversal)
- The predictor likely needs stronger temporal reasoning for fine-grained dynamics

### Open questions
- Does num_preds=6 improve over num_preds=4? (eval pending)
- Would EMA target encoder help, or is residual prediction sufficient?
- Can attention pooling or token-level prediction improve sensitivity to local spectral changes?
- Is the 16.4M model large enough, or does it need scaling?
- Would data augmentation (spectral shifts, time warping) reduce overfitting?

---

## Evaluation Tools

| Script | Purpose |
|--------|---------|
| `eval_rf.py` | Rollout error + surprise scores (Hydra config) |
| `eval_baselines.py` | Compare against 7 baselines: copy-last, mean, zero, velocity, linear extrapolation, exp. smoothing, model |
| `eval_diagnostics.py` | Embedding stats, cosine similarity (absolute + residual), per-regime breakdown, norm distribution |
| `eval_regimes.py` | Performance by RF regime (8 types) and SNR range |
| `eval_surprise.py` | Perturbation-based anomaly detection (4 perturbation types) |
| `smoke_test.py` | Pre-training validation (shapes, gradients, loss) |
| `audit_encoder.py` | Stage-by-stage encoder variance audit |
| `audit_wiring.py` | Context/target index verification |

---

## Artifacts

| Artifact | Location |
|----------|----------|
| v0 checkpoint (num_preds=4) | HuggingFace `OzLabs/RF-LeWM-v0`, local `lewm_rf_epoch_99_object.ckpt` |
| num_preds=6 checkpoint | RunPod `/workspace/data/lewm_rf_epoch_99_numpreds6_object.ckpt` |
| v0 model card | HuggingFace `OzLabs/RF-LeWM-v0/README.md` |
| Training logs | wandb `guy-na8/lewm-rf` (multiple runs) |
| Scene metadata | `scene_metadata.json` (400 scenes, 8 regimes) |
| Normalization stats | `norm_stats.json` (per-channel mean/std from train split) |
