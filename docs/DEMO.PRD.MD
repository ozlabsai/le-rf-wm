# RF-LeWM Demo — Product Requirements Document

**Project:** RF Behavioral Engine Demo  
**Owner:** Oz Labs  
**Status:** Draft v1.0  
**Date:** April 2026

---

## 1. Purpose & Goals

### What this demo is

A web application demonstrating that RF-LeWM — Oz Labs' JEPA-based RF world model — understands the temporal dynamics of radio frequency environments. It is a research demonstration and capability proof, not a production system UI.

### Three things a client must walk away believing

1. **This model understands RF environments** — it has learned the structure of how signals appear, move, interact, and disappear, not just pattern-matched on training data.
2. **This is a practical anomaly detection capability** — the model detects physically implausible RF events with measurable reliability.
3. **Oz Labs is doing serious, novel RF AI research** — the depth of the evaluation methodology, the honest presentation of both successes and current limits, and the clarity of the roadmap signal a mature research operation.

### What this demo is not

- A production deployment interface
- A real-time SDR system (data is simulated)
- A polished consumer product

---

## 2. Audience

**Primary:** Defense and intelligence clients and prospects. Technical sophistication varies — assume someone with RF domain knowledge but not necessarily deep ML background. The demo must be legible to a signals analyst as well as a program manager.

**Secondary:** Oz Labs internal team for alignment and iteration.

**Tone:** Precision over spectacle. The demo should feel like a serious research instrument, not a marketing artifact. Honest presentation of current limitations builds more credibility with this audience than overselling.

---

## 3. System Architecture

### Frontend
Single-page React application (Vite + React). Self-contained, deployable to any cloud host. No classification constraints for the initial version. Canvas for spectrograms, Recharts for timelines, D3 for latent space scatter.

### Backend
FastAPI inference server hosting the RF-LeWM checkpoint. Exposes three endpoints:

| Endpoint | Input | Output |
|----------|-------|--------|
| `POST /predict` | Trajectory index or uploaded `.h5` slice | Predicted Δz sequence, per-patch norm heatmaps, per-step surprise scores |
| `POST /inject` | Trajectory index + perturbation config | Modified trajectory + surprise scores pre/post perturbation |
| `GET /trajectory/{id}` | Trajectory ID | Raw observations, ground truth embeddings, regime label, SNR |

### Data
- Pre-indexed test set trajectories from `rf-spectral-trajectories` (2,999 trajectories, 8 regimes)
- Synthetic anomaly injection at request time: noise burst, signal injection, frequency shift, signal dropout, temporal reversal
- No real captured RF data in v1

### Deployment
Docker container (FastAPI + model) + static frontend build. Cloud hosted (initial). Architecture must support future self-hosted / air-gapped deployment without frontend changes — all inference calls go through the backend, no direct model access from client.

---

## 4. Interface — Screens & Panels

### 4.1 Landing / Scene Selection

**Purpose:** Orient the user, let them pick what to explore.

**Elements:**
- Brief one-paragraph framing: what RF-LeWM is and what the demo shows
- Scene browser: filterable grid of test trajectories by regime (8 types), SNR range, and scene ID
- Each card shows: regime label, SNR, a thumbnail spectrogram of the first frame
- "Load Scene" button → navigates to main demo view
- Curator's picks: 3–5 pre-selected trajectories that show the most interesting model behavior (one per key finding: good dynamics prediction, noise burst detection, structural perturbation failure)

**Design note:** The curator's picks are important. Don't make the user find interesting cases themselves. Pre-select the best evidence.

---

### 4.2 Main Demo View — Three-Panel Layout

The core of the application. Three panels, always visible, synchronized to the same loaded trajectory.

---

#### Panel A: RF Scene Playback

**Purpose:** Show what the model sees and what it predicts.

**Layout:** Two side-by-side animated spectrograms.

**Left — Ground Truth:**
- Animated heatmap of the actual STFT spectrogram evolving over 16 timesteps
- Frequency on Y-axis (256 bins, ±75 kHz), time on X-axis (51 bins per frame)
- Power encoded as color (diverging scale: dark = low power, bright = high power)
- Playback controls: play/pause, step forward/back, speed control
- Current timestep indicator
- Regime label and SNR displayed as metadata

**Right — Encoder Attention Map:**
- Per-patch embedding norm heatmap overlaid on the spectrogram (16×17 grid mapped to 256×51)
- Shows which frequency regions and time bins the encoder weighted most heavily
- Visually distinct border/label to clearly distinguish from ground truth
- More compelling than reconstruction for defense audience: answers "what is the model paying attention to"
- Optional toggle: show prediction rollout overlay (predicted vs actual trajectory in latent space)
- Note: full spectrogram reconstruction (decoder) deferred to v2

**Interaction:**
- Scrub through timesteps — both panels stay synchronized
- Hover over a frequency bin to see its power value
- Click a timestep to jump to it in Panel C

---

#### Panel B: Latent Space Dynamics

**Purpose:** Show that the latent space encodes meaningful temporal structure — the "model understands RF" proof.

**Layout:** 2D projection (PCA over the test set, computed offline) of the latent trajectory.

**What it shows:**
- The trajectory of latent embeddings z₀ → z₁ → ... → z₁₅ as a path through 2D space (connected dots)
- Alongside it: the model's predicted path (predicted Δz accumulated from context)
- Color encodes timestep (gradient from light to dark)
- Predicted points shown as open circles, actual as filled circles

**Background context:**
- Faint scatter of 200–300 other test trajectories in the same PCA space (grey, non-interactive)
- Colored by regime — shows clustering structure
- This makes the claim "the latent space is structured" visually immediate

**Interaction:**
- Hover a point to see its timestep, embedding norm, and delta cosine to ground truth
- Toggle: show all regimes in background / show only same regime / show none
- Animate the trajectory step by step synchronized with Panel A playback

**Key metric displayed:** Delta cosine similarity (current step) — the single number that measures whether the model is predicting in the right direction.

---

#### Panel C: Surprise Score Timeline

**Purpose:** Demonstrate anomaly detection capability.

**Layout:** Line chart, timestep on X-axis, surprise score on Y-axis.

**Normal mode:**
- Surprise score (prediction error in latent space) plotted over all 16 timesteps
- Baseline band shown (mean ± 1 std of surprise scores across the test set for this regime)
- Scores above the band highlighted in red

**Anomaly injection mode:**
- User selects a perturbation type and injection timestep from a dropdown
- Perturbation types:
  - Noise burst (known to work — 1.66× surprise, 80.7% detection)
  - Signal injection (known limitation — ~1.0× surprise, structural blind spot)
  - Frequency shift (unknown — to be tested)
  - Signal dropout (known limitation)
  - Temporal reversal (known limitation)
- Backend re-runs inference on the modified trajectory
- Before/after surprise curves overlaid on the same chart
- Detection result: DETECTED / NOT DETECTED with the surprise ratio

**Honest framing:** Each perturbation type shows its known detection rate from the eval results. The limitations are labeled, not hidden. Signal injection shows "Current model blind spot — under active development." This is not a weakness — it's credibility.

**Interaction:**
- Inject button triggers backend call, result animates in
- Toggle between single-run and average-over-regime views
- Export button: download the surprise scores as CSV

---

### 4.3 Evaluation Dashboard (Secondary View)

**Purpose:** For the technically sophisticated viewer. Shows aggregate performance across the full test set.

**Sections:**

**Model vs. Baselines table:**
- Reproduces the baseline comparison from RESULTS.md
- Interactive: hover each cell for interpretation
- Footnote explaining why Zero wins on raw MSE (embedding centering) — this must be explained, not hidden

**Regime breakdown:**
- Heatmap: regime × metric (MSE improvement, delta cosine, rollout degradation)
- Shows which RF dynamics the model handles well vs. poorly
- Populated with regime-stratified eval results once available

**Surprise detection summary:**
- Bar chart: detection rate by perturbation type
- Color coded: green (working), yellow (partial), red (blind spot)

**Architecture card:**
- 16.4M params, SpectrogramViT encoder, AR transformer predictor
- Key design decisions (mean pooling, residual prediction, SIGReg) with one-sentence rationale each
- Links to HuggingFace checkpoint and dataset

---

### 4.4 Roadmap Panel (Footer / Slide-out)

**Purpose:** Signal that Oz Labs knows where this is going.

**Content:**
- v0 (current): Stable JEPA, noise burst detection, dynamics prediction
- v1 (in progress): Rollout-aware training — temporal composition
- v2 (planned): Token-level prediction — structural anomaly detection
- vN: Real RF data ingestion, online adaptation

One sentence per milestone. No dates. Honest about what's proven vs. planned.

---

## 5. Data & Inference Spec

### Trajectory selection
- Test set only (2,999 trajectories)
- Pre-indexed on server startup — no file I/O at request time
- Served by trajectory ID

### Inference pipeline (per request)
1. Load trajectory observations `[16, 256, 51, 2]`
2. Encode all 16 frames → `[16, 192]` embeddings
3. Run predictor on context frames → predicted Δz sequence
4. Compute surprise scores: per-step L2 distance between predicted and actual normalized embedding
5. If perturbation requested: apply perturbation to raw observations, re-encode, re-predict, return both curves
6. Extract per-patch norms from encoder (16×17 heatmap per frame) for attention overlay
7. Return: embeddings, predicted embeddings, surprise scores, patch norm heatmaps, regime metadata

### Latent space projection
- PCA fit on full test set embeddings (offline, at server startup)
- Per-request: project trajectory embeddings into 2D PCA space
- Background scatter: 300 randomly sampled test trajectories, pre-projected at startup

### Performance targets
- Inference latency: <500ms per trajectory on server GPU
- Perturbation injection: <800ms (includes re-encode)
- Page load: <2s (frontend is static, no blocking calls)

---

## 6. Visual Design Direction

**Aesthetic:** Precision instrument. Dark theme. Technical without being gratuitous. Feels like a signals analysis workstation, not a SaaS dashboard.

**Color palette:**
- Background: near-black (`#0A0B0E`)
- Surface: dark grey (`#13151A`)
- Border: subtle (`#1E2028`)
- Primary accent: Oz Labs orange (`#E8520A`)
- Success/normal: muted green (`#2A7A4B`)
- Alert/anomaly: amber → red gradient
- Spectrogram colormap: custom — dark navy (zero) through teal to bright amber (max power). Not viridis, not jet. Something that reads well as a signal.

**Typography:**
- Data/labels: monospace (IBM Plex Mono or similar) — this is technical instrument UI, numbers should be monospaced
- UI chrome: a geometric sans that isn't Inter. Something with a slight engineered quality.
- No decorative display fonts — this is not a marketing page

**Layout:**
- Three-panel main view is fixed-width, information-dense
- No gratuitous whitespace — this is a workstation, not a landing page
- Panels have clear visual hierarchy but equal weight
- Mobile is out of scope — this is a desktop demo

**Spectrogram rendering:**
- Canvas-based (not SVG) for performance
- Custom colormap applied at render time
- Smooth animation between timesteps (interpolated, not jump cuts)

---

## 7. What to Show in a Live Demo (Script)

A 10-minute demo flow for a client meeting:

1. **(1 min) Scene selection** — "We have 2,999 RF scenes across 8 behavioral regimes. Let me show you a correlated leader-follower scene — two signals where one consistently leads the other by 1–3 timesteps."

2. **(2 min) Playback + reconstruction** — "The left panel is what's actually happening in the RF environment. The right is what our model sees — what it encoded into a 192-dimensional latent representation. Notice it preserves the signal structure but discards noise."

3. **(2 min) Latent space** — "This is the model's internal representation of the scene evolving over time. Each point is a moment in time. The open circles are the model's predictions of where the scene will go — the filled circles are where it actually went. The delta cosine metric tells us how well the model predicts the *direction* of change."

4. **(3 min) Anomaly injection** — "Now I'm going to inject a noise burst at timestep 8. Watch the surprise score." [inject] "The model flags it at 1.66× baseline surprise — 80% detection rate across our test set. Now let me show you something honest: if I inject a new signal instead of a noise burst, the model currently doesn't flag it. That's a known limitation we're actively solving with our next architecture iteration."

5. **(2 min) Eval dashboard + roadmap** — "Here's the full picture across all 2,999 test scenes and all 8 RF regimes. And here's where we're going."

**The honest moment is the most important part of the demo.** Showing a limitation and explaining the roadmap to fix it is more credible to a technical defense audience than a demo that only shows successes.

---

## 8. Out of Scope (v1)

- Real-time SDR ingestion
- User-uploaded RF data
- Multi-model comparison UI
- Authentication / access control
- Mobile layout
- Export of full trajectory data (surprise scores CSV only)

---

## 9. Open Questions

| Question | Owner | Needed by |
|----------|-------|-----------|
| Which 5 curator trajectories best demonstrate model capability? | Guy (after eval results) | Before frontend build |
| ~~Post-hoc decoder~~ Replaced with per-patch norm attention overlay | Resolved — no decoder needed for v1 | N/A |
| Regime-stratified delta cosine results (pending training run) | Training run | Before Eval Dashboard |
| Server hosting — RunPod persistent instance or dedicated? | Guy | Before deployment |
| Oz Labs branding assets (logo, wordmark) for demo header | Guy | Before visual polish pass |

---

## 10. Build Sequence

Given the dependency on eval results (training run currently in progress):

**Now (pre-results):**
- Frontend scaffold: layout, navigation, color system, typography
- Panel A: spectrogram renderer with static test data
- Panel C: surprise score chart with hardcoded example data
- Backend: FastAPI skeleton, trajectory loading, inference pipeline

**After eval results (~6 hours):**
- Populate Eval Dashboard with real regime-stratified numbers
- Select curator trajectories
- Validate post-hoc decoder output quality
- Wire Panel B with real PCA projection

**After rollout-aware training eval:**
- Update baseline comparison table with new checkpoint results
- Update roadmap panel milestone status

**Polish pass (last):**
- Spectrogram colormap refinement
- Animation timing
- Demo script rehearsal
- Curator trajectory selection