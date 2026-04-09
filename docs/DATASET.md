---
license: mit
task_categories:
  - time-series-forecasting
tags:
  - radio-frequency
  - spectrogram
  - signal-processing
  - simulation
  - trajectory
  - stft
  - iq-data
  - wireless-communication
pretty_name: RF Spectral Trajectories
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.h5
      - split: validation
        path: val.h5
      - split: test
        path: test.h5
---

# RF Spectral Trajectories

Temporally ordered sequences of RF (radio frequency) spectrograms from simulated wideband communication environments. Each trajectory is a sliding window of 16 consecutive STFT observations from a continuous RF scene, capturing how multiple signals appear, disappear, drift in frequency, and vary in power over time.

## Dataset Structure

### Files

| File | Split | Trajectories | Size |
|------|-------|-------------|------|
| `train.h5` | train | 13,841 | 10 GB |
| `val.h5` | validation | 2,938 | 1.1 GB |
| `test.h5` | test | 2,999 | 1.2 GB |

### HDF5 Schema

Each file contains:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `observations` | `[N, 16, 256, 51, 2]` | float16 | STFT magnitude (real, imaginary channels) |
| `timestamps` | `[N, 16]` | float64 | Time in seconds for each timestep |
| `source_ids` | `[N]` | string | Scene identifier (for provenance) |
| `sequence_ids` | `[N]` | string | Unique trajectory identifier |

**Observation tensor dimensions:**
- `N` — number of trajectories
- `16` — timesteps per trajectory (80ms each, 1.28s total)
- `256` — frequency bins (STFT, covering ±75 kHz)
- `51` — time bins within each 80ms STFT window
- `2` — real and imaginary components

### Splits

Split by **source scene** (not by trajectory) to prevent temporal leakage from overlapping sliding windows.

| Split | Scenes | Ratio |
|-------|--------|-------|
| train | 280 | 70% |
| validation | 60 | 15% |
| test | 60 | 15% |

## Loading

### With h5py (recommended for large files)

```python
import h5py
import numpy as np

with h5py.File("train.h5", "r") as f:
    # Lazy access — reads only what you index
    obs = f["observations"][0]      # [16, 256, 51, 2] float16
    ts = f["timestamps"][0]         # [16] float64
    src = f["source_ids"][0]        # bytes
```

### With PyTorch

```python
from lewm_pipeline.dataset import LeWMDataset

ds = LeWMDataset("train.h5")
item = ds[0]
# item["observations"]: torch.float32 tensor [16, 256, 51, 2]
# item["timestamps"]: torch.float64 tensor [16]
# item["source_id"]: str
# item["sequence_id"]: str
```

## Generation Parameters

### Signal Simulation

| Parameter | Value |
|-----------|-------|
| Sample rate | 150 kHz |
| Timestep duration | 80 ms (12,000 samples) |
| Scene duration | 5.2 s (65 timesteps) |
| Trajectory length | 16 timesteps (1.28 s) |
| Sliding window stride | 1 timestep |
| Modulation types | BPSK, QPSK, 8PSK, 16QAM, 64QAM |
| Signals per scene | 2–4 (randomly placed in frequency) |
| Channel models | Rayleigh, Rician (with evolving fading) |
| SNR range | -8 to +30 dB |
| Doppler speeds | 0–12 m/s |

### STFT Parameters

| Parameter | Value |
|-----------|-------|
| Window | Hamming, 256 samples |
| Overlap | 16 samples |
| FFT size | 256 |
| Sided | Two-sided (complex input) |

### Activity Regimes

Each scene follows one of 8 activity regimes (50 scenes each, 400 total):

| Regime | Description |
|--------|-------------|
| `quiet` | Few active signals, low duty cycle, long silences |
| `dense` | Many signals active simultaneously, high overlap |
| `bursty` | Rapid on/off transitions, short bursts |
| `ramp_up` | Signals appear progressively through the scene |
| `interference_event` | Stable signals, then a disruptive signal appears mid-scene |
| `correlated_alternating` | Signal pairs alternate: A on when B is off |
| `correlated_leader_follower` | Signal B appears 1–3 timesteps after signal A |
| `random` | Independent random burst patterns |

### Dynamic Features

- **Frequency drift**: Signals slowly wander in frequency (random walk, bounded to ±50% of signal bandwidth)
- **Power variation**: Smooth per-timestep power levels (0.3–1.0 when active), gradual fade-in/out, correlated power drift
- **Channel fading**: Rayleigh/Rician fading evolves continuously within each scene (no state reset between timesteps)

## Source

Generated using the [ChangShuoRadioData (CSRD)](https://github.com/Singingkettle/ChangShuoRadioData) MATLAB simulation framework with the LeWM dataset pipeline.

### Citation

```bibtex
@software{csrd_rf_spectral_trajectories_2026,
  title = {RF Spectral Trajectories},
  author = {Ozlabs},
  year = {2026},
  url = {https://huggingface.co/datasets/ozlabs/rf-spectral-trajectories}
}
```

### Related

- [ChangShuoRadioData](https://github.com/Singingkettle/ChangShuoRadioData) — MATLAB RF simulation framework
- [Joint Signal Detection and AMC via Deep Learning](https://ieeexplore.ieee.org/abstract/document/10667001) — IEEE TWC paper using CSRD data
