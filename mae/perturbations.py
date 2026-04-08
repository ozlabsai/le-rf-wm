"""RF perturbation functions for anomaly detection testing.

Each function operates on raw observations [T, 256, 51, 2] (real/imag STFT)
and returns a modified copy. Perturbations are applied before encoding.
"""

import torch
import numpy as np


def noise_burst(obs, timestep, freq_range=(0, 256), intensity=3.0):
    """Inject broadband noise at a single timestep.

    Args:
        obs: (T, 256, 51, 2) real/imag observations
        timestep: which frame to perturb
        freq_range: (lo, hi) frequency bin range
        intensity: noise magnitude multiplier
    Returns:
        perturbed copy of obs
    """
    out = obs.clone()
    lo, hi = freq_range
    noise = torch.randn_like(out[timestep, lo:hi, :, :]) * intensity
    out[timestep, lo:hi, :, :] += noise
    return out


def signal_injection(obs, timestep, freq_center=128, bandwidth=10, power=1.0, duration=4):
    """Inject a narrowband signal starting at timestep.

    Args:
        obs: (T, 256, 51, 2)
        timestep: start frame
        freq_center: center frequency bin
        bandwidth: number of frequency bins
        power: signal power
        duration: number of frames to persist
    """
    out = obs.clone()
    T = out.shape[0]
    lo = max(0, freq_center - bandwidth // 2)
    hi = min(256, freq_center + bandwidth // 2)
    for t in range(timestep, min(timestep + duration, T)):
        # Inject a constant-power tone across time bins
        out[t, lo:hi, :, 0] += power  # real component
        out[t, lo:hi, :, 1] += power * 0.5  # imag component (phase offset)
    return out


def signal_dropout(obs, timestep, freq_range=(50, 200), duration=4):
    """Zero out a frequency band for several frames.

    Args:
        obs: (T, 256, 51, 2)
        timestep: start frame
        freq_range: (lo, hi) frequency bins to zero
        duration: number of frames
    """
    out = obs.clone()
    T = out.shape[0]
    lo, hi = freq_range
    for t in range(timestep, min(timestep + duration, T)):
        out[t, lo:hi, :, :] = 0.0
    return out


def frequency_shift(obs, timestep, shift_bins=20):
    """Shift the frequency content by shift_bins starting at timestep.

    Args:
        obs: (T, 256, 51, 2)
        timestep: first frame to shift
        shift_bins: positive = shift up, negative = shift down
    """
    out = obs.clone()
    T = out.shape[0]
    for t in range(timestep, T):
        frame = out[t].clone()
        out[t] = 0.0
        if shift_bins > 0:
            out[t, shift_bins:, :, :] = frame[:-shift_bins, :, :]
        elif shift_bins < 0:
            out[t, :shift_bins, :, :] = frame[-shift_bins:, :, :]
    return out


def temporal_reversal(obs, start_step, end_step):
    """Reverse the temporal order of frames in [start_step, end_step).

    Args:
        obs: (T, 256, 51, 2)
        start_step: first frame (inclusive)
        end_step: last frame (exclusive)
    """
    out = obs.clone()
    out[start_step:end_step] = obs[start_step:end_step].flip(0)
    return out
