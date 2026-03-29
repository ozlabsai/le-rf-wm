"""RF Spectral Trajectories dataset for PyTorch."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def compute_norm_stats(h5_path):
    """Compute global per-channel mean/std from an HDF5 file.

    Iterates in chunks to avoid loading the full dataset into memory.
    Returns dict with 'mean' and 'std', each of shape (2,).
    """
    h5_path = str(h5_path)
    chunk_size = 500

    with h5py.File(h5_path, "r") as f:
        obs = f["observations"]  # [N, 16, 256, 51, 2] float16
        N = obs.shape[0]

        # online Welford-style accumulation
        total = 0
        channel_sum = np.zeros(2, dtype=np.float64)
        channel_sum_sq = np.zeros(2, dtype=np.float64)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = obs[start:end].astype(np.float32)  # (B, 16, 256, 51, 2)
            # sum over all dims except last (channel)
            n_elements = chunk[..., 0].size  # elements per channel
            channel_sum += chunk.sum(axis=(0, 1, 2, 3))
            channel_sum_sq += (chunk ** 2).sum(axis=(0, 1, 2, 3))
            total += n_elements

    mean = channel_sum / total
    std = np.sqrt(channel_sum_sq / total - mean ** 2)
    std = np.maximum(std, 1e-8)

    return {"mean": mean.tolist(), "std": std.tolist()}


def save_norm_stats(stats, path):
    """Save normalization stats to JSON."""
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(path):
    """Load normalization stats from JSON."""
    with open(path, "r") as f:
        return json.load(f)


class RFSpectralDataset(Dataset):
    """Loads RF spectrogram trajectories from HDF5.

    Each HDF5 file contains:
        observations: [N, 16, 256, 51, 2] float16
        timestamps:   [N, 16] float64

    Returns subsequences of length (history_size + num_preds) from
    the 16-step trajectories.

    Args:
        path: path to HDF5 file
        history_size: number of context frames
        num_preds: number of prediction targets
        norm_stats: dict with 'mean' and 'std' (each length-2 list),
                    or path to JSON file. None disables normalization.
        transform: optional additional transform applied after normalization
    """

    def __init__(self, path, history_size=3, num_preds=1, norm_stats=None,
                 transform=None, preload=True):
        self.path = str(path)
        self.history_size = history_size
        self.num_preds = num_preds
        self.seq_len = history_size + num_preds
        self.transform = transform

        # load normalization stats
        if norm_stats is None:
            self._mean = None
            self._std = None
        elif isinstance(norm_stats, (str, Path)):
            stats = load_norm_stats(norm_stats)
            self._mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 2, 1, 1)
            self._std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 2, 1, 1)
        else:
            self._mean = torch.tensor(norm_stats["mean"], dtype=torch.float32).view(1, 2, 1, 1)
            self._std = torch.tensor(norm_stats["std"], dtype=torch.float32).view(1, 2, 1, 1)

        # load data
        with h5py.File(self.path, "r") as f:
            self.n_trajectories = f["observations"].shape[0]
            self.traj_len = f["observations"].shape[1]  # 16

            if preload:
                # load entire array into RAM — (N, 16, F, T, 2) float32
                # permute to (N, 16, 2, F, T) and apply normalization upfront
                print(f"Preloading {self.path} into RAM...")
                raw = torch.from_numpy(f["observations"][()]).float()
                self._data = raw.permute(0, 1, 4, 2, 3)  # (N, 16, 2, F, T)
                if self._mean is not None:
                    self._data = (self._data - self._mean) / self._std
                print(f"  Loaded {self._data.shape}, {self._data.element_size() * self._data.nelement() / 1e9:.1f} GB")
            else:
                self._data = None

        self.subs_per_traj = self.traj_len - self.seq_len + 1
        assert self.subs_per_traj > 0, (
            f"traj_len {self.traj_len} too short for seq_len {self.seq_len}"
        )

        # lazy h5py handle for non-preloaded mode
        self._file = None

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self):
        return self.n_trajectories * self.subs_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.subs_per_traj
        step_offset = idx % self.subs_per_traj

        if self._data is not None:
            # fast path: slice from RAM, already normalized
            obs = self._data[traj_idx, step_offset : step_offset + self.seq_len]
        else:
            # slow path: read from HDF5
            f = self._get_file()
            obs = f["observations"][traj_idx, step_offset : step_offset + self.seq_len]
            obs = torch.from_numpy(obs).float()
            obs = obs.permute(0, 3, 1, 2)  # (T, 2, 256, 51)
            if self._mean is not None:
                obs = (obs - self._mean) / self._std

        sample = {"observations": obs}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
