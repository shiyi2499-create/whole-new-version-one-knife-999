"""
Onset Detection Dataset
=======================
PyTorch Dataset wrapper for onset detection training / evaluation.

Supports:
  - Loading from onset_dataset.npz (output of onset_preprocessor.py)
  - Session-level train/val/test splitting (prevents data leakage)
  - On-the-fly augmentation (time jitter, noise, scaling, channel dropout)
  - Class-balanced sampling via WeightedRandomSampler
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import defaultdict
from typing import Optional


class OnsetWindowDataset(Dataset):
    """
    Windowed onset detection dataset.

    Each sample is a (window, label) pair where:
      window: (timesteps, 6)  float32
      label:  int  {0, 1}
    """

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        normalize: bool = True,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
    ):
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

        # Channel-wise normalization
        if normalize:
            if means is None:
                means = self.windows.mean(axis=(0, 1))
            if stds is None:
                stds = self.windows.std(axis=(0, 1))
            self.means = means.astype(np.float32)
            self.stds = np.maximum(stds.astype(np.float32), 1e-10)
            self.windows = (self.windows - self.means) / self.stds
        else:
            self.means = np.zeros(windows.shape[-1], dtype=np.float32)
            self.stds = np.ones(windows.shape[-1], dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        w = torch.from_numpy(self.windows[idx].copy())
        lbl = self.labels[idx]

        if self.augment:
            w = self._apply_augmentation(w)

        return w, torch.tensor(lbl, dtype=torch.float32)

    @staticmethod
    def _apply_augmentation(w: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """
        Light augmentation for onset detection.
        Applied independently per sample with probability p.
        """
        if torch.rand(1).item() > p:
            return w

        T, C = w.shape
        aug_type = torch.randint(0, 4, (1,)).item()

        if aug_type == 0:
            # Time shift: roll by ±10%
            shift = torch.randint(-max(1, T // 10), max(2, T // 10 + 1), (1,)).item()
            w = torch.roll(w, shifts=shift, dims=0)
        elif aug_type == 1:
            # Additive Gaussian noise
            std = w.std() * 0.02
            w = w + torch.randn_like(w) * std
        elif aug_type == 2:
            # Amplitude scaling
            scale = 0.8 + 0.4 * torch.rand(1).item()
            w = w * scale
        elif aug_type == 3:
            # Channel dropout
            ch = torch.randint(0, C, (1,)).item()
            w[:, ch] = 0.0

        return w


# ── Session-level splitting ──────────────────────────────────

def session_split(
    sessions: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into train/val/test by session ID.
    No session appears in more than one split.

    Returns:
        (train_indices, val_indices, test_indices) as numpy arrays
    """
    rng = np.random.RandomState(seed)

    unique_sessions = np.array(sorted(set(sessions.tolist())))
    rng.shuffle(unique_sessions)

    n = len(unique_sessions)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))

    train_sessions = set(unique_sessions[:n_train].tolist())
    val_sessions = set(unique_sessions[n_train:n_train + n_val].tolist())
    test_sessions = set(unique_sessions[n_train + n_val:].tolist())

    # If test is empty (too few sessions), steal from val
    if not test_sessions and len(val_sessions) > 1:
        test_sessions = {val_sessions.pop()}

    train_idx = np.where(np.isin(sessions, list(train_sessions)))[0]
    val_idx = np.where(np.isin(sessions, list(val_sessions)))[0]
    test_idx = np.where(np.isin(sessions, list(test_sessions)))[0]

    return train_idx, val_idx, test_idx


# ── Balanced sampler ─────────────────────────────────────────

def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that balances positive/negative windows."""
    counts = np.bincount(labels.astype(int), minlength=2)
    weights_per_class = 1.0 / np.maximum(counts, 1).astype(np.float64)
    sample_weights = weights_per_class[labels.astype(int)]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )


# ── Loader from npz ─────────────────────────────────────────

def load_onset_dataset(path: str) -> dict:
    """Load onset_dataset.npz and return all arrays + metadata."""
    data = np.load(path, allow_pickle=True)
    return {
        "windows": data["windows"],
        "labels": data["labels"],
        "times_s": data["times_s"],
        "sessions": data["sessions"].astype(str),
        "sources": data["sources"].astype(str),
        "window_ms": int(data["window_ms"]),
        "stride_ms": int(data["stride_ms"]),
        "label_radius_ms": int(data["label_radius_ms"]),
        "target_rate_hz": int(data["target_rate_hz"]),
        "n_channels": int(data["n_channels"]),
    }
