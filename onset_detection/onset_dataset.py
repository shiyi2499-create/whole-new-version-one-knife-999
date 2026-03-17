"""
Onset Detection Dataset
=======================

PyTorch dataset wrapper for:
  - onset (binary)
  - activity (binary, legacy)
  - password_boundary (4-class)

The implementation stays small but supports:
  - session-level splitting
  - train-only normalization
  - lightweight augmentation
  - balanced sampling for binary or multi-class labels
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


class OnsetWindowDataset(Dataset):
    """Windowed dataset for binary or multi-class training."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        normalize: bool = True,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None,
        n_classes: Optional[int] = None,
    ):
        self.windows = windows.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment
        self.n_classes = int(n_classes) if n_classes is not None else int(np.max(self.labels) + 1) if len(self.labels) else 2
        self.is_multiclass = self.n_classes > 2 or np.max(self.labels, initial=0) > 1

        if normalize:
            if means is None:
                means = self.windows.mean(axis=(0, 1)) if len(self.windows) else np.zeros(self.windows.shape[-1], dtype=np.float32)
            if stds is None:
                stds = self.windows.std(axis=(0, 1)) if len(self.windows) else np.ones(self.windows.shape[-1], dtype=np.float32)
            self.means = means.astype(np.float32)
            self.stds = np.maximum(stds.astype(np.float32), 1e-10)
            self.windows = (self.windows - self.means) / self.stds
        else:
            self.means = np.zeros(self.windows.shape[-1], dtype=np.float32)
            self.stds = np.ones(self.windows.shape[-1], dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        w = torch.from_numpy(self.windows[idx].copy())
        lbl = self.labels[idx]
        if self.augment:
            w = self._apply_augmentation(w)
        if self.is_multiclass:
            y = torch.tensor(lbl, dtype=torch.long)
        else:
            y = torch.tensor(float(lbl), dtype=torch.float32)
        return w, y

    @staticmethod
    def _apply_augmentation(w: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        if torch.rand(1).item() > p:
            return w
        T, C = w.shape
        aug_type = torch.randint(0, 5, (1,)).item()
        if aug_type == 0:
            shift = torch.randint(-max(1, T // 10), max(2, T // 10 + 1), (1,)).item()
            w = torch.roll(w, shifts=shift, dims=0)
        elif aug_type == 1:
            std = max(w.std().item(), 1e-6) * 0.02
            w = w + torch.randn_like(w) * std
        elif aug_type == 2:
            scale = 0.85 + 0.30 * torch.rand(1).item()
            w = w * scale
        elif aug_type == 3:
            ch = torch.randint(0, C, (1,)).item()
            w[:, ch] = 0.0
        else:
            cut = max(1, T // 12)
            start = torch.randint(0, max(1, T - cut), (1,)).item()
            w[start:start + cut] = 0.0
        return w


# ── Session-level split ──────────────────────────────────────

def session_split(
    sessions: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split by session id to avoid leakage across windows."""
    rng = np.random.RandomState(seed)
    unique_sessions = np.array(sorted(set(sessions.tolist())))
    if len(unique_sessions) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    rng.shuffle(unique_sessions)
    n = len(unique_sessions)
    if n == 1:
        idx = np.arange(len(sessions))
        return idx, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if n == 2:
        train_sessions = {unique_sessions[0]}
        test_sessions = {unique_sessions[1]}
        train_idx = np.where(np.isin(sessions, list(train_sessions)))[0]
        test_idx = np.where(np.isin(sessions, list(test_sessions)))[0]
        return train_idx, np.array([], dtype=np.int64), test_idx

    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_sessions = set(unique_sessions[:n_train].tolist())
    val_sessions = set(unique_sessions[n_train:n_train + n_val].tolist())
    test_sessions = set(unique_sessions[n_train + n_val:].tolist())
    if not test_sessions:
        moved = next(iter(val_sessions))
        val_sessions.remove(moved)
        test_sessions.add(moved)

    train_idx = np.where(np.isin(sessions, list(train_sessions)))[0]
    val_idx = np.where(np.isin(sessions, list(val_sessions)))[0]
    test_idx = np.where(np.isin(sessions, list(test_sessions)))[0]
    return train_idx, val_idx, test_idx


# ── Balanced sampler ─────────────────────────────────────────

def make_balanced_sampler(labels: np.ndarray, n_classes: Optional[int] = None) -> WeightedRandomSampler:
    labels = labels.astype(np.int64)
    if n_classes is None:
        n_classes = int(np.max(labels) + 1) if len(labels) else 2
    counts = np.bincount(labels, minlength=n_classes)
    weights_per_class = 1.0 / np.maximum(counts, 1).astype(np.float64)
    sample_weights = weights_per_class[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(labels),
        replacement=True,
    )


# ── Loader ───────────────────────────────────────────────────

def load_onset_dataset(path: str) -> dict:
    """Load an onset/activity/password_boundary dataset from .npz."""
    data = np.load(path, allow_pickle=True)
    result = {
        "windows": data["windows"],
        "labels": data["labels"].astype(np.int64),
        "times_s": data["times_s"],
        "sessions": data["sessions"].astype(str),
        "sources": data["sources"].astype(str),
        "window_ms": int(data["window_ms"]),
        "stride_ms": int(data["stride_ms"]),
        "label_radius_ms": int(data["label_radius_ms"]),
        "target_rate_hz": int(data["target_rate_hz"]),
        "n_channels": int(data["n_channels"]),
    }
    if "task" in data:
        result["task"] = str(data["task"])
    else:
        result["task"] = "onset"

    if "activity_labels" in data:
        result["activity_labels"] = data["activity_labels"].astype(str)
    if "boundary_labels" in data:
        result["boundary_labels"] = data["boundary_labels"].astype(str)
    if "label_names" in data:
        result["label_names"] = data["label_names"].astype(str)
    else:
        if result["task"] == "password_boundary":
            result["label_names"] = np.array(["non_password", "password_start", "password_active", "password_end"])
        else:
            result["label_names"] = np.array(["negative", "positive"])

    result["n_classes"] = int(len(result["label_names"]))
    return result
