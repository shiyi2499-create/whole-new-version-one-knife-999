from __future__ import annotations

import numpy as np


def append_diff_channels(window: np.ndarray) -> np.ndarray:
    """
    Append first-order temporal difference channels.

    Supports:
    - (T, C) -> (T, 2C)
    - (B, T, C) -> (B, T, 2C)
    """
    w = np.asarray(window, dtype=np.float32)
    if w.ndim == 2:
        diff = np.diff(w, axis=0)
        diff = np.pad(diff, ((0, 1), (0, 0)), mode="edge")
        return np.concatenate([w, diff], axis=1)
    if w.ndim == 3:
        diff = np.diff(w, axis=1)
        diff = np.pad(diff, ((0, 0), (0, 1), (0, 0)), mode="edge")
        return np.concatenate([w, diff], axis=2)
    raise ValueError(f"Expected 2D or 3D input, got shape={w.shape}")
