"""
Signal processing utilities for stage2_ctc.
Reuses core logic from stage2_episode, local copy for package independence.
"""
import numpy as np


def compute_magnitude(data: np.ndarray) -> np.ndarray:
    """[T, 6] -> [T, 8] with ||accel|| and ||gyro|| appended."""
    accel_mag = np.linalg.norm(data[:, 0:3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)
    return np.concatenate([data[:, :6], accel_mag, gyro_mag], axis=1)


def normalize(data, mean=None, std=None):
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
        std[std < 1e-8] = 1.0
    return (data - mean) / std, mean, std


def preprocess(data: np.ndarray, sample_rate=100, add_mag=True, norm=True,
               stats=None):
    """Full preprocess: magnitude -> normalize. Returns (processed, stats)."""
    out = data[:, :6].copy().astype(np.float32)
    if add_mag:
        out = compute_magnitude(out)
    st = {}
    if norm:
        m, s = (stats['mean'], stats['std']) if stats else (None, None)
        out, m, s = normalize(out, m, s)
        st = {'mean': m, 'std': s}
    return out, st


def time_stretch(data: np.ndarray, rate: float) -> np.ndarray:
    """rate > 1 = speed up (shorter output)."""
    T, C = data.shape
    new_T = max(int(T / rate), 2)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_T)
    out = np.zeros((new_T, C), dtype=data.dtype)
    for c in range(C):
        out[:, c] = np.interp(x_new, x_old, data[:, c])
    return out
