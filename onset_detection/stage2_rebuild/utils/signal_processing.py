"""
Signal processing utilities for IMU data.
"""
import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional


def compute_magnitude(data: np.ndarray) -> np.ndarray:
    """
    Compute vector magnitude for 3-axis groups.
    Input: [T, C] where C=6 (accel_xyz, gyro_xyz)
    Output: [T, C+2] with accel_mag and gyro_mag appended.
    """
    assert data.shape[1] >= 6, f"Expected >=6 channels, got {data.shape[1]}"
    accel_mag = np.linalg.norm(data[:, 0:3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)
    return np.concatenate([data[:, :6], accel_mag, gyro_mag], axis=1)


def normalize_signal(data: np.ndarray,
                     mean: Optional[np.ndarray] = None,
                     std: Optional[np.ndarray] = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-channel z-score normalization.
    Returns (normalized_data, mean, std).
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
        std[std < 1e-8] = 1.0  # avoid div by zero
    return (data - mean) / std, mean, std


def bandpass_filter(data: np.ndarray,
                    low: float, high: float,
                    fs: int, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter per channel.
    """
    nyq = fs / 2.0
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    b, a = scipy_signal.butter(order, [low_n, high_n], btype='band')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = scipy_signal.filtfilt(b, a, data[:, ch])
    return filtered


def preprocess_imu(data: np.ndarray,
                   sample_rate: int = 100,
                   normalize: bool = True,
                   bandpass: bool = True,
                   bandpass_low: float = 0.5,
                   bandpass_high: float = 45.0,
                   add_magnitude: bool = True,
                   norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
                   ) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for raw IMU data.

    Input: [T, 6] raw accel_xyz + gyro_xyz
    Output: [T, C_out] preprocessed, plus stats dict

    Returns:
        processed: np.ndarray [T, C_out]
        stats: dict with 'mean', 'std' for reproducibility
    """
    out = data.copy()

    # Bandpass filter
    if bandpass and sample_rate > 2 * bandpass_high:
        out = bandpass_filter(out, bandpass_low, bandpass_high, sample_rate)

    # Add magnitude channels
    if add_magnitude:
        out = compute_magnitude(out)

    # Normalize
    stats = {}
    if normalize:
        if norm_stats is not None:
            out, m, s = normalize_signal(out, norm_stats[0], norm_stats[1])
        else:
            out, m, s = normalize_signal(out)
        stats['mean'] = m
        stats['std'] = s

    return out, stats


def compute_energy_envelope(data: np.ndarray,
                            window_size: int = 50) -> np.ndarray:
    """
    Compute sliding-window energy envelope.
    Useful for Stage 2A as an auxiliary feature.

    Input: [T, C]
    Output: [T] energy values
    """
    energy = np.sum(data ** 2, axis=1)
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(energy, kernel, mode='same')
    return smoothed


def splice_smooth(sig1: np.ndarray, sig2: np.ndarray,
                  overlap_samples: int = 10) -> np.ndarray:
    """
    Smoothly splice two signals using overlap-add with linear crossfade.
    Both signals are [T_i, C].
    """
    if overlap_samples <= 0 or overlap_samples > min(len(sig1), len(sig2)):
        return np.concatenate([sig1, sig2], axis=0)

    fade_out = np.linspace(1.0, 0.0, overlap_samples)[:, None]
    fade_in = np.linspace(0.0, 1.0, overlap_samples)[:, None]

    blended = sig1[-overlap_samples:] * fade_out + sig2[:overlap_samples] * fade_in
    result = np.concatenate([
        sig1[:-overlap_samples],
        blended,
        sig2[overlap_samples:]
    ], axis=0)
    return result


def time_stretch(data: np.ndarray, rate: float) -> np.ndarray:
    """
    Simple time stretch via linear interpolation.
    rate > 1 = speed up (shorter), rate < 1 = slow down (longer).
    """
    T, C = data.shape
    new_T = int(T / rate)
    if new_T < 2:
        return data
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, new_T)
    stretched = np.zeros((new_T, C))
    for ch in range(C):
        stretched[:, ch] = np.interp(x_new, x_old, data[:, ch])
    return stretched
