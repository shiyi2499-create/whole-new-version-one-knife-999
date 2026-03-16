"""
Feature Extractor
=================
Converts raw sensor windows (30 timesteps × 6 channels) into
fixed-length feature vectors for traditional ML models.

Features per channel (6 channels × 16 = 96):
  Time domain (11): mean, std, max, min, peak-to-peak, RMS,
                     skewness, kurtosis, zero-crossing rate,
                     peak position, energy
  Frequency domain (5): dominant freq, dominant amplitude,
                         spectral centroid, spectral bandwidth,
                         spectral energy

Cross-channel features (20):
  - NCC peaks between all channel pairs: C(6,2) = 15
  - Accel magnitude stats: 5

Total: 96 + 15 + 5 = 116 features
"""

import numpy as np
from itertools import combinations


CHANNEL_NAMES = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


def _time_domain_features(signal: np.ndarray) -> list[float]:
    """Extract 11 time-domain features from a 1D signal."""
    n = len(signal)
    if n == 0:
        return [0.0] * 11

    mean = np.mean(signal)
    std = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    peak_to_peak = max_val - min_val
    rms = np.sqrt(np.mean(signal ** 2))

    # Skewness
    if std > 1e-10:
        skewness = np.mean(((signal - mean) / std) ** 3)
    else:
        skewness = 0.0

    # Kurtosis
    if std > 1e-10:
        kurtosis = np.mean(((signal - mean) / std) ** 4) - 3.0
    else:
        kurtosis = 0.0

    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.sign(signal - mean)) != 0)
    zcr = zero_crossings / max(n - 1, 1)

    # Peak position (normalized 0-1, where in the window the max occurs)
    peak_pos = np.argmax(np.abs(signal)) / max(n - 1, 1)

    # Energy
    energy = np.sum(signal ** 2)

    return [mean, std, max_val, min_val, peak_to_peak,
            rms, skewness, kurtosis, zcr, peak_pos, energy]


def _freq_domain_features(signal: np.ndarray, sample_rate: float = 100.0) -> list[float]:
    """Extract 5 frequency-domain features from a 1D signal."""
    n = len(signal)
    if n < 4:
        return [0.0] * 5

    # FFT (real signal, take positive frequencies only)
    fft_vals = np.fft.rfft(signal)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Skip DC component (index 0)
    if len(fft_mag) > 1:
        fft_mag_no_dc = fft_mag[1:]
        freqs_no_dc = freqs[1:]
    else:
        return [0.0] * 5

    total_energy = np.sum(fft_mag_no_dc ** 2)

    if total_energy < 1e-15:
        return [0.0] * 5

    # Dominant frequency and its amplitude
    dom_idx = np.argmax(fft_mag_no_dc)
    dom_freq = freqs_no_dc[dom_idx]
    dom_amp = fft_mag_no_dc[dom_idx]

    # Spectral centroid (weighted average frequency)
    spectral_centroid = np.sum(freqs_no_dc * fft_mag_no_dc) / np.sum(fft_mag_no_dc)

    # Spectral bandwidth (weighted std of frequencies)
    spectral_bw = np.sqrt(
        np.sum(((freqs_no_dc - spectral_centroid) ** 2) * fft_mag_no_dc)
        / np.sum(fft_mag_no_dc)
    )

    # Spectral energy
    spectral_energy = total_energy

    return [dom_freq, dom_amp, spectral_centroid, spectral_bw, spectral_energy]


def _ncc_peak(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    Compute peak of normalized cross-correlation between two signals.
    Ref: de Souza Faria (2013) - NCC as primary feature for vibration-based
    key identification.
    """
    a = sig_a - np.mean(sig_a)
    b = sig_b - np.mean(sig_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    # Full cross-correlation via FFT
    n = len(a) + len(b) - 1
    fft_size = 1
    while fft_size < n:
        fft_size *= 2

    fft_a = np.fft.rfft(a, fft_size)
    fft_b = np.fft.rfft(b, fft_size)
    cc = np.fft.irfft(fft_a * np.conj(fft_b), fft_size)

    # Normalize
    cc = cc / (norm_a * norm_b)

    return float(np.max(np.abs(cc)))


def extract_features_single(window: np.ndarray, sample_rate: float = 100.0) -> np.ndarray:
    """
    Extract feature vector from a single window.

    Args:
        window: shape (T, 6) - one keystroke window
        sample_rate: Hz

    Returns:
        1D feature vector of length 116
    """
    n_channels = window.shape[1]
    features = []

    # ── Per-channel features (6 × 16 = 96) ──
    for ch in range(n_channels):
        sig = window[:, ch]
        features.extend(_time_domain_features(sig))     # 11
        features.extend(_freq_domain_features(sig, sample_rate))  # 5

    # ── Cross-channel NCC peaks: C(6,2) = 15 ──
    for i, j in combinations(range(n_channels), 2):
        ncc = _ncc_peak(window[:, i], window[:, j])
        features.append(ncc)

    # ── Accelerometer magnitude features (5) ──
    accel_mag = np.sqrt(
        window[:, 0] ** 2 + window[:, 1] ** 2 + window[:, 2] ** 2
    )
    mag_mean = np.mean(accel_mag)
    mag_std = np.std(accel_mag)
    mag_max = np.max(accel_mag)
    mag_min = np.min(accel_mag)
    mag_rms = np.sqrt(np.mean(accel_mag ** 2))
    features.extend([mag_mean, mag_std, mag_max, mag_min, mag_rms])

    return np.array(features, dtype=np.float64)


def extract_features_batch(X: np.ndarray, sample_rate: float = 100.0) -> np.ndarray:
    """
    Extract features for all samples.

    Args:
        X: shape (N, T, 6) - all windows
        sample_rate: Hz

    Returns:
        shape (N, 116) feature matrix
    """
    N = X.shape[0]
    feat_list = []

    for i in range(N):
        f = extract_features_single(X[i], sample_rate)
        feat_list.append(f)

        if (i + 1) % 200 == 0:
            print(f"    Extracted features: {i+1}/{N}")

    features = np.array(feat_list)
    print(f"    ✓ Feature extraction complete: {features.shape}")
    return features


def get_feature_names() -> list[str]:
    """Return human-readable names for all 116 features."""
    names = []

    time_names = ["mean", "std", "max", "min", "p2p",
                  "rms", "skew", "kurt", "zcr", "peak_pos", "energy"]
    freq_names = ["dom_freq", "dom_amp", "spec_centroid", "spec_bw", "spec_energy"]

    for ch in CHANNEL_NAMES:
        for tn in time_names:
            names.append(f"{ch}_{tn}")
        for fn in freq_names:
            names.append(f"{ch}_{fn}")

    for i, j in combinations(range(6), 2):
        names.append(f"ncc_{CHANNEL_NAMES[i]}_{CHANNEL_NAMES[j]}")

    for s in ["mean", "std", "max", "min", "rms"]:
        names.append(f"accel_mag_{s}")

    return names


# ── Zone / Region label mappings ─────────────────────────────

ZONE_MAPS = {
    "row": {
        # Row on keyboard (QWERTY layout)
        "q": 0, "w": 0, "e": 0, "r": 0, "t": 0,
        "y": 0, "u": 0, "i": 0, "o": 0, "p": 0,
        "a": 1, "s": 1, "d": 1, "f": 1, "g": 1,
        "h": 1, "j": 1, "k": 1, "l": 1,
        "z": 2, "x": 2, "c": 2, "v": 2, "b": 2,
        "n": 2, "m": 2,
        "0": 3, "1": 3, "2": 3, "3": 3, "4": 3,
        "5": 3, "6": 3, "7": 3, "8": 3, "9": 3,
        # Special keys
        # enter/backspace are on the far right of the qwerty/number rows
        "enter": 0,      # far-right of qwerty row
        "backspace": 3,  # far-right of number row
        # comma/period are on the bottom row
        ",": 2, ".": 2,
        # space and shift sit on the lowest physical row
        "space": 4, "shift": 4,
    },
    "hand": {
        # Left vs Right hand (standard touch typing)
        "q": 0, "w": 0, "e": 0, "r": 0, "t": 0,
        "a": 0, "s": 0, "d": 0, "f": 0, "g": 0,
        "z": 0, "x": 0, "c": 0, "v": 0, "b": 0,
        "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
        "y": 1, "u": 1, "i": 1, "o": 1, "p": 1,
        "h": 1, "j": 1, "k": 1, "l": 1,
        "n": 1, "m": 1,
        "6": 1, "7": 1, "8": 1, "9": 1, "0": 1,
        # Special keys
        "enter": 1,     # right hand (pinky)
        "backspace": 1, # right hand (pinky)
        ",": 1,         # right hand
        ".": 1,         # right hand
        "space": 1,     # right thumb (most common)
        "shift": 0,     # left hand (left shift more common)
    },
    "quadrant": {
        # 4 quadrants: top-left, top-right, bottom-left, bottom-right
        "q": 0, "w": 0, "e": 0, "r": 0, "t": 0,
        "a": 0, "s": 0, "d": 0, "f": 0, "g": 0,
        "1": 0, "2": 0, "3": 0, "4": 0, "5": 0,
        "y": 1, "u": 1, "i": 1, "o": 1, "p": 1,
        "h": 1, "j": 1, "k": 1, "l": 1,
        "6": 1, "7": 1, "8": 1, "9": 1, "0": 1,
        "z": 2, "x": 2, "c": 2, "v": 2, "b": 2,
        "n": 3, "m": 3,
        # Special keys
        "enter": 1,      # top-right (far right of keyboard)
        "backspace": 1,  # top-right
        ",": 3,          # bottom-right
        ".": 3,          # bottom-right
        "space": 2,      # bottom-left (long key, thumb hits left-center)
        "shift": 2,      # bottom-left
    },
}

ZONE_LABELS = {
    "row": {
        0: "top_row(qwerty)",
        1: "home_row(asdf)",
        2: "bottom_row(zxcv)",
        3: "number_row",
        4: "space_modifier_row",
    },
    "hand": {0: "left_hand", 1: "right_hand"},
    "quadrant": {0: "top_left", 1: "top_right", 2: "bottom_left", 3: "bottom_right"},
}


def map_to_zones(y: np.ndarray, zone_type: str) -> np.ndarray:
    """
    Map key labels to zone labels.

    Args:
        y: array of key labels like ['a', 'b', ...]
        zone_type: 'row', 'hand', or 'quadrant'

    Returns:
        array of integer zone labels
    """
    zone_map = ZONE_MAPS[zone_type]
    return np.array([zone_map.get(k, -1) for k in y])