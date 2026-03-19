"""
Optional onset refinement for stage2_episode.

We reuse the proven Stage 2B-style onset detector architecture locally so
episode detection can stay open-ended while per-episode keypoint extraction
can benefit from a sharper onset model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal as scipy_signal
from scipy.signal import find_peaks


def compute_magnitude(data: np.ndarray) -> np.ndarray:
    accel_mag = np.linalg.norm(data[:, 0:3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)
    return np.concatenate([data[:, :6], accel_mag, gyro_mag], axis=1)


def normalize_signal(data: np.ndarray,
                     mean: Optional[np.ndarray] = None,
                     std: Optional[np.ndarray] = None):
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
        std[std < 1e-8] = 1.0
    return (data - mean) / std, mean, std


def bandpass_filter(data: np.ndarray, low: float, high: float,
                    fs: int, order: int = 4) -> np.ndarray:
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
                   add_magnitude: bool = True):
    out = data.copy()
    if bandpass and sample_rate > 2 * bandpass_high and len(out) > 27:
        out = bandpass_filter(out, bandpass_low, bandpass_high, sample_rate)
    if add_magnitude:
        out = compute_magnitude(out)
    if normalize:
        out, _, _ = normalize_signal(out)
    return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return self.norm(out + residual)


class SingleStageTCN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int = 64,
                 output_channels: int = 1,
                 num_layers: int = 6,
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.input_conv = nn.Conv1d(input_channels, hidden_channels, 1)
        self.input_norm = nn.BatchNorm1d(hidden_channels)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(
                hidden_channels, kernel_size,
                dilation=2 ** i, dropout=dropout
            )
            for i in range(num_layers)
        ])
        self.output_conv = nn.Conv1d(hidden_channels, output_channels, 1)

    def forward(self, x):
        out = F.relu(self.input_norm(self.input_conv(x)))
        for layer in self.layers:
            out = layer(out)
        return self.output_conv(out)


class OnsetRefiner(nn.Module):
    def __init__(self, input_channels=8, hidden_channels=64,
                 num_layers=6, kernel_size=3, dropout=0.3):
        super().__init__()
        self.tcn = SingleStageTCN(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_channels=1,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x):
        return self.tcn(x)

    def predict_probs(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(1)


def pick_onset_peaks(
    probs: np.ndarray,
    expected_onsets: Optional[int] = None,
    min_iki_samples: int = 5,
    base_threshold: float = 0.3,
    fallback_thresholds: Optional[List[float]] = None,
) -> np.ndarray:
    if fallback_thresholds is None:
        fallback_thresholds = [0.2, 0.15, 0.1, 0.05]

    thresholds = [base_threshold] + fallback_thresholds
    best_peaks = np.array([], dtype=np.int64)
    best_heights = np.array([])
    best_diff = float("inf")

    target = expected_onsets if expected_onsets is not None else None

    for thresh in thresholds:
        peaks, props = find_peaks(probs, height=thresh, distance=min_iki_samples)
        if target is None:
            if len(peaks) > 0:
                return np.sort(peaks)
            continue
        diff = abs(len(peaks) - target)
        if diff < best_diff:
            best_diff = diff
            best_peaks = peaks
            best_heights = props["peak_heights"] if len(peaks) > 0 else np.array([])
        if len(peaks) == target:
            return np.sort(peaks)

    if target is None:
        return np.sort(best_peaks)

    if len(best_peaks) > target:
        top_idx = np.argsort(best_heights)[-target:]
        return np.sort(best_peaks[top_idx])
    return np.sort(best_peaks)


@dataclass
class RefineConfig:
    input_channels: int = 8
    hidden_channels: int = 64
    num_layers: int = 6
    kernel_size: int = 3
    dropout: float = 0.3
    min_iki_ms: float = 50.0
    peak_height_threshold: float = 0.3
    fallback_thresholds: List[float] = None
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    use_magnitude: bool = True
    normalize: bool = True

    def __post_init__(self):
        if self.fallback_thresholds is None:
            self.fallback_thresholds = [0.2, 0.15, 0.1, 0.05]


def load_stage2b_refiner(ckpt_path: str, device: torch.device):
    # Compatibility: stage2_rebuild checkpoints pickle Stage2BConfig from a
    # module path that now resolves to stage2_episode/configs/config.py.
    import importlib
    cfg_mod = importlib.import_module("configs.config")
    if not hasattr(cfg_mod, "Stage2BConfig"):
        setattr(cfg_mod, "Stage2BConfig", RefineConfig)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_obj = ckpt.get("config", {})
    valid = set(RefineConfig.__annotations__.keys())
    if hasattr(cfg_obj, "__dict__"):
        cfg_dict = {k: v for k, v in cfg_obj.__dict__.items() if k in valid}
        cfg = RefineConfig(**cfg_dict)
    elif isinstance(cfg_obj, dict):
        cfg = RefineConfig(**{k: v for k, v in cfg_obj.items() if k in valid})
    else:
        cfg = RefineConfig()

    model = OnsetRefiner(
        input_channels=cfg.input_channels,
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def detect_onsets_with_refiner(model: OnsetRefiner,
                               cfg: RefineConfig,
                               imu_episode: np.ndarray,
                               sample_rate: int,
                               expected_onsets: Optional[int] = None,
                               keys_per_second: float = 0.87,
                               min_expected_onsets: int = 4,
                               max_expected_onsets: int = 16) -> np.ndarray:
    if len(imu_episode) < 5:
        return np.array([], dtype=np.int64)
    if expected_onsets is None:
        duration_s = len(imu_episode) / max(sample_rate, 1)
        expected_onsets = int(round(duration_s * keys_per_second))
        expected_onsets = int(np.clip(expected_onsets, min_expected_onsets, max_expected_onsets))
    proc = preprocess_imu(
        imu_episode,
        sample_rate=sample_rate,
        normalize=cfg.normalize,
        bandpass=True,
        bandpass_low=cfg.bandpass_low,
        bandpass_high=cfg.bandpass_high,
        add_magnitude=cfg.use_magnitude,
    )
    x = torch.from_numpy(proc.T).float().unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    with torch.no_grad():
        probs = model.predict_probs(x)[0].detach().cpu().numpy()
    min_iki_samples = int(cfg.min_iki_ms / 1000.0 * sample_rate)
    return pick_onset_peaks(
        probs,
        expected_onsets=expected_onsets,
        min_iki_samples=min_iki_samples,
        base_threshold=cfg.peak_height_threshold,
        fallback_thresholds=cfg.fallback_thresholds,
    )


def predict_onset_probs(model: OnsetRefiner,
                        cfg: RefineConfig,
                        imu_episode: np.ndarray,
                        sample_rate: int) -> np.ndarray:
    if len(imu_episode) < 5:
        return np.zeros(len(imu_episode), dtype=np.float32)
    proc = preprocess_imu(
        imu_episode,
        sample_rate=sample_rate,
        normalize=cfg.normalize,
        bandpass=True,
        bandpass_low=cfg.bandpass_low,
        bandpass_high=cfg.bandpass_high,
        add_magnitude=cfg.use_magnitude,
    )
    x = torch.from_numpy(proc.T).float().unsqueeze(0)
    x = x.to(next(model.parameters()).device)
    with torch.no_grad():
        probs = model.predict_probs(x)[0].detach().cpu().numpy()
    return np.asarray(probs, dtype=np.float32)


def refine_onsets_with_guidance(model: OnsetRefiner,
                                cfg: RefineConfig,
                                imu_episode: np.ndarray,
                                sample_rate: int,
                                anchor_onsets: np.ndarray,
                                keys_per_second: float = 0.87,
                                min_expected_onsets: int = 4,
                                max_expected_onsets: int = 16,
                                search_radius_ms: float = 120.0,
                                min_local_prob: float = 0.18) -> np.ndarray:
    """
    Refine onset anchors locally instead of replacing the whole onset set.

    The energy decoder is good at finding candidate keystroke impulses but can
    over-segment inside long episodes. We use Stage2B only as a local scorer:
    each anchor searches for the best nearby onset probability peak, then we
    keep a count consistent with episode duration.
    """
    anchor_onsets = np.asarray(anchor_onsets, dtype=np.int64)
    if len(imu_episode) < 5:
        return anchor_onsets
    if anchor_onsets.size == 0:
        return detect_onsets_with_refiner(
            model,
            cfg,
            imu_episode,
            sample_rate=sample_rate,
            expected_onsets=None,
            keys_per_second=keys_per_second,
            min_expected_onsets=min_expected_onsets,
            max_expected_onsets=max_expected_onsets,
        )

    probs = predict_onset_probs(model, cfg, imu_episode, sample_rate)
    min_iki_samples = max(2, int(cfg.min_iki_ms / 1000.0 * sample_rate))
    search_radius = max(2, int(search_radius_ms / 1000.0 * sample_rate))

    candidates = []
    for anchor in anchor_onsets:
        left = max(0, int(anchor) - search_radius)
        right = min(len(probs), int(anchor) + search_radius + 1)
        if right - left <= 0:
            continue
        local = probs[left:right]
        peak_rel = int(np.argmax(local))
        peak_idx = left + peak_rel
        peak_prob = float(probs[peak_idx])
        if peak_prob >= min_local_prob:
            candidates.append((peak_idx, peak_prob))
        else:
            # Keep the anchor if the local scorer is uncertain rather than
            # deleting it outright; this helps protect recall.
            safe_idx = int(np.clip(anchor, 0, len(probs) - 1))
            candidates.append((safe_idx, float(probs[safe_idx])))

    if not candidates:
        return anchor_onsets

    # Deduplicate nearby candidates, keeping the stronger local score.
    candidates.sort(key=lambda x: x[0])
    merged: list[tuple[int, float]] = []
    for idx, score in candidates:
        if not merged or idx - merged[-1][0] >= min_iki_samples:
            merged.append((idx, score))
        elif score > merged[-1][1]:
            merged[-1] = (idx, score)

    duration_s = len(imu_episode) / max(sample_rate, 1)
    expected = int(round(duration_s * keys_per_second))
    expected = int(np.clip(expected, min_expected_onsets, max_expected_onsets))
    target_upper = max(expected + 2, int(round(expected * 1.35)))
    target_upper = min(target_upper, len(merged))

    if len(merged) > target_upper:
        keep_idx = np.argsort([score for _, score in merged])[-target_upper:]
        merged = [merged[i] for i in sorted(keep_idx)]

    out = np.array([idx for idx, _score in merged], dtype=np.int64)
    return out
