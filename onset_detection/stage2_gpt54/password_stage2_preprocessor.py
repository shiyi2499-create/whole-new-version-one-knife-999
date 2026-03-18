"""
Password Stage 2 Preprocessor
=============================

Helpers for constructing patch-level sequence data inside the coarse password
region. This file is deliberately narrow: it does not touch Stage 1 logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PatchConfig:
    patch_width_ms: int = 160
    patch_stride_ms: int = 20
    key_radius_ms: int = 60
    boundary_radius_ms: int = 120
    gap_expand_ms: int = 80


def ns_to_s(x: np.ndarray | float | int):
    return np.asarray(x, dtype=np.float64) / 1e9


def _gaussian_label(distance_ms: np.ndarray, radius_ms: float) -> np.ndarray:
    sigma = max(radius_ms / 2.0, 1.0)
    return np.exp(-0.5 * (distance_ms / sigma) ** 2)


def build_patch_views(sensor: np.ndarray, patch_width_ms: int, patch_stride_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build patch-level features from a sensor array shaped [N, 7]:
    [timestamp_ns, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]

    Returns:
        features: [T, C]
        patch_times_ns: [T]
    """
    if sensor.ndim != 2 or sensor.shape[1] < 7:
        raise ValueError("Expected sensor shape [N, >=7]")

    ts = sensor[:, 0].astype(np.int64)
    sig = sensor[:, 1:7].astype(np.float32)
    if len(sig) < 4:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    start_ns = int(ts[0])
    end_ns = int(ts[-1])
    width_ns = int(patch_width_ms * 1e6)
    stride_ns = int(patch_stride_ms * 1e6)

    accel_mag = np.linalg.norm(sig[:, 0:3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(sig[:, 3:6], axis=1, keepdims=True)
    delta = np.vstack([np.zeros((1, 6), dtype=np.float32), np.diff(sig, axis=0)])
    delta_mag = np.linalg.norm(delta, axis=1, keepdims=True)
    feat_full = np.concatenate([sig, accel_mag, gyro_mag, delta_mag], axis=1)

    patch_feats: list[np.ndarray] = []
    patch_times: list[int] = []
    center = start_ns + width_ns // 2
    while center <= end_ns - width_ns // 2:
        lo = center - width_ns // 2
        hi = center + width_ns // 2
        m = (ts >= lo) & (ts <= hi)
        if m.sum() >= 2:
            win = feat_full[m]
            stats = np.concatenate([
                win.mean(axis=0),
                win.std(axis=0),
                win.max(axis=0),
                win.min(axis=0),
            ], axis=0)
            patch_feats.append(stats.astype(np.float32))
            patch_times.append(int(center))
        center += stride_ns

    if not patch_feats:
        return np.zeros((0, feat_full.shape[1] * 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(patch_feats), np.asarray(patch_times, dtype=np.int64)


def dense_targets_from_groups(
    patch_times_ns: np.ndarray,
    gt_password_groups_ns: list[list[int]],
    cfg: PatchConfig = PatchConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct dense targets from GT password groups.

    key_target[t]      : patch near key center
    boundary_target[t] : patch near inter-password boundary center
    inside_target[t]   : patch lies inside any password span
    """
    T = len(patch_times_ns)
    if T == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z

    patch_ms = ns_to_s(patch_times_ns) * 1000.0
    key_target = np.zeros((T,), dtype=np.float32)
    boundary_target = np.zeros((T,), dtype=np.float32)
    inside_target = np.zeros((T,), dtype=np.float32)

    # key targets
    key_centers_ns = np.asarray([t for group in gt_password_groups_ns for t in group], dtype=np.int64)
    if len(key_centers_ns):
        key_ms = ns_to_s(key_centers_ns) * 1000.0
        for km in key_ms:
            key_target = np.maximum(key_target, _gaussian_label(np.abs(patch_ms - km), cfg.key_radius_ms).astype(np.float32))

    # inside targets from per-password span
    spans: list[tuple[int, int]] = []
    for group in gt_password_groups_ns:
        if not group:
            continue
        start_ns = int(group[0]) - int(cfg.gap_expand_ms * 1e6)
        end_ns = int(group[-1]) + int(cfg.gap_expand_ms * 1e6)
        spans.append((start_ns, end_ns))
        inside_target[(patch_times_ns >= start_ns) & (patch_times_ns <= end_ns)] = 1.0

    # boundary targets between neighboring password groups
    for left, right in zip(gt_password_groups_ns[:-1], gt_password_groups_ns[1:]):
        if not left or not right:
            continue
        center_ns = int((left[-1] + right[0]) // 2)
        center_ms = float(center_ns) / 1e6
        boundary_target = np.maximum(
            boundary_target,
            _gaussian_label(np.abs(patch_ms - center_ms), cfg.boundary_radius_ms).astype(np.float32),
        )

    return key_target, boundary_target, inside_target
