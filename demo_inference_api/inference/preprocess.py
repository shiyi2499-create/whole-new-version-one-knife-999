from __future__ import annotations

import csv
import io
from typing import Tuple

import numpy as np
from scipy.signal import resample


CSV_FIELDS = (
    'timestamp_ns',
    'accel_x',
    'accel_y',
    'accel_z',
    'gyro_x',
    'gyro_y',
    'gyro_z',
)


def _parse_capture_csv(csv_string: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    reader = csv.DictReader(io.StringIO(csv_string.strip()))
    for row in reader:
        if not row:
            continue
        rows.append([
            int(row['timestamp_ns']),
            float(row['accel_x']),
            float(row['accel_y']),
            float(row['accel_z']),
            float(row['gyro_x']),
            float(row['gyro_y']),
            float(row['gyro_z']),
        ])
    if not rows:
        return np.asarray([], dtype=np.int64), np.zeros((0, 6), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float64)
    return arr[:, 0].astype(np.int64), arr[:, 1:].astype(np.float32)


def csv_to_array(csv_string: str) -> np.ndarray:
    """CSV 字符串 -> (T, 6) numpy array."""
    _, imu = _parse_capture_csv(csv_string)
    return imu


def extract_timestamps(csv_string: str) -> np.ndarray:
    """CSV 字符串 -> (T,) timestamp_ns."""
    ts, _ = _parse_capture_csv(csv_string)
    return ts


def estimate_sample_rate(imu_array: np.ndarray, timestamp_col: np.ndarray) -> float:
    """从 timestamp 列估算采样率。"""
    if len(timestamp_col) < 2 or len(imu_array) < 2:
        return 0.0
    diffs = np.diff(np.asarray(timestamp_col, dtype=np.int64)).astype(np.float64)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    median_dt_ns = float(np.median(diffs))
    return 1e9 / max(median_dt_ns, 1.0)


def resample_to_190hz(imu_array: np.ndarray, original_hz: float) -> np.ndarray:
    """重采样到 190Hz。"""
    imu = np.asarray(imu_array, dtype=np.float32)
    if len(imu) == 0:
        return imu.reshape(0, 6)
    if original_hz <= 1e-6:
        return imu
    if abs(float(original_hz) - 190.0) <= 1.0:
        return imu
    target_len = max(1, int(round(len(imu) * 190.0 / float(original_hz))))
    out = resample(imu, target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)
