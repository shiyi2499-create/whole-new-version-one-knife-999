"""
Password Segment Detector — Two-Stage + Classifier
====================================================

Full Path B pipeline:
  mixed2 stream
    → Stage 1: binary segment classifier → coarse password region
    → Stage 2: onset detector + IKI rhythm → refined boundary + per-password onset groups
    → Stage 3: password classifier → char top-k / sequence_topN / CER
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_ONSET_DIR = os.path.dirname(HERE)
if PARENT_ONSET_DIR not in sys.path:
    sys.path.insert(0, PARENT_ONSET_DIR)

from onset_model import build_onset_model
from onset_preprocessor import (
    DEFAULT_TARGET_RATE_HZ,
    DEFAULT_WINDOW_MS,
    DEFAULT_STRIDE_MS,
    DEFAULT_LABEL_RADIUS_MS,
    load_sensor_csv,
    load_events_csv,
    load_activity_log,
    resample_window,
    window_samples,
    get_password_segments_from_activity_log,
    refine_password_segments_with_events,
)
from onset_utils import (
    Episode,
    detect_peaks,
    nms_1d,
    match_episodes,
)
from password_segment_preprocessor import (
    SEGMENT_WINDOW_MS,
    SEGMENT_STRIDE_MS,
    N_CHANNELS,
    _iterate_window_chunks,
    discover_sessions,
)


# ── Lazy imports for password classifier (set up via _setup_imports) ──

_classifier_imported = False


def _remap_legacy_onset_state_dict(state_dict: dict) -> dict:
    """
    Support older onset checkpoints whose encoder was stored under
    ``features.*`` instead of ``encoder.net.*``.
    """
    if any(k.startswith("encoder.net.") for k in state_dict):
        return state_dict
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("features."):
            remapped["encoder.net." + key[len("features."):]] = value
        else:
            remapped[key] = value
    return remapped


def _setup_imports(project_root: str = ""):
    global _classifier_imported
    if _classifier_imported:
        return
    root = os.path.abspath(project_root) if project_root else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (root, os.path.join(root, "phase3_password_inception")):
        if p not in sys.path:
            sys.path.insert(0, p)
    _classifier_imported = True


def _load_press_rows(events_path: str) -> list[dict]:
    rows = []
    if not os.path.exists(events_path):
        return rows
    with open(events_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event_type") != "press":
                continue
            try:
                ts = int(row["timestamp_ns"])
            except Exception:
                continue
            rows.append({"timestamp_ns": ts, "key": (row.get("key") or "").lower()})
    return rows


def _load_supported_press_timestamps(events_path: str) -> np.ndarray:
    from run_password_closure_inception import supported_key

    keep = []
    for row in _load_press_rows(events_path):
        key = row["key"]
        if key in {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
                   "left", "right", "up", "down", "delete", "enter", "return",
                   "space", "backspace"}:
            continue
        if not supported_key(key):
            continue
        keep.append(row["timestamp_ns"])
    return np.asarray(keep, dtype=np.int64)


def _extract_gt_password_groups(events_path: str, gt_refined_segs: list[dict]) -> list[list[int]]:
    """
    Split GT password typing into per-password groups using Enter as delimiter.

    We keep only classifier-supported character keys inside each group, and we
    drop spaces/backspaces/modifiers so the GT baseline matches the main
    password-classifier evaluation path more closely.
    """
    from run_password_closure_inception import supported_key

    rows = _load_press_rows(events_path)
    if not rows:
        return []

    all_groups: list[list[int]] = []
    for seg in gt_refined_segs:
        seg_rows = [
            r for r in rows
            if int(seg["start_time_ns"]) <= r["timestamp_ns"] <= int(seg["end_time_ns"])
        ]
        cur: list[int] = []
        for row in seg_rows:
            key = row["key"]
            ts = row["timestamp_ns"]
            if key in {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
                       "left", "right", "up", "down", "delete"}:
                continue
            if key in {"enter", "return"}:
                if cur:
                    all_groups.append(cur.copy())
                    cur = []
                continue
            if key in {"space", "backspace"}:
                continue
            if not supported_key(key):
                continue
            cur.append(ts)
        if cur:
            all_groups.append(cur.copy())
    return all_groups


# ── Stage 1: Binary Segment Classifier ──────────────────────

@dataclass
class CoarseRegion:
    start_s: float
    end_s: float
    mean_prob: float = 0.0
    max_prob: float = 0.0

    @property
    def duration_s(self):
        return max(0.0, self.end_s - self.start_s)


def _select_primary_coarse_regions(regions, max_regions=1):
    if not regions:
        return []
    ranked = sorted(
        regions,
        key=lambda r: (r.duration_s * (0.7 * r.mean_prob + 0.3 * r.max_prob)),
        reverse=True,
    )
    return sorted(ranked[:max_regions], key=lambda r: r.start_s)


def _allocate_password_counts(regions, total_passwords):
    """
    Distribute an expected password count across coarse regions.

    Each kept region gets at least one password, and the remainder is assigned
    proportionally to a region salience score based on duration and probability.
    """
    if not regions or total_passwords <= 0:
        return []
    if len(regions) == 1:
        return [total_passwords]

    n_regions = len(regions)
    if total_passwords <= n_regions:
        order = sorted(
            range(n_regions),
            key=lambda i: regions[i].duration_s * (0.7 * regions[i].mean_prob + 0.3 * regions[i].max_prob),
            reverse=True,
        )
        counts = [0] * n_regions
        for i in order[:total_passwords]:
            counts[i] = 1
        return counts

    base = [1] * n_regions
    remaining = total_passwords - n_regions
    weights = np.asarray(
        [
            max(1e-6, r.duration_s * (0.7 * r.mean_prob + 0.3 * r.max_prob))
            for r in regions
        ],
        dtype=np.float64,
    )
    raw = remaining * (weights / weights.sum())
    extra = np.floor(raw).astype(int)
    counts = [base[i] + int(extra[i]) for i in range(n_regions)]
    assigned = sum(counts)
    if assigned < total_passwords:
        frac_order = np.argsort(-(raw - np.floor(raw)))
        for idx in frac_order[: total_passwords - assigned]:
            counts[int(idx)] += 1
    return counts


def load_segment_detector(checkpoint_path, scaler_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_onset_model(
        ckpt.get("model_name", "cnn"),
        n_channels=ckpt.get("n_channels", 6),
        n_classes=int(ckpt.get("n_classes", 1)),
        task=ckpt.get("task", "password_segment"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    scaler = np.load(scaler_path)
    meta = {
        "window_ms": int(ckpt.get("window_ms", SEGMENT_WINDOW_MS)),
        "stride_ms": int(ckpt.get("stride_ms", SEGMENT_STRIDE_MS)),
        "target_rate_hz": int(ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ)),
        "n_channels": int(ckpt.get("n_channels", 6)),
    }
    return model, scaler["means"].astype(np.float32), np.maximum(scaler["stds"].astype(np.float32), 1e-10), meta


def run_binary_inference(model, sensor, means, stds, window_ms, stride_ms,
                         target_rate_hz, device, batch_size=256):
    windows, times_s = [], []
    for centre, win in _iterate_window_chunks(sensor, window_ms, stride_ms, target_rate_hz):
        windows.append((win - means) / stds)
        times_s.append(centre / 1e9)
    if not windows:
        return np.array([]), np.array([])
    X = np.stack(windows).astype(np.float32)
    ts = np.asarray(times_s)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.from_numpy(X[i:i+batch_size]).to(device)
            logits = model(b)
            p = torch.sigmoid(logits.squeeze(-1)) if logits.shape[-1] == 1 else torch.softmax(logits, -1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.concatenate(probs), ts


def extract_coarse_regions(probs, times_s, threshold=0.5, merge_gap_s=1.5,
                           min_duration_s=2.0, margin_s=1.0):
    if len(probs) == 0:
        return []
    active = probs >= threshold
    # Bridge short gaps
    in_act, last_end = False, -1
    for i in range(len(active)):
        if active[i]:
            if not in_act and last_end >= 0 and (times_s[i] - times_s[last_end]) <= merge_gap_s:
                active[last_end:i+1] = True
            in_act = True
        else:
            if in_act: last_end = i - 1
            in_act = False
    # Contiguous regions
    regions = []
    start = None
    for i in range(len(active)):
        if active[i] and start is None:
            start = i
        elif not active[i] and start is not None:
            rp = probs[start:i]
            regions.append(CoarseRegion(times_s[start] - margin_s, times_s[i-1] + margin_s,
                                        float(np.mean(rp)), float(np.max(rp))))
            start = None
    if start is not None:
        rp = probs[start:]
        regions.append(CoarseRegion(times_s[start] - margin_s, times_s[-1] + margin_s,
                                    float(np.mean(rp)), float(np.max(rp))))
    return [r for r in regions if r.duration_s >= min_duration_s]


# ── Stage 2v2: Energy-Valley Segmentation ─────────────────────
#
# New approach: instead of detecting all onsets first and then trying to
# group them (bottom-up), we first find the inter-password gaps by looking
# at energy valleys in the raw IMU signal (top-down), then detect onsets
# within each resulting sub-segment.
#
# Why this works better:
#   - Inter-password gaps (user pausing to read next prompt, press Enter,
#     wait for feedback) produce clear low-energy valleys (~0.5-2s of calm)
#   - Finding 4 split points is much easier than grouping ~40 onsets
#   - The "exactly N passwords" constraint is encoded directly
#   - Per-segment onset detection is easier (expect ~8 onsets, can use
#     adaptive threshold per segment)


def compute_energy_envelope(sensor: np.ndarray, window_ms: int = 80,
                            stride_ms: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RMS energy envelope of accelerometer magnitude.

    Returns (energy, center_times_ns) arrays.
    """
    ts_ns = sensor[:, 0]
    # Accelerometer magnitude (columns 1,2,3)
    accel = sensor[:, 1:4]
    mag = np.sqrt(np.sum(accel ** 2, axis=1))

    win_ns = int(window_ms * 1_000_000)
    stride_ns = int(stride_ms * 1_000_000)
    half = win_ns // 2

    energies = []
    centers = []
    t_start = int(ts_ns[0]) + half
    t_end = int(ts_ns[-1]) - half

    centre = t_start
    while centre <= t_end:
        i0 = np.searchsorted(ts_ns, centre - half, side="left")
        i1 = np.searchsorted(ts_ns, centre + half, side="right")
        if i1 - i0 >= 2:
            rms = float(np.sqrt(np.mean(mag[i0:i1] ** 2)))
            energies.append(rms)
            centers.append(centre)
        centre += stride_ns

    return np.asarray(energies, dtype=np.float64), np.asarray(centers, dtype=np.float64)


def find_energy_valleys(energy: np.ndarray, times_ns: np.ndarray,
                        min_valley_width_ms: int = 200,
                        smooth_n: int = 7) -> list[dict]:
    """
    Find low-energy valleys in the energy envelope.

    A valley is a contiguous region where energy is below the adaptive
    threshold (local energy statistics).  Returns valley candidates sorted
    by depth (deepest first).
    """
    if len(energy) < 10:
        return []

    # Smooth energy
    kernel = np.ones(smooth_n, dtype=np.float64) / smooth_n
    e_smooth = np.convolve(energy, kernel, mode="same")

    # Adaptive threshold: valleys are regions below median energy
    median_e = float(np.median(e_smooth))
    # Use a threshold between the 25th percentile and the median
    q25 = float(np.percentile(e_smooth, 25))
    threshold = 0.5 * (q25 + median_e)

    below = e_smooth < threshold
    min_width_ns = min_valley_width_ms * 1_000_000

    # Find contiguous below-threshold regions
    valleys = []
    in_valley = False
    v_start = 0
    for i in range(len(below)):
        if below[i] and not in_valley:
            v_start = i
            in_valley = True
        elif not below[i] and in_valley:
            width_ns = times_ns[i - 1] - times_ns[v_start]
            if width_ns >= min_width_ns:
                center_idx = v_start + np.argmin(e_smooth[v_start:i])
                depth = median_e - float(e_smooth[center_idx])
                valleys.append({
                    "start_idx": v_start,
                    "end_idx": i - 1,
                    "center_idx": int(center_idx),
                    "center_ns": float(times_ns[center_idx]),
                    "depth": depth,
                    "width_ns": float(width_ns),
                    "min_energy": float(e_smooth[center_idx]),
                })
            in_valley = False
    if in_valley:
        width_ns = times_ns[-1] - times_ns[v_start]
        if width_ns >= min_width_ns:
            center_idx = v_start + np.argmin(e_smooth[v_start:])
            depth = median_e - float(e_smooth[center_idx])
            valleys.append({
                "start_idx": v_start,
                "end_idx": len(below) - 1,
                "center_idx": int(center_idx),
                "center_ns": float(times_ns[center_idx]),
                "depth": depth,
                "width_ns": float(width_ns),
                "min_energy": float(e_smooth[center_idx]),
            })

    return sorted(valleys, key=lambda v: -v["depth"])


def select_split_valleys_dp(valleys: list[dict], energy: np.ndarray,
                            times_ns: np.ndarray, n_splits: int,
                            n_passwords: int, expected_len: int = 8,
                            min_segment_duration_ms: int = 1500) -> list[dict]:
    """
    Select the best n_splits valleys to divide the coarse region into
    n_passwords segments.  Uses a scoring function that rewards:
      - deep valleys (strong gaps)
      - roughly equal segment durations
      - segment durations consistent with expected_len keystrokes

    Falls back to greedy top-N if DP is infeasible.
    """
    if n_splits <= 0 or not valleys:
        return []

    min_seg_ns = min_segment_duration_ms * 1_000_000
    region_start_ns = float(times_ns[0])
    region_end_ns = float(times_ns[-1])
    region_dur_ns = region_end_ns - region_start_ns

    # Expected segment duration (rough): total / n_passwords
    expected_seg_ns = region_dur_ns / max(n_passwords, 1)

    # Filter valleys that are not too close to region boundaries
    margin_ns = min_seg_ns * 0.5
    valid_valleys = [
        v for v in valleys
        if (v["center_ns"] - region_start_ns) > margin_ns
        and (region_end_ns - v["center_ns"]) > margin_ns
    ]

    if len(valid_valleys) < n_splits:
        # Not enough valleys; return what we have
        return sorted(valid_valleys[:n_splits], key=lambda v: v["center_ns"])

    # Greedy approach for small search space; DP for larger
    if len(valid_valleys) <= 15:
        # Enumerate all combinations
        from itertools import combinations
        best_score = -1e18
        best_combo = None
        for combo in combinations(range(len(valid_valleys)), n_splits):
            chosen = [valid_valleys[i] for i in combo]
            chosen_sorted = sorted(chosen, key=lambda v: v["center_ns"])
            score = _score_valley_combo(
                chosen_sorted, region_start_ns, region_end_ns,
                expected_seg_ns, min_seg_ns)
            if score > best_score:
                best_score = score
                best_combo = chosen_sorted
        return best_combo if best_combo else []
    else:
        # Too many candidates; take top by depth, then optimize
        top_valleys = sorted(valid_valleys, key=lambda v: -v["depth"])[:n_splits * 3]
        from itertools import combinations
        best_score = -1e18
        best_combo = None
        for combo in combinations(range(len(top_valleys)), n_splits):
            chosen = [top_valleys[i] for i in combo]
            chosen_sorted = sorted(chosen, key=lambda v: v["center_ns"])
            score = _score_valley_combo(
                chosen_sorted, region_start_ns, region_end_ns,
                expected_seg_ns, min_seg_ns)
            if score > best_score:
                best_score = score
                best_combo = chosen_sorted
        return best_combo if best_combo else []


def _score_valley_combo(valleys_sorted, region_start_ns, region_end_ns,
                        expected_seg_ns, min_seg_ns):
    """Score a combination of split valleys."""
    boundaries = [region_start_ns] + [v["center_ns"] for v in valleys_sorted] + [region_end_ns]
    n_segs = len(boundaries) - 1
    durations = [boundaries[i + 1] - boundaries[i] for i in range(n_segs)]

    # Penalty: any segment too short
    for d in durations:
        if d < min_seg_ns:
            return -1e18

    # Reward: valley depth (sum)
    depth_score = sum(v["depth"] for v in valleys_sorted)

    # Reward: segment duration uniformity (penalize variance)
    mean_dur = sum(durations) / len(durations)
    dur_variance = sum((d - mean_dur) ** 2 for d in durations) / len(durations)
    dur_cv = math.sqrt(dur_variance) / max(mean_dur, 1e-9)
    uniformity_score = math.exp(-dur_cv)

    # Reward: segments close to expected duration
    dur_match = sum(math.exp(-abs(d - expected_seg_ns) / max(expected_seg_ns, 1e-9))
                    for d in durations) / n_segs

    score = 1.0 * depth_score + 2.0 * uniformity_score + 1.5 * dur_match
    return score


def segment_by_energy_valleys(
    sensor: np.ndarray,
    region: "CoarseRegion",
    n_passwords: int = 5,
    expected_password_len: int = 8,
    energy_window_ms: int = 80,
    energy_stride_ms: int = 10,
    min_valley_width_ms: int = 200,
    min_segment_duration_ms: int = 1500,
) -> list[tuple[float, float]]:
    """
    Split a coarse password region into n_passwords sub-segments using
    energy valley detection.

    Returns list of (start_s, end_s) tuples for each password segment.
    """
    ts_ns = sensor[:, 0]
    mask = (ts_ns >= region.start_s * 1e9) & (ts_ns <= region.end_s * 1e9)
    if mask.sum() < 20:
        return [(region.start_s, region.end_s)]

    region_sensor = sensor[mask]
    energy, energy_times = compute_energy_envelope(
        region_sensor, window_ms=energy_window_ms, stride_ms=energy_stride_ms)

    if len(energy) < 10:
        return [(region.start_s, region.end_s)]

    valleys = find_energy_valleys(
        energy, energy_times, min_valley_width_ms=min_valley_width_ms)

    n_splits = n_passwords - 1
    if n_splits <= 0:
        return [(region.start_s, region.end_s)]

    chosen = select_split_valleys_dp(
        valleys, energy, energy_times, n_splits=n_splits,
        n_passwords=n_passwords, expected_len=expected_password_len,
        min_segment_duration_ms=min_segment_duration_ms)

    if not chosen:
        # Fallback: equal-duration split
        dur = region.end_s - region.start_s
        seg_dur = dur / n_passwords
        return [(region.start_s + i * seg_dur, region.start_s + (i + 1) * seg_dur)
                for i in range(n_passwords)]

    # Build segments from split points
    split_times_s = [v["center_ns"] / 1e9 for v in chosen]
    boundaries_s = [region.start_s] + split_times_s + [region.end_s]
    segments = []
    for i in range(len(boundaries_s) - 1):
        segments.append((boundaries_s[i], boundaries_s[i + 1]))
    return segments


def detect_onsets_in_segment(
    onset_model, sensor, onset_means, onset_stds,
    seg_start_s, seg_end_s,
    window_ms, stride_ms, target_rate_hz,
    device, expected_n_onsets=8,
    base_threshold=0.3, nms_radius_s=0.10,
    smooth_n=3, batch_size=256,
) -> list[dict]:
    """
    Detect onsets within a single password sub-segment.

    Uses a lower base threshold and then selects the top-N by probability,
    where N is guided by expected_n_onsets.
    """
    region = CoarseRegion(start_s=seg_start_s, end_s=seg_end_s)
    min_keep = max(5, expected_n_onsets - 2)
    max_keep = expected_n_onsets + 4

    def _collect_candidates():
        passes = [
            {"threshold": base_threshold, "nms_radius_s": nms_radius_s, "adaptive_quantile": 0.60, "adaptive_floor_scale": 0.92},
            {"threshold": max(0.12, base_threshold - 0.08), "nms_radius_s": max(0.06, nms_radius_s - 0.02), "adaptive_quantile": 0.50, "adaptive_floor_scale": 0.88},
            {"threshold": max(0.08, base_threshold - 0.14), "nms_radius_s": max(0.05, nms_radius_s - 0.03), "adaptive_quantile": None, "adaptive_floor_scale": 1.0},
        ]
        merged = {}
        for cfg in passes:
            peaks = detect_onsets_in_region(
                onset_model, sensor, onset_means, onset_stds, region,
                window_ms, stride_ms, target_rate_hz,
                device,
                threshold=cfg["threshold"],
                nms_radius_s=cfg["nms_radius_s"],
                smooth_n=smooth_n,
                max_peaks=0,
                batch_size=batch_size,
                adaptive_quantile=cfg["adaptive_quantile"],
                adaptive_floor_scale=cfg["adaptive_floor_scale"],
            )
            for pk in peaks:
                key = round(float(pk["time_s"]), 3)
                prev = merged.get(key)
                if prev is None or float(pk["prob"]) > float(prev["prob"]):
                    merged[key] = pk
            if len(merged) >= min_keep:
                break
        return sorted(merged.values(), key=lambda pk: pk["time_s"])

    peaks = _collect_candidates()
    if not peaks:
        return []
    if len(peaks) <= max_keep:
        return peaks

    ranked = sorted(peaks, key=lambda pk: -pk["prob"])
    kept = ranked[:max_keep]
    if len(kept) < min_keep:
        kept = ranked[:min_keep]
    return sorted(kept, key=lambda pk: pk["time_s"])


def run_stage2_energy_valley(
    sensor, coarse_regions,
    onset_model, onset_means, onset_stds, onset_meta,
    device,
    n_passwords=5,
    expected_password_len=8,
    onset_base_threshold=0.3,
    onset_nms_radius_s=0.10,
    onset_smooth_n=3,
    energy_window_ms=80,
    energy_stride_ms=10,
    min_valley_width_ms=200,
    min_segment_duration_ms=1500,
) -> tuple[list[list[float]], list[dict]]:
    """
    Stage 2v2: Energy-Valley Segmentation + Per-Segment Onset Detection.

    1. For each coarse region, compute energy envelope and find valleys
    2. Use valleys to split into n_passwords sub-segments
    3. Within each sub-segment, detect onsets with adaptive thresholding

    Returns:
        password_groups_s: list of onset time lists (in seconds), one per password
        debug: debug info dict
    """
    target_rate_hz = onset_meta["target_rate_hz"]
    window_ms = onset_meta["window_ms"]
    stride_ms = onset_meta["stride_ms"]

    all_segments = []
    debug_info = {"coarse_regions": [], "valleys": [], "segments": []}
    region_password_counts = _allocate_password_counts(coarse_regions, n_passwords)

    for region, region_n_passwords in zip(coarse_regions, region_password_counts):
        if region_n_passwords <= 0:
            continue
        segments = segment_by_energy_valleys(
            sensor, region,
            n_passwords=region_n_passwords,
            expected_password_len=expected_password_len,
            energy_window_ms=energy_window_ms,
            energy_stride_ms=energy_stride_ms,
            min_valley_width_ms=min_valley_width_ms,
            min_segment_duration_ms=min_segment_duration_ms,
        )
        all_segments.extend(segments)
        debug_info["coarse_regions"].append({
            "start_s": region.start_s,
            "end_s": region.end_s,
            "duration_s": region.duration_s,
            "allocated_passwords": int(region_n_passwords),
            "n_segments": len(segments),
        })
        for seg_start, seg_end in segments:
            debug_info["segments"].append({
                "start_s": seg_start,
                "end_s": seg_end,
                "duration_s": seg_end - seg_start,
            })

    # Detect onsets within each segment
    password_groups_s = []
    for seg_start, seg_end in all_segments:
        peaks = detect_onsets_in_segment(
            onset_model, sensor, onset_means, onset_stds,
            seg_start, seg_end,
            window_ms, stride_ms, target_rate_hz,
            device,
            expected_n_onsets=expected_password_len,
            base_threshold=onset_base_threshold,
            nms_radius_s=onset_nms_radius_s,
            smooth_n=onset_smooth_n,
        )
        onset_times = [pk["time_s"] for pk in peaks]
        password_groups_s.append(onset_times)
        debug_info["segments"][-len(all_segments) + len(password_groups_s) - 1]["n_onsets"] = len(peaks)
        debug_info["segments"][-len(all_segments) + len(password_groups_s) - 1]["onset_probs"] = [
            float(pk["prob"]) for pk in peaks[:12]
        ]

    return password_groups_s, debug_info


# ── Stage 2 (original): Onset detection + IKI rhythm ────────────────────

def load_onset_detector(checkpoint_path, scaler_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_onset_model(
        ckpt.get("model_name", "cnn"),
        n_channels=ckpt.get("n_channels", 6),
        n_classes=int(ckpt.get("n_classes", 1)),
        task=ckpt.get("task", "onset"),
    )
    model_state = _remap_legacy_onset_state_dict(ckpt["model_state"])
    model.load_state_dict(model_state)
    model.to(device).eval()
    scaler = np.load(scaler_path)
    meta = {"window_ms": int(ckpt.get("window_ms", DEFAULT_WINDOW_MS)),
            "stride_ms": int(ckpt.get("stride_ms", DEFAULT_STRIDE_MS)),
            "target_rate_hz": int(ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ))}
    return model, scaler["means"].astype(np.float32), np.maximum(scaler["stds"].astype(np.float32), 1e-10), meta


def detect_onsets_in_region(model, sensor, means, stds, region,
                            window_ms, stride_ms, target_rate_hz,
                            device, threshold=0.5, nms_radius_s=0.12,
                            smooth_n=3, max_peaks=0, batch_size=256,
                            adaptive_quantile=0.70, adaptive_floor_scale=1.0):
    ts_ns = sensor[:, 0]
    mask = (ts_ns >= region.start_s * 1e9) & (ts_ns <= region.end_s * 1e9)
    if mask.sum() < 10:
        return []
    rsensor = sensor[mask]
    windows, times = [], []
    for c, w in _iterate_window_chunks(rsensor, window_ms, stride_ms, target_rate_hz):
        windows.append((w - means) / stds)
        times.append(c / 1e9)
    if not windows:
        return []
    X = np.stack(windows).astype(np.float32)
    ts = np.asarray(times)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.from_numpy(X[i:i+batch_size]).to(device)
            probs.append(torch.sigmoid(model(b).squeeze(-1)).cpu().numpy())
    p = np.concatenate(probs)
    raw_peaks = detect_peaks(p, ts, threshold=threshold, smooth_n=smooth_n)
    peaks = nms_1d(raw_peaks, radius_s=nms_radius_s)
    if peaks:
        peak_probs = np.asarray([pk["prob"] for pk in peaks], dtype=np.float64)
        if adaptive_quantile is not None:
            adaptive_floor = max(
                float(threshold),
                float(np.quantile(peak_probs, adaptive_quantile)) * float(adaptive_floor_scale),
            )
        else:
            adaptive_floor = float(threshold)
        peaks = [pk for pk in peaks if pk["prob"] >= adaptive_floor]
    if max_peaks and len(peaks) > max_peaks:
        peaks = sorted(peaks, key=lambda pk: -pk["prob"])[:max_peaks]
        peaks = sorted(peaks, key=lambda pk: pk["time_s"])
    return peaks


def find_password_rhythm_clusters(onset_times, expected_len=8, iki_min_s=0.15,
                                  iki_max_s=2.0, iki_cv_max=0.8, group_gap_s=2.5,
                                  min_onsets_per_group=4):
    if len(onset_times) < min_onsets_per_group:
        return []
    ot = sorted(onset_times)
    groups, cur = [], [ot[0]]
    for i in range(1, len(ot)):
        if ot[i] - ot[i-1] > group_gap_s:
            groups.append(cur); cur = [ot[i]]
        else:
            cur.append(ot[i])
    groups.append(cur)

    clusters = []
    for g in groups:
        if len(g) < min_onsets_per_group:
            continue
        ikis = np.diff(g)
        valid = ikis[(ikis >= iki_min_s) & (ikis <= iki_max_s)]
        if len(valid) < max(2, min_onsets_per_group - 2):
            continue
        med = float(np.median(valid))
        cv = float(np.std(valid) / max(np.mean(valid), 1e-6))
        if cv <= iki_cv_max:
            clusters.append({"onsets": g, "start_s": g[0], "end_s": g[-1],
                             "n_onsets": len(g), "median_iki": med, "iki_cv": cv})
    return clusters


def split_cluster_into_passwords(cluster, enter_gap_min_s=0.8):
    onsets = cluster["onsets"]
    med_iki = cluster["median_iki"]
    if len(onsets) < 3:
        return [onsets]
    ikis = np.diff(onsets)
    subs, cur = [], [onsets[0]]
    for i in range(len(ikis)):
        if ikis[i] >= enter_gap_min_s and ikis[i] > med_iki * 2.5:
            if len(cur) >= 3:
                subs.append(cur)
            cur = [onsets[i+1]]
        else:
            cur.append(onsets[i+1])
    if len(cur) >= 3:
        subs.append(cur)
    return subs


def _group_score(onset_times_s, onset_score_map, expected_len=8):
    if not onset_times_s:
        return -1.0, {}
    n = len(onset_times_s)
    probs = np.asarray([onset_score_map.get(round(t, 6), 0.0) for t in onset_times_s], dtype=np.float64)
    mean_prob = float(np.mean(probs)) if len(probs) else 0.0
    ikis = np.diff(onset_times_s) if len(onset_times_s) >= 2 else np.array([], dtype=np.float64)
    if len(ikis):
        med_iki = float(np.median(ikis))
        cv = float(np.std(ikis) / max(np.mean(ikis), 1e-6))
    else:
        med_iki = 0.0
        cv = 1.5

    len_penalty = abs(n - expected_len)
    len_term = math.exp(-len_penalty / 2.5)
    prob_term = mean_prob
    iki_term = 1.0 if 0.15 <= med_iki <= 1.2 else 0.55
    cv_term = math.exp(-min(cv, 2.0))
    overlong_penalty = 0.65 if n > expected_len + 4 else 1.0

    score = (0.45 * prob_term + 0.30 * len_term + 0.15 * iki_term + 0.10 * cv_term) * overlong_penalty
    meta = {
        "n_onsets": n,
        "mean_prob": mean_prob,
        "median_iki": med_iki,
        "iki_cv": cv,
        "score": score,
    }
    return float(score), meta


def _best_password_subsequence(onset_times_s, onset_score_map, expected_len=8, min_len=5, max_len=12):
    if len(onset_times_s) <= max_len:
        score, meta = _group_score(onset_times_s, onset_score_map, expected_len=expected_len)
        return onset_times_s, meta

    best_seq = []
    best_meta = {"score": -1.0}
    n = len(onset_times_s)
    for i in range(n):
        for j in range(i + min_len, min(n, i + max_len) + 1):
            seq = onset_times_s[i:j]
            score, meta = _group_score(seq, onset_score_map, expected_len=expected_len)
            if score > best_meta["score"]:
                best_seq = seq
                best_meta = meta
    return best_seq, best_meta


def _select_final_groups(group_candidates, max_groups=0):
    if not group_candidates:
        return []
    if max_groups and len(group_candidates) > max_groups:
        chosen = sorted(group_candidates, key=lambda g: (-g["score"], g["start_s"]))[:max_groups]
    else:
        chosen = group_candidates
    return sorted(chosen, key=lambda g: g["start_s"])


def _split_onsets_by_largest_gaps(onset_times_s, n_groups, min_group_len=4):
    if n_groups <= 1 or len(onset_times_s) < n_groups * min_group_len:
        return []
    gaps = np.diff(onset_times_s)
    ranked = sorted([(float(g), int(i)) for i, g in enumerate(gaps)], reverse=True)
    chosen_split_idx = []

    def _valid_groups(split_idx):
        split_idx = sorted(split_idx)
        groups = []
        start = 0
        for si in split_idx:
            grp = onset_times_s[start:si + 1]
            if len(grp) < min_group_len:
                return []
            groups.append(grp)
            start = si + 1
        last = onset_times_s[start:]
        if len(last) < min_group_len:
            return []
        groups.append(last)
        return groups

    for _gap, idx in ranked:
        trial = chosen_split_idx + [idx]
        groups = _valid_groups(trial)
        if groups:
            chosen_split_idx = trial
        if len(chosen_split_idx) >= n_groups - 1:
            break

    groups = _valid_groups(chosen_split_idx)
    return groups if len(groups) == n_groups else []


# ── Stage 3: Classifier + Metrics ────────────────────────────

def cut_classifier_windows(sensor, onset_times_ns, pre_ms=100, post_ms=200,
                           target_rate_hz=DEFAULT_TARGET_RATE_HZ):
    ts = sensor[:, 0]
    vals = sensor[:, 1:]
    tgt = window_samples(pre_ms + post_ms, target_rate_hz)
    out = []
    for t_ns in onset_times_ns:
        i0 = np.searchsorted(ts, t_ns - pre_ms * 1e6, side="left")
        i1 = np.searchsorted(ts, t_ns + post_ms * 1e6, side="right")
        if i1 - i0 < 4:
            out.append(None)
        else:
            out.append(resample_window(vals[i0:i1], tgt))
    return out


def classify_windows(windows, classifier, means, stds, device):
    valid_idx = [i for i, w in enumerate(windows) if w is not None]
    if not valid_idx:
        return [None] * len(windows)
    X = np.stack([windows[i] for i in valid_idx]).astype(np.float32)
    for ch in range(X.shape[2]):
        X[:, :, ch] = (X[:, :, ch] - means[ch]) / (stds[ch] + 1e-10)
    classifier.eval()
    with torch.no_grad():
        probs = torch.softmax(classifier(torch.from_numpy(X).to(device)), dim=1).cpu().numpy()
    out = [None] * len(windows)
    for bi, oi in enumerate(valid_idx):
        out[oi] = probs[bi]
    return out


def levenshtein(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(0 if ca==cb else 1)))
        prev = cur
    return prev[-1]


SEQ_HIT_CUTOFFS = (10, 50, 100)


def score_one_password(onset_times_ns, ref, sensor, classifier, cls_classes,
                       cls_means, cls_stds, device, target_rate_hz):
    """Score a single predicted password group against its reference string."""
    from run_password_closure_inception import topk_strings_from_prob_vectors

    windows = cut_classifier_windows(sensor, onset_times_ns, target_rate_hz=target_rate_hz)
    prob_vecs = classify_windows(windows, classifier, cls_means, cls_stds, device)
    valid = [p for p in prob_vecs if p is not None]

    result = {"n_onsets": len(onset_times_ns), "n_valid_windows": len(valid),
              "reference": ref, "hypothesis": "", "cer": 1.0,
              "char_top1": 0.0, "char_top3": 0.0, "char_top5": 0.0}
    for cutoff in SEQ_HIT_CUTOFFS:
        result[f"seq_top{cutoff}"] = 0

    if not valid:
        return result

    hyp = "".join(cls_classes[int(np.argmax(p))] for p in valid)
    result["hypothesis"] = hyp
    result["cer"] = levenshtein(ref, hyp) / max(len(ref), 1)

    topk_hits = {1: 0, 3: 0, 5: 0}
    for i, ref_ch in enumerate(ref):
        if i >= len(valid): break
        ranked = [cls_classes[r] for r in np.argsort(-valid[i])]
        for k in (1, 3, 5):
            if ref_ch in ranked[:k]:
                topk_hits[k] += 1
    n = max(len(ref), 1)
    result["char_top1"] = topk_hits[1] / n
    result["char_top3"] = topk_hits[3] / n
    result["char_top5"] = topk_hits[5] / n

    try:
        cands = topk_strings_from_prob_vectors(
            np.stack(valid), cls_classes, branch_topk=5, beam_width=max(SEQ_HIT_CUTOFFS))
        cand_strs = [c["candidate"] for c in cands]
        for cutoff in SEQ_HIT_CUTOFFS:
            result[f"seq_top{cutoff}"] = 1 if ref in cand_strs[:cutoff] else 0
    except Exception:
        pass

    return result


# ── Full pipeline ────────────────────────────────────────────

def run_full_pipeline(
    sensor, sess_prefix,
    seg_model, seg_means, seg_stds, seg_meta,
    onset_model, onset_means, onset_stds, onset_meta,
    classifier, cls_classes, cls_means, cls_stds,
    device,
    gt_passwords, gt_refined_segs, gt_password_groups_ns,
    expected_password_count=5,
    segment_threshold=0.5, onset_threshold=0.5,
    expected_password_len=8,
    max_coarse_regions=1,
    onset_nms_radius_s=0.12,
    onset_smooth_n=3,
    max_onsets_per_region=56,
    group_quality_threshold=0.48,
    stage2_method="energy_valley",
    energy_window_ms=80,
    energy_stride_ms=10,
    min_valley_width_ms=200,
    min_segment_duration_ms=1500,
    onset_base_threshold=0.3,
    ctc_model=None,
    ctc_means=None,
    ctc_stds=None,
    ctc_beam_width=20,
):
    """
    Full pipeline: Stage 1 → Stage 2 → Stage 3.

    stage2_method:
        "dense_ctc"      – MAIN LINE: dense CTC sequence model on coarse region,
                           directly outputs character sequences with separator tokens
        "energy_valley"  – split coarse region by energy valleys, then onset detection
        "iki_heuristic"  – LEGACY: onset + IKI/gap heuristic grouping
    """
    target_rate_hz = seg_meta["target_rate_hz"]

    # ── Stage 1: coarse regions (shared) ──
    probs, times = run_binary_inference(
        seg_model, sensor, seg_means, seg_stds,
        seg_meta["window_ms"], seg_meta["stride_ms"], target_rate_hz, device)
    coarse_all = extract_coarse_regions(probs, times, threshold=segment_threshold)
    coarse = _select_primary_coarse_regions(coarse_all, max_regions=max_coarse_regions)

    # ── Stage 2 ──
    stage2_debug = {}
    n_total_onsets = 0
    target_password_count = max(1, int(expected_password_count))
    ctc_passwords = None  # only set by dense_ctc path

    if stage2_method == "dense_ctc":
        # ── Stage 2 main-line: Dense CTC ──
        from stage2_ctc import run_stage2_ctc, score_ctc_passwords

        if ctc_model is None:
            raise ValueError("dense_ctc requires ctc_model/ctc_means/ctc_stds to be loaded")

        ctc_passwords, ctc_debug = run_stage2_ctc(
            sensor, coarse,
            ctc_model, ctc_means, ctc_stds, device,
            n_passwords=target_password_count,
            password_len=expected_password_len,
            beam_width=ctc_beam_width,
        )
        stage2_debug = ctc_debug
        # CTC directly outputs password strings — no onset groups needed
        # We still build dummy refined_episodes for boundary metrics
        password_groups_s = []  # not used for CTC scoring, but needed for episode metrics
        n_total_onsets = sum(len(pw.replace('?', '')) for pw in ctc_passwords)

    elif stage2_method == "energy_valley":
        # ── Stage 2v2: Energy-Valley Segmentation ──
        password_groups_s, ev_debug = run_stage2_energy_valley(
            sensor, coarse,
            onset_model, onset_means, onset_stds, onset_meta,
            device,
            n_passwords=target_password_count,
            expected_password_len=expected_password_len,
            onset_base_threshold=onset_base_threshold,
            onset_nms_radius_s=onset_nms_radius_s,
            onset_smooth_n=onset_smooth_n,
            energy_window_ms=energy_window_ms,
            energy_stride_ms=energy_stride_ms,
            min_valley_width_ms=min_valley_width_ms,
            min_segment_duration_ms=min_segment_duration_ms,
        )
        n_total_onsets = sum(len(g) for g in password_groups_s)
        stage2_debug = ev_debug
        stage2_debug["method"] = "energy_valley"

    else:
        # ── Stage 2 legacy: onset + IKI heuristic grouping ──
        all_peaks = []
        coarse_debug = []
        for region in coarse:
            region_max_peaks = max_onsets_per_region or max(16, min(64, int(region.duration_s * 0.7) + 12))
            peaks = detect_onsets_in_region(
                onset_model, sensor, onset_means, onset_stds, region,
                onset_meta["window_ms"], onset_meta["stride_ms"], target_rate_hz,
                device, threshold=onset_threshold, nms_radius_s=onset_nms_radius_s,
                smooth_n=onset_smooth_n, max_peaks=region_max_peaks)
            all_peaks.extend(peaks)
            coarse_debug.append({
                "start_s": region.start_s, "end_s": region.end_s,
                "duration_s": region.duration_s, "mean_prob": region.mean_prob,
                "max_prob": region.max_prob, "kept_peaks": len(peaks),
                "peak_probs": [float(pk["prob"]) for pk in peaks[:20]],
            })
        all_peaks = sorted(all_peaks, key=lambda pk: pk["time_s"])
        all_onsets = [pk["time_s"] for pk in all_peaks]
        onset_score_map = {round(pk["time_s"], 6): float(pk["prob"]) for pk in all_peaks}
        n_total_onsets = len(all_onsets)

        clusters = find_password_rhythm_clusters(all_onsets, expected_len=expected_password_len)
        group_candidates = []
        protocol_groups = _split_onsets_by_largest_gaps(
            all_onsets, n_groups=target_password_count,
            min_group_len=max(4, expected_password_len // 2))
        for sub in protocol_groups:
            best_seq, meta = _best_password_subsequence(
                sub, onset_score_map, expected_len=expected_password_len, min_len=5, max_len=12)
            if not best_seq: continue
            group_candidates.append({"onsets": best_seq, "start_s": best_seq[0],
                                     "end_s": best_seq[-1], **meta, "source": "protocol_gap_split"})
        for cluster in clusters:
            subs = split_cluster_into_passwords(cluster)
            for sub in subs:
                best_seq, meta = _best_password_subsequence(
                    sub, onset_score_map, expected_len=expected_password_len, min_len=5, max_len=12)
                if not best_seq: continue
                if meta["score"] < group_quality_threshold: continue
                group_candidates.append({"onsets": best_seq, "start_s": best_seq[0],
                                         "end_s": best_seq[-1], **meta, "source": "rhythm_cluster"})
        selected_groups = _select_final_groups(group_candidates, max_groups=target_password_count)
        password_groups_s = [g["onsets"] for g in selected_groups]
        stage2_debug = {
            "method": "iki_heuristic",
            "coarse_regions": coarse_debug,
            "group_candidates": [
                {"start_s": g["start_s"], "end_s": g["end_s"], "n_onsets": g["n_onsets"],
                 "median_iki": g["median_iki"], "iki_cv": g["iki_cv"],
                 "mean_prob": g["mean_prob"], "score": g["score"], "source": g.get("source", "")}
                for g in selected_groups
            ],
        }

    # ── Convert to ns for classifier (not needed for dense_ctc) ──
    if stage2_method == "dense_ctc":
        password_groups_ns = []
        refined_episodes = []
        # For CTC, we don't have onset-level groups — build approximate episodes
        # from coarse regions for boundary metric reporting
        for region in coarse:
            refined_episodes.append(Episode(start_s=region.start_s, end_s=region.end_s, label="password"))
    else:
        password_groups_ns = [[int(t * 1e9) for t in g] for g in password_groups_s]
        refined_episodes = []
        for g in password_groups_s:
            if g:
                refined_episodes.append(Episode(start_s=g[0] - 0.15, end_s=g[-1] + 0.25, label="password"))

    # GT episodes for boundary evaluation
    gt_episodes = [Episode(start_s=s["start_time_ns"]/1e9, end_s=s["end_time_ns"]/1e9, label="password")
                   for s in gt_refined_segs]
    ep_match = match_episodes(refined_episodes, gt_episodes, min_iou=0.3)

    # ── Stage 3: classify / score ──
    e2e_results = []
    gt_results = []

    for pw_idx, ref in enumerate(gt_passwords):
        if stage2_method == "dense_ctc" and ctc_passwords is not None:
            # CTC path: hypothesis comes directly from CTC decode
            hyp = ctc_passwords[pw_idx] if pw_idx < len(ctc_passwords) else ""
            hyp_clean = hyp.replace('?', '')  # remove padding markers
            e2e_r = {
                "reference": ref,
                "hypothesis": hyp,
                "n_onsets": len(hyp_clean),
                "n_valid_windows": len(hyp_clean),
                "cer": levenshtein(ref, hyp) / max(len(ref), 1),
                "char_top1": sum(1 for a, b in zip(hyp, ref) if a == b) / max(len(ref), 1),
                "char_top3": sum(1 for a, b in zip(hyp, ref) if a == b) / max(len(ref), 1),  # CTC has no top-k, top1=top3
                "char_top5": sum(1 for a, b in zip(hyp, ref) if a == b) / max(len(ref), 1),
            }
            for c in SEQ_HIT_CUTOFFS:
                e2e_r[f"seq_top{c}"] = 1 if hyp == ref else 0
        elif pw_idx < len(password_groups_ns):
            e2e_r = score_one_password(
                password_groups_ns[pw_idx], ref, sensor,
                classifier, cls_classes, cls_means, cls_stds, device, target_rate_hz)
        else:
            e2e_r = {"reference": ref, "hypothesis": "", "cer": 1.0,
                     "char_top1": 0.0, "char_top3": 0.0, "char_top5": 0.0,
                     "n_onsets": 0, "n_valid_windows": 0}
            for c in SEQ_HIT_CUTOFFS: e2e_r[f"seq_top{c}"] = 0
        e2e_results.append(e2e_r)

        # GT baseline always uses the existing onset→classifier path
        if pw_idx < len(gt_password_groups_ns):
            gt_r = score_one_password(
                gt_password_groups_ns[pw_idx], ref, sensor,
                classifier, cls_classes, cls_means, cls_stds, device, target_rate_hz)
        else:
            gt_r = dict(e2e_r)
        gt_results.append(gt_r)

    # ── Aggregate metrics ──
    def aggregate(results, n_seqs, n_chars):
        out = {}
        for k in ("char_top1", "char_top3", "char_top5"):
            out[k] = sum(r[k] * len(r["reference"]) for r in results) / max(n_chars, 1)
        out["cer"] = sum(levenshtein(r["reference"], r.get("hypothesis", "")) for r in results) / max(n_chars, 1)
        for cutoff in SEQ_HIT_CUTOFFS:
            out[f"sequence_top{cutoff}"] = sum(r.get(f"seq_top{cutoff}", 0) for r in results) / max(n_seqs, 1)
        return out

    n_seqs = len(gt_passwords)
    n_chars = sum(len(pw) for pw in gt_passwords)

    return {
        "session": os.path.basename(sess_prefix),
        "n_gt_passwords": n_seqs,
        "n_gt_chars": n_chars,
        "n_coarse_regions": len(coarse),
        "n_coarse_regions_raw": len(coarse_all),
        "n_refined_episodes": len(refined_episodes),
        "n_predicted_groups": len(password_groups_ns),
        "n_total_onsets": n_total_onsets,
        "episode_iou": ep_match.mean_iou,
        "episode_precision": ep_match.precision,
        "episode_recall": ep_match.recall,
        "start_error_ms": ep_match.mean_start_error_ms,
        "end_error_ms": ep_match.mean_end_error_ms,
        "e2e_full": aggregate(e2e_results, n_seqs, n_chars),
        "gt_baseline": aggregate(gt_results, n_seqs, n_chars),
        "e2e_examples": [{"ref": r["reference"], "hyp": r.get("hypothesis", "")} for r in e2e_results[:5]],
        "stage2_method": stage2_method,
        "debug": stage2_debug,
    }


def _aggregate_results(all_results):
    if not all_results:
        return {}
    n_seqs = sum(r["n_gt_passwords"] for r in all_results)
    n_chars = sum(r["n_gt_chars"] for r in all_results)
    out = {
        "n_sessions": len(all_results),
        "n_passwords": n_seqs,
        "n_chars": n_chars,
    }
    for tag in ("e2e_full", "gt_baseline"):
        metrics = {}
        for k in ("char_top1", "char_top3", "char_top5", "cer"):
            metrics[k] = sum(r[tag][k] * r["n_gt_chars"] for r in all_results) / max(n_chars, 1)
        for cutoff in SEQ_HIT_CUTOFFS:
            key = f"sequence_top{cutoff}"
            metrics[key] = sum(r[tag][key] * r["n_gt_passwords"] for r in all_results) / max(n_seqs, 1)
        out[tag] = metrics
    out["episode_iou"] = sum(r["episode_iou"] for r in all_results) / max(len(all_results), 1)
    out["episode_precision"] = sum(r["episode_precision"] for r in all_results) / max(len(all_results), 1)
    out["episode_recall"] = sum(r["episode_recall"] for r in all_results) / max(len(all_results), 1)
    out["n_coarse_regions"] = sum(r["n_coarse_regions"] for r in all_results) / max(len(all_results), 1)
    out["n_predicted_groups"] = sum(r["n_predicted_groups"] for r in all_results) / max(len(all_results), 1)
    out["n_total_onsets"] = sum(r["n_total_onsets"] for r in all_results) / max(len(all_results), 1)
    return out


def _sweep_score(summary, expected_password_count=5):
    if not summary:
        return -1e9
    e2e = summary["e2e_full"]
    expected_groups = max(1.0, float(expected_password_count))
    group_penalty = abs(summary["n_predicted_groups"] - expected_groups) / expected_groups
    onset_penalty = max(0.0, (summary["n_total_onsets"] - 60.0) / 60.0)
    score = (
        2.5 * summary["episode_iou"]
        + 1.2 * e2e["char_top3"]
        + 0.6 * e2e["char_top1"]
        - 1.2 * e2e["cer"]
        - 0.8 * group_penalty
        - 0.4 * onset_penalty
    )
    return float(score)


def _parse_grid(arg_value, cast=float):
    if isinstance(arg_value, (list, tuple)):
        return [cast(x) for x in arg_value]
    parts = [p.strip() for p in str(arg_value).split(",") if p.strip()]
    return [cast(p) for p in parts]


# ── CLI entry point ──────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Two-stage password segment detection + classifier (Path B)")
    p.add_argument("--project-root", default="")
    p.add_argument("--segment-checkpoint", default="results/password_segment_detector.pt")
    p.add_argument("--segment-scaler", default="results/password_segment_scaler.npz")
    p.add_argument("--onset-checkpoint", default="results/onset_detector.pt")
    p.add_argument("--onset-scaler", default="results/onset_scaler.npz")
    p.add_argument("--classifier-checkpoint", default="results/inception_password_final.pt")
    p.add_argument("--classifier-scaler", default="results/inception_password_scaler.npz")
    p.add_argument("--mixed2-dirs", nargs="+", default=["data/raw/onset_mixed2"])
    p.add_argument("--report", default="results/password_segment_e2e_report.json")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--segment-threshold", type=float, default=0.5)
    p.add_argument("--onset-threshold", type=float, default=0.5)
    p.add_argument("--expected-password-count", type=int, default=5,
                   help="Protocol prior: expected number of passwords in each mixed2 password block")
    p.add_argument("--expected-password-len", type=int, default=8)
    p.add_argument("--max-coarse-regions", type=int, default=1)
    p.add_argument("--onset-nms-radius-s", type=float, default=0.12)
    p.add_argument("--onset-smooth-n", type=int, default=3)
    p.add_argument("--max-onsets-per-region", type=int, default=56)
    p.add_argument("--group-quality-threshold", type=float, default=0.48)
    p.add_argument("--stage2-method", choices=["energy_valley", "iki_heuristic", "dense_ctc"], default="energy_valley",
                   help="Stage 2 strategy: energy_valley, iki_heuristic, or dense_ctc")
    p.add_argument("--energy-window-ms", type=int, default=80, help="Energy envelope window (ms)")
    p.add_argument("--energy-stride-ms", type=int, default=10, help="Energy envelope stride (ms)")
    p.add_argument("--min-valley-width-ms", type=int, default=200, help="Minimum valley width to qualify as inter-password gap")
    p.add_argument("--min-segment-duration-ms", type=int, default=1500, help="Minimum password segment duration (ms)")
    p.add_argument("--onset-base-threshold", type=float, default=0.3, help="Lower onset threshold for energy_valley per-segment detection")
    p.add_argument("--ctc-checkpoint", default="results/claude_ctc_stage2.pt",
                   help="Dense CTC Stage-2 checkpoint path")
    p.add_argument("--ctc-scaler", default="results/claude_ctc_stage2_scaler.npz",
                   help="Dense CTC Stage-2 scaler path")
    p.add_argument("--ctc-beam-width", type=int, default=20,
                   help="Beam width for dense_ctc decoding")
    p.add_argument("--auto-sweep", action="store_true",
                   help="Try multiple Stage-2 parameter combinations and print a ranked summary.")
    p.add_argument("--sweep-onset-thresholds", default="0.45,0.5,0.55")
    p.add_argument("--sweep-onset-nms-radii", default="0.10,0.12,0.14")
    p.add_argument("--sweep-max-onsets", default="40,48,56")
    p.add_argument("--sweep-group-quality", default="0.35,0.42,0.48")
    p.add_argument("--sweep-max-coarse-regions", default="1")
    p.add_argument("--sweep-energy-window-ms", default="60,80,100")
    p.add_argument("--sweep-min-valley-width-ms", default="150,200,250")
    p.add_argument("--sweep-min-segment-duration-ms", default="1200,1500,1800")
    p.add_argument("--sweep-onset-base-thresholds", default="0.25,0.30,0.35")
    args = p.parse_args()

    if args.project_root:
        _setup_imports(args.project_root)
        root = os.path.abspath(args.project_root)
        for attr in ["segment_checkpoint", "segment_scaler", "onset_checkpoint",
                     "onset_scaler", "classifier_checkpoint", "classifier_scaler",
                     "ctc_checkpoint", "ctc_scaler", "report"]:
            v = getattr(args, attr)
            if not os.path.isabs(v):
                setattr(args, attr, os.path.join(root, v))
        args.mixed2_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in args.mixed2_dirs]
    else:
        _setup_imports()

    from run_password_closure_inception import load_final_inception, normalize_sequence

    req = (args.device or "auto").lower()
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    device = torch.device(req)
    print(f"Device: {device}")

    seg_model, seg_means, seg_stds, seg_meta = load_segment_detector(args.segment_checkpoint, args.segment_scaler, device)
    print(f"Segment detector loaded")
    onset_model, onset_means, onset_stds, onset_meta = load_onset_detector(args.onset_checkpoint, args.onset_scaler, device)
    print(f"Onset detector loaded")
    classifier, cls_classes, cls_means, cls_stds = load_final_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    print(f"Classifier loaded ({len(cls_classes)} classes)")
    ctc_model = ctc_means = ctc_stds = None
    if args.stage2_method == "dense_ctc":
        from stage2_ctc import load_ctc_model
        ctc_model, ctc_means, ctc_stds, _ctc_meta = load_ctc_model(args.ctc_checkpoint, args.ctc_scaler, device)
        print("Dense CTC loaded")

    sessions = discover_sessions(args.mixed2_dirs, mode_filter="mixed2", dedup=False)
    if not sessions:
        sessions = discover_sessions(args.mixed2_dirs, mode_filter="", dedup=False)
    print(f"Found {len(sessions)} mixed2 sessions\n")

    session_payloads = []
    for sess in sessions:
        alog = sess + "_activity_log.csv"
        events_path = sess + "_events.csv"
        if not os.path.exists(alog):
            continue
        sensor = load_sensor_csv(sess + "_sensor.csv")
        activity_segments = load_activity_log(alog)
        events = _load_supported_press_timestamps(events_path) if os.path.exists(events_path) else np.array([], dtype=np.int64)
        gt_refined = refine_password_segments_with_events(activity_segments, events)
        if not gt_refined:
            continue
        gt_passwords = []
        for seg in gt_refined:
            gt_passwords.extend([normalize_sequence(p) for p in seg.get("prompts", []) if p])
        if not gt_passwords:
            continue
        gt_password_groups = _extract_gt_password_groups(events_path, gt_refined) if os.path.exists(events_path) else []
        session_payloads.append((sess, sensor, gt_passwords, gt_refined, gt_password_groups))

    def evaluate_with_params(run_args):
        all_results = []
        for sess, sensor, gt_passwords, gt_refined, gt_password_groups in session_payloads:
            print(f"  Session: {os.path.basename(sess)}  ({len(gt_passwords)} passwords)")
            result = run_full_pipeline(
                sensor, sess,
                seg_model, seg_means, seg_stds, seg_meta,
                onset_model, onset_means, onset_stds, onset_meta,
                classifier, cls_classes, cls_means, cls_stds,
                device, gt_passwords, gt_refined, gt_password_groups,
                expected_password_count=run_args.get("expected_password_count", args.expected_password_count),
                segment_threshold=run_args["segment_threshold"],
                onset_threshold=run_args["onset_threshold"],
                expected_password_len=args.expected_password_len,
                max_coarse_regions=run_args["max_coarse_regions"],
                onset_nms_radius_s=run_args["onset_nms_radius_s"],
                onset_smooth_n=run_args["onset_smooth_n"],
                max_onsets_per_region=run_args["max_onsets_per_region"],
                group_quality_threshold=run_args["group_quality_threshold"],
                stage2_method=run_args.get("stage2_method", args.stage2_method),
                energy_window_ms=run_args.get("energy_window_ms", args.energy_window_ms),
                energy_stride_ms=run_args.get("energy_stride_ms", args.energy_stride_ms),
                min_valley_width_ms=run_args.get("min_valley_width_ms", args.min_valley_width_ms),
                min_segment_duration_ms=run_args.get("min_segment_duration_ms", args.min_segment_duration_ms),
                onset_base_threshold=run_args.get("onset_base_threshold", args.onset_base_threshold),
                ctc_model=ctc_model,
                ctc_means=ctc_means,
                ctc_stds=ctc_stds,
                ctc_beam_width=run_args.get("ctc_beam_width", args.ctc_beam_width),
            )
            all_results.append(result)
            e = result["e2e_full"]
            g = result["gt_baseline"]
            print(f"    Coarse regions: {result['n_coarse_regions']}/{result['n_coarse_regions_raw']}  |  Pred groups: {result['n_predicted_groups']}  |  Onsets: {result['n_total_onsets']}")
            print(f"    Episode IoU: {result['episode_iou']:.3f}  P={result['episode_precision']:.3f}  R={result['episode_recall']:.3f}")
            print(f"    E2E  char_top1={e['char_top1']:.1%}  top3={e['char_top3']:.1%}  top5={e['char_top5']:.1%}  CER={e['cer']:.1%}")
            print(f"    GT   char_top1={g['char_top1']:.1%}  top3={g['char_top3']:.1%}  top5={g['char_top5']:.1%}  CER={g['cer']:.1%}")
            for ex in result.get("e2e_examples", []):
                print(f"      ref={ex['ref']}  hyp={ex['hyp']}")
        summary = _aggregate_results(all_results)
        return all_results, summary

    if args.auto_sweep:
        max_coarse_vals = _parse_grid(args.sweep_max_coarse_regions, int)
        sweep_trials = []
        if args.stage2_method == "energy_valley":
            energy_windows = _parse_grid(args.sweep_energy_window_ms, int)
            valley_widths = _parse_grid(args.sweep_min_valley_width_ms, int)
            min_seg_durations = _parse_grid(args.sweep_min_segment_duration_ms, int)
            onset_base_thresholds = _parse_grid(args.sweep_onset_base_thresholds, float)
            combos = list(itertools.product(
                energy_windows, valley_widths, min_seg_durations, onset_base_thresholds, max_coarse_vals
            ))
        else:
            onset_thresholds = _parse_grid(args.sweep_onset_thresholds, float)
            onset_nms_radii = _parse_grid(args.sweep_onset_nms_radii, float)
            max_onsets_vals = _parse_grid(args.sweep_max_onsets, int)
            group_quality_vals = _parse_grid(args.sweep_group_quality, float)
            combos = list(itertools.product(
                onset_thresholds, onset_nms_radii, max_onsets_vals, group_quality_vals, max_coarse_vals
            ))
        print(f"Auto sweep: {len(combos)} parameter combinations\n")
        for idx, combo in enumerate(combos, 1):
            if args.stage2_method == "energy_valley":
                energy_window_ms, min_valley_width_ms, min_segment_duration_ms, onset_base_threshold, max_coarse = combo
                run_args = {
                    "expected_password_count": args.expected_password_count,
                    "segment_threshold": args.segment_threshold,
                    "onset_threshold": args.onset_threshold,
                    "max_coarse_regions": max_coarse,
                    "onset_nms_radius_s": args.onset_nms_radius_s,
                    "onset_smooth_n": args.onset_smooth_n,
                    "max_onsets_per_region": args.max_onsets_per_region,
                    "group_quality_threshold": args.group_quality_threshold,
                    "stage2_method": args.stage2_method,
                    "energy_window_ms": energy_window_ms,
                    "energy_stride_ms": args.energy_stride_ms,
                    "min_valley_width_ms": min_valley_width_ms,
                    "min_segment_duration_ms": min_segment_duration_ms,
                    "onset_base_threshold": onset_base_threshold,
                }
            else:
                onset_thr, nms_r, max_onsets, group_q, max_coarse = combo
                run_args = {
                    "expected_password_count": args.expected_password_count,
                    "segment_threshold": args.segment_threshold,
                    "onset_threshold": onset_thr,
                    "max_coarse_regions": max_coarse,
                    "onset_nms_radius_s": nms_r,
                    "onset_smooth_n": args.onset_smooth_n,
                    "max_onsets_per_region": max_onsets,
                    "group_quality_threshold": group_q,
                    "stage2_method": args.stage2_method,
                    "energy_window_ms": args.energy_window_ms,
                    "energy_stride_ms": args.energy_stride_ms,
                    "min_valley_width_ms": args.min_valley_width_ms,
                    "min_segment_duration_ms": args.min_segment_duration_ms,
                    "onset_base_threshold": args.onset_base_threshold,
                }
            print(f"\n--- Sweep {idx}/{len(combos)} --- {run_args}")
            run_results, summary = evaluate_with_params(run_args)
            score = _sweep_score(summary, expected_password_count=args.expected_password_count)
            sweep_trials.append({
                "params": run_args,
                "summary": summary,
                "score": score,
                "sessions": run_results,
            })
            print(f"  Sweep score: {score:.4f}")
            print(f"  Summary: IoU={summary.get('episode_iou', 0):.3f}  groups={summary.get('n_predicted_groups', 0):.2f}  onsets={summary.get('n_total_onsets', 0):.1f}  top3={summary.get('e2e_full', {}).get('char_top3', 0):.1%}  CER={summary.get('e2e_full', {}).get('cer', 1):.1%}")

        sweep_trials.sort(key=lambda t: t["score"], reverse=True)
        best = sweep_trials[0] if sweep_trials else None
        print(f"\n{'='*60}")
        print("  SWEEP SUMMARY (best first)")
        print(f"{'='*60}")
        for idx, trial in enumerate(sweep_trials[:10], 1):
            s = trial["summary"]
            pset = trial["params"]
            print(f"{idx:2d}. score={trial['score']:.4f}  IoU={s.get('episode_iou', 0):.3f}  groups={s.get('n_predicted_groups', 0):.2f}  onsets={s.get('n_total_onsets', 0):.1f}  top3={s.get('e2e_full', {}).get('char_top3', 0):.1%}  CER={s.get('e2e_full', {}).get('cer', 1):.1%}  params={pset}")

        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump({
                "mode": "auto_sweep",
                "best": best,
                "trials": sweep_trials,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved sweep report → {args.report}")
        return

    all_results, summary = evaluate_with_params({
        "expected_password_count": args.expected_password_count,
        "segment_threshold": args.segment_threshold,
        "onset_threshold": args.onset_threshold,
        "max_coarse_regions": args.max_coarse_regions,
        "onset_nms_radius_s": args.onset_nms_radius_s,
        "onset_smooth_n": args.onset_smooth_n,
        "max_onsets_per_region": args.max_onsets_per_region,
        "group_quality_threshold": args.group_quality_threshold,
        "stage2_method": args.stage2_method,
        "energy_window_ms": args.energy_window_ms,
        "energy_stride_ms": args.energy_stride_ms,
        "min_valley_width_ms": args.min_valley_width_ms,
        "min_segment_duration_ms": args.min_segment_duration_ms,
        "onset_base_threshold": args.onset_base_threshold,
        "ctc_beam_width": args.ctc_beam_width,
    })

    if all_results:
        print(f"\n{'='*60}")
        print(f"  AGGREGATE ({summary['n_sessions']} sessions, {summary['n_passwords']} passwords, {summary['n_chars']} chars)")
        print(f"{'='*60}")
        for tag in ("e2e_full", "gt_baseline"):
            label = "E2E Full" if tag == "e2e_full" else "GT Baseline"
            metrics = summary[tag]
            print(f"\n  {label}:")
            print(f"    char_top1: {metrics['char_top1']:.1%}   top3: {metrics['char_top3']:.1%}   top5: {metrics['char_top5']:.1%}")
            for c in SEQ_HIT_CUTOFFS:
                print(f"    seq_top{c}: {metrics[f'sequence_top{c}']:.1%}")
            print(f"    CER: {metrics['cer']:.1%}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump({"mode": "single_run", "summary": summary, "sessions": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {args.report}")


if __name__ == "__main__":
    main()
