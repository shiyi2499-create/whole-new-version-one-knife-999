"""
Onset Detection Utilities
=========================
Core math for the onset detection pipeline:
  - 1D peak detection on probability curves
  - Non-maximum suppression (NMS)
  - Greedy event-level matching (predicted ↔ ground truth)
  - Event-level metric computation (P / R / F1 / timing error / FA per min)

All time values are in **seconds** unless a variable name ends with _ns or _ms.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Peak detection ────────────────────────────────────────────

def detect_peaks(
    probs: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 0.5,
    smooth_n: int = 0,
) -> list[dict]:
    """
    Find local maxima in a 1-D probability curve that exceed *threshold*.

    Args:
        probs:      shape (N,) – per-window onset probability
        timestamps: shape (N,) – centre timestamp of each window (seconds)
        threshold:  minimum probability to consider
        smooth_n:   if >0, apply uniform moving-average of this width first

    Returns:
        list of {"time_s": float, "prob": float, "index": int}
    """
    assert len(probs) == len(timestamps)
    if len(probs) < 3:
        return []

    p = probs.copy().astype(np.float64)
    if smooth_n > 1:
        kernel = np.ones(smooth_n) / smooth_n
        p = np.convolve(p, kernel, mode="same")

    peaks = []
    for i in range(1, len(p) - 1):
        if p[i] > p[i - 1] and p[i] >= p[i + 1] and p[i] >= threshold:
            peaks.append({
                "time_s": float(timestamps[i]),
                "prob": float(p[i]),
                "index": int(i),
            })
    return peaks


# ── Non-maximum suppression ──────────────────────────────────

def nms_1d(
    peaks: list[dict],
    radius_s: float = 0.100,
) -> list[dict]:
    """
    Suppress weaker peaks within ±radius_s of a stronger peak.

    Greedy: sort descending by prob, keep only if no already-kept peak
    is within the suppression radius.
    """
    if not peaks:
        return []

    sorted_peaks = sorted(peaks, key=lambda p: -p["prob"])
    kept: list[dict] = []
    kept_times: list[float] = []

    for pk in sorted_peaks:
        t = pk["time_s"]
        if any(abs(t - kt) < radius_s for kt in kept_times):
            continue
        kept.append(pk)
        kept_times.append(t)

    # Return in chronological order
    return sorted(kept, key=lambda p: p["time_s"])


# ── Event-level matching ─────────────────────────────────────

@dataclass
class MatchResult:
    """Result of greedy event matching."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    timing_errors_s: list[float] = field(default_factory=list)
    matched_pairs: list[tuple[float, float]] = field(default_factory=list)
    # (pred_time, gt_time) for each TP

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-12)

    @property
    def timing_error_mean(self) -> float:
        return float(np.mean(self.timing_errors_s)) if self.timing_errors_s else 0.0

    @property
    def timing_error_median(self) -> float:
        return float(np.median(self.timing_errors_s)) if self.timing_errors_s else 0.0

    @property
    def timing_error_std(self) -> float:
        return float(np.std(self.timing_errors_s)) if self.timing_errors_s else 0.0


def match_events(
    predicted_times_s: list[float] | np.ndarray,
    ground_truth_times_s: list[float] | np.ndarray,
    tolerance_s: float = 0.050,
) -> MatchResult:
    """
    Greedy bipartite matching between predicted and ground-truth onset times.

    For each predicted onset, find the nearest *unmatched* ground-truth onset.
    If the distance is ≤ tolerance_s, count as TP; otherwise FP.
    Any ground-truth onset that was never matched is a FN.

    Matching is done greedily by ascending distance to favour the
    tightest-paired matches first.
    """
    pred = np.asarray(predicted_times_s, dtype=np.float64)
    gt = np.asarray(ground_truth_times_s, dtype=np.float64)

    result = MatchResult()

    if len(pred) == 0:
        result.fn = len(gt)
        return result
    if len(gt) == 0:
        result.fp = len(pred)
        return result

    # Build all candidate pairs sorted by distance
    pairs: list[tuple[float, int, int]] = []  # (dist, pred_idx, gt_idx)
    for pi, pt in enumerate(pred):
        for gi, gt_t in enumerate(gt):
            d = abs(pt - gt_t)
            if d <= tolerance_s:
                pairs.append((d, pi, gi))
    pairs.sort(key=lambda x: x[0])

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()

    for dist, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        result.tp += 1
        result.timing_errors_s.append(dist)
        result.matched_pairs.append((float(pred[pi]), float(gt[gi])))

    result.fp = len(pred) - len(matched_pred)
    result.fn = len(gt) - len(matched_gt)
    return result


def false_alarms_per_minute(n_fp: int, total_duration_s: float) -> float:
    """False alarm rate in events per minute."""
    if total_duration_s <= 0:
        return 0.0
    return n_fp / (total_duration_s / 60.0)


# ── Multi-tolerance sweep ────────────────────────────────────

def tolerance_sweep(
    predicted_times_s: list[float] | np.ndarray,
    ground_truth_times_s: list[float] | np.ndarray,
    tolerances_ms: tuple[float, ...] = (25, 50, 75, 100),
) -> dict[float, dict]:
    """
    Run event matching at multiple tolerances and return a dict
    keyed by tolerance_ms with P / R / F1 at each level.
    """
    results = {}
    for tol_ms in tolerances_ms:
        m = match_events(predicted_times_s, ground_truth_times_s, tolerance_s=tol_ms / 1000.0)
        results[tol_ms] = {
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "timing_error_mean_ms": m.timing_error_mean * 1000,
            "timing_error_median_ms": m.timing_error_median * 1000,
        }
    return results


# ── Pretty-print helpers ─────────────────────────────────────

def format_match_result(m: MatchResult, label: str = "") -> str:
    lines = []
    if label:
        lines.append(f"  {label}")
    lines.append(f"    TP={m.tp}  FP={m.fp}  FN={m.fn}")
    lines.append(f"    Precision={m.precision:.3f}  Recall={m.recall:.3f}  F1={m.f1:.3f}")
    if m.timing_errors_s:
        lines.append(
            f"    Timing error: mean={m.timing_error_mean*1000:.1f}ms  "
            f"median={m.timing_error_median*1000:.1f}ms  "
            f"std={m.timing_error_std*1000:.1f}ms"
        )
    return "\n".join(lines)
