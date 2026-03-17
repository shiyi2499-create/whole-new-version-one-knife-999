"""
Onset Detection Utilities
=========================
Core math for the onset detection pipeline:
  - 1D peak detection on probability curves
  - Non-maximum suppression (NMS)
  - Greedy event-level matching (predicted ↔ ground truth)
  - Event-level metric computation (P / R / F1 / timing error / FA per min)
  - Episode boundary metrics (start/end error, IoU, separation)
  - Frame-level activity curve → episode extraction

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

    pairs: list[tuple[float, int, int]] = []
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
    if total_duration_s <= 0:
        return 0.0
    return n_fp / (total_duration_s / 60.0)


# ── Multi-tolerance sweep ────────────────────────────────────

def tolerance_sweep(
    predicted_times_s: list[float] | np.ndarray,
    ground_truth_times_s: list[float] | np.ndarray,
    tolerances_ms: tuple[float, ...] = (25, 50, 75, 100),
) -> dict[float, dict]:
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


# ══════════════════════════════════════════════════════════════
# Episode-level utilities (NEW for activity segmentation)
# ══════════════════════════════════════════════════════════════

@dataclass
class Episode:
    """A contiguous keyboard-activity episode."""
    start_s: float
    end_s: float
    label: str = ""          # "typing_1", "typing_2", "keyboard", etc.
    onset_count: int = 0     # number of detected onsets within

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def extract_episodes_from_activity_curve(
    probs: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 0.5,
    min_duration_s: float = 0.5,
    merge_gap_s: float = 0.8,
    smooth_n: int = 5,
) -> list[Episode]:
    """
    Convert a frame-level keyboard-active probability curve into
    discrete Episode intervals.

    Steps:
      1. Smooth the probability curve
      2. Threshold → binary active/inactive
      3. Find contiguous active runs
      4. Merge runs separated by < merge_gap_s
      5. Filter out episodes shorter than min_duration_s

    Args:
        probs:          (N,) per-frame activity probability
        timestamps:     (N,) centre timestamp of each frame (seconds)
        threshold:      binarisation threshold
        min_duration_s: discard episodes shorter than this
        merge_gap_s:    merge adjacent episodes closer than this
        smooth_n:       moving-average smoothing width (frames)

    Returns:
        list of Episode sorted by start_s
    """
    assert len(probs) == len(timestamps)
    if len(probs) < 2:
        return []

    p = probs.copy().astype(np.float64)
    if smooth_n > 1:
        kernel = np.ones(smooth_n) / smooth_n
        p = np.convolve(p, kernel, mode="same")

    active = p >= threshold
    ts = timestamps.astype(np.float64)

    # Find contiguous active runs
    raw_episodes: list[Episode] = []
    in_episode = False
    ep_start = 0.0

    for i in range(len(active)):
        if active[i] and not in_episode:
            ep_start = ts[i]
            in_episode = True
        elif not active[i] and in_episode:
            raw_episodes.append(Episode(start_s=ep_start, end_s=ts[i]))
            in_episode = False
    if in_episode:
        raw_episodes.append(Episode(start_s=ep_start, end_s=ts[-1]))

    if not raw_episodes:
        return []

    # Merge episodes separated by < merge_gap_s
    merged: list[Episode] = [raw_episodes[0]]
    for ep in raw_episodes[1:]:
        if ep.start_s - merged[-1].end_s < merge_gap_s:
            merged[-1] = Episode(start_s=merged[-1].start_s, end_s=ep.end_s)
        else:
            merged.append(ep)

    # Filter short episodes
    filtered = [ep for ep in merged if ep.duration_s >= min_duration_s]

    return filtered


def classify_episodes_by_density(
    episodes: list[Episode],
    onset_times_s: list[float],
    iki_threshold_s: float = 0.6,
    rate_threshold_hz: float = 2.5,
) -> list[Episode]:
    """
    **Demo-protocol heuristic** for classifying keyboard-activity episodes
    as typing_1 (free/fast typing) vs typing_2 (password-style slow typing).

    NOT a learned classifier — this is a hand-tuned rule designed for the
    structured 2-minute mixed2 protocol where typing_1 is continuous free
    text and typing_2 is slow, deliberate password entry.

    Decision logic (an episode is labelled typing_2 if ANY condition holds):
      1. Median IKI > iki_threshold_s  (large gaps between keystrokes)
      2. Keystroke rate < rate_threshold_hz  (few keystrokes per second)

    The dual criteria make the heuristic more robust than IKI alone:
    IKI captures the inter-key timing pattern, while keystroke rate
    catches episodes where IKI is near the boundary but overall density
    is clearly low (e.g., a short password with only a few keys).

    Args:
        episodes:           list of Episode from activity segmenter
        onset_times_s:      all detected onset timestamps (seconds)
        iki_threshold_s:    median IKI above this → typing_2
        rate_threshold_hz:  keystroke rate below this → typing_2

    Limitations:
      - Assumes exactly two typing styles in the stream (free vs password)
      - Thresholds are tuned for 8-char a-z0-9 passwords typed deliberately
      - Will not generalise to arbitrary typing tasks without re-tuning
    """
    onset_arr = np.array(onset_times_s, dtype=np.float64) if onset_times_s else np.array([])

    classified = []
    for ep in episodes:
        # Find onsets within this episode
        if len(onset_arr) > 0:
            mask = (onset_arr >= ep.start_s) & (onset_arr <= ep.end_s)
            ep_onsets = onset_arr[mask]
        else:
            ep_onsets = np.array([])

        new_ep = Episode(
            start_s=ep.start_s,
            end_s=ep.end_s,
            onset_count=len(ep_onsets),
        )

        # Compute features
        keystroke_rate = len(ep_onsets) / max(ep.duration_s, 0.01)

        if len(ep_onsets) >= 2:
            ikis = np.diff(np.sort(ep_onsets))
            median_iki = float(np.median(ikis))
            # typing_2 if EITHER criterion fires
            if median_iki > iki_threshold_s or keystroke_rate < rate_threshold_hz:
                new_ep.label = "typing_2"
            else:
                new_ep.label = "typing_1"
        elif len(ep_onsets) == 1:
            # Single keystroke in an episode → conservatively label typing_2
            new_ep.label = "typing_2"
        else:
            # No onsets detected but activity segmenter fired →
            # likely non-keyboard motion that leaked through; label typing_1
            # so it doesn't get fed to the password classifier
            new_ep.label = "typing_1"

        classified.append(new_ep)

    return classified


def group_onsets_by_gap(
    onset_times_ns: list[int],
    n_groups: int = 0,
    default_gap_ns: int = 1_500_000_000,
) -> list[list[int]]:
    """
    Split a list of onset timestamps into groups using gap-based heuristic.

    Two modes:
      1. n_groups > 0:  Find the (n_groups - 1) largest inter-onset gaps
         and split there.  This is useful when the protocol specifies a
         known number of passwords.
      2. n_groups == 0:  Split at any gap > default_gap_ns (fallback).

    Args:
        onset_times_ns: sorted onset timestamps in nanoseconds
        n_groups:       desired number of groups (0 = auto)
        default_gap_ns: gap threshold for auto mode (default 1.5 s)

    Returns:
        list of groups, each a list of onset timestamps (ns)
    """
    if not onset_times_ns:
        return []

    sorted_onsets = sorted(onset_times_ns)

    if len(sorted_onsets) == 1:
        return [sorted_onsets]

    gaps = [(sorted_onsets[i + 1] - sorted_onsets[i], i)
            for i in range(len(sorted_onsets) - 1)]

    if n_groups > 1 and len(gaps) >= n_groups - 1:
        # Pick the (n_groups - 1) largest gaps as split points
        sorted_gaps = sorted(gaps, key=lambda x: -x[0])
        split_indices = sorted([g[1] for g in sorted_gaps[:n_groups - 1]])
    else:
        # Fallback: split at gaps exceeding threshold
        split_indices = [i for gap_ns, i in gaps if gap_ns > default_gap_ns]

    # Build groups from split indices
    groups: list[list[int]] = []
    prev = 0
    for idx in split_indices:
        groups.append(sorted_onsets[prev:idx + 1])
        prev = idx + 1
    groups.append(sorted_onsets[prev:])

    return groups


# ── Episode-level metrics ────────────────────────────────────

@dataclass
class EpisodeMatchResult:
    """Result of episode-level boundary evaluation."""
    n_gt: int = 0
    n_pred: int = 0
    n_matched: int = 0
    start_errors_s: list[float] = field(default_factory=list)
    end_errors_s: list[float] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)
    correctly_separated: bool = False  # True if 2 GT eps → 2 pred eps

    @property
    def mean_start_error_ms(self) -> float:
        return float(np.mean(self.start_errors_s)) * 1000 if self.start_errors_s else float("inf")

    @property
    def mean_end_error_ms(self) -> float:
        return float(np.mean(self.end_errors_s)) * 1000 if self.end_errors_s else float("inf")

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.ious)) if self.ious else 0.0


def episode_iou(pred: Episode, gt: Episode) -> float:
    """Compute temporal IoU between two episodes."""
    overlap_start = max(pred.start_s, gt.start_s)
    overlap_end = min(pred.end_s, gt.end_s)
    intersection = max(0.0, overlap_end - overlap_start)

    union = (pred.end_s - pred.start_s) + (gt.end_s - gt.start_s) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def match_episodes(
    predicted: list[Episode],
    ground_truth: list[Episode],
    min_iou: float = 0.3,
) -> EpisodeMatchResult:
    """
    Match predicted episodes to ground-truth episodes greedily by IoU.

    For each GT episode, find the predicted episode with highest IoU.
    If IoU >= min_iou, it's a match; record boundary errors.

    Also checks whether 2 GT episodes are correctly separated
    (i.e., matched to 2 distinct predicted episodes).
    """
    result = EpisodeMatchResult(n_gt=len(ground_truth), n_pred=len(predicted))

    if not predicted or not ground_truth:
        return result

    # Compute IoU matrix
    iou_matrix = np.zeros((len(ground_truth), len(predicted)))
    for gi, gt_ep in enumerate(ground_truth):
        for pi, pred_ep in enumerate(predicted):
            iou_matrix[gi, pi] = episode_iou(pred_ep, gt_ep)

    matched_pred: set[int] = set()
    matched_gt_to_pred: dict[int, int] = {}

    # Greedy matching by descending IoU
    while True:
        if iou_matrix.size == 0:
            break
        best_iou = iou_matrix.max()
        if best_iou < min_iou:
            break
        gi, pi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        gi, pi = int(gi), int(pi)

        if pi in matched_pred:
            iou_matrix[gi, pi] = -1
            continue

        matched_pred.add(pi)
        matched_gt_to_pred[gi] = pi
        result.n_matched += 1
        result.ious.append(float(best_iou))
        result.start_errors_s.append(abs(predicted[pi].start_s - ground_truth[gi].start_s))
        result.end_errors_s.append(abs(predicted[pi].end_s - ground_truth[gi].end_s))

        # Invalidate this row/col
        iou_matrix[gi, :] = -1
        iou_matrix[:, pi] = -1

    # Check separation: if 2 GT episodes, are they matched to 2 different preds?
    if len(ground_truth) >= 2:
        matched_preds = set(matched_gt_to_pred.values())
        result.correctly_separated = len(matched_preds) >= 2

    return result


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


def format_episode_match_result(m: EpisodeMatchResult, label: str = "") -> str:
    lines = []
    if label:
        lines.append(f"  {label}")
    lines.append(f"    GT episodes: {m.n_gt}  Pred episodes: {m.n_pred}  Matched: {m.n_matched}")
    lines.append(f"    Mean IoU: {m.mean_iou:.3f}")
    lines.append(f"    Start boundary error: {m.mean_start_error_ms:.1f}ms")
    lines.append(f"    End boundary error:   {m.mean_end_error_ms:.1f}ms")
    lines.append(f"    2-episode separation:  {'✓' if m.correctly_separated else '✗'}")
    return "\n".join(lines)
