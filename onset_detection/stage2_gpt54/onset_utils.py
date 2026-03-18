"""
Onset / Password-Boundary Utilities
===================================

Core helpers for:
  - onset event matching
  - peak detection + NMS
  - episode extraction / boundary evaluation
  - password-boundary decoding from 4-class window predictions
  - onset-group alignment for Path B evaluation
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


PASSWORD_BOUNDARY_LABELS = [
    "non_password",
    "password_start",
    "password_active",
    "password_end",
]
PASSWORD_BOUNDARY_TO_ID = {name: i for i, name in enumerate(PASSWORD_BOUNDARY_LABELS)}


def boundary_label_id(name: str) -> int:
    return PASSWORD_BOUNDARY_TO_ID[name]


def boundary_label_name(idx: int) -> str:
    return PASSWORD_BOUNDARY_LABELS[int(idx)]


# ── Peak detection ────────────────────────────────────────────

def detect_peaks(
    probs: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 0.5,
    smooth_n: int = 0,
) -> list[dict]:
    """Find local maxima in a 1-D probability curve."""
    assert len(probs) == len(timestamps)
    if len(probs) < 3:
        return []

    p = probs.astype(np.float64).copy()
    if smooth_n and smooth_n > 1:
        kernel = np.ones(smooth_n, dtype=np.float64) / smooth_n
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

def nms_1d(peaks: list[dict], radius_s: float = 0.100) -> list[dict]:
    """Suppress weaker peaks within ±radius_s of a stronger peak."""
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
        return 2.0 * p * r / max(p + r, 1e-12)

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
    """Greedy bipartite event matching under a timing tolerance."""
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
        result.timing_errors_s.append(float(dist))
        result.matched_pairs.append((float(pred[pi]), float(gt[gi])))

    result.fp = len(pred) - len(matched_pred)
    result.fn = len(gt) - len(matched_gt)
    return result


def false_alarms_per_minute(n_fp: int, total_duration_s: float) -> float:
    if total_duration_s <= 0:
        return 0.0
    return n_fp / max(total_duration_s / 60.0, 1e-12)


def tolerance_sweep(
    predicted_times_s: list[float] | np.ndarray,
    ground_truth_times_s: list[float] | np.ndarray,
    tolerances_ms: tuple[float, ...] = (25, 50, 75, 100),
) -> dict[float, dict]:
    results: dict[float, dict] = {}
    for tol_ms in tolerances_ms:
        m = match_events(predicted_times_s, ground_truth_times_s, tolerance_s=tol_ms / 1000.0)
        results[float(tol_ms)] = {
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "timing_error_mean_ms": m.timing_error_mean * 1000.0,
            "timing_error_median_ms": m.timing_error_median * 1000.0,
        }
    return results


# ── Episode helpers ──────────────────────────────────────────

@dataclass
class Episode:
    start_s: float
    end_s: float
    label: str = ""
    onset_count: int = 0
    score: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s - self.start_s))


@dataclass
class EpisodeMatchResult:
    n_gt: int = 0
    n_pred: int = 0
    n_matched: int = 0
    start_errors_s: list[float] = field(default_factory=list)
    end_errors_s: list[float] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)
    correctly_separated: bool = False

    @property
    def mean_start_error_ms(self) -> float:
        if not self.start_errors_s:
            return float("inf")
        return float(np.mean(self.start_errors_s)) * 1000.0

    @property
    def mean_end_error_ms(self) -> float:
        if not self.end_errors_s:
            return float("inf")
        return float(np.mean(self.end_errors_s)) * 1000.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.ious)) if self.ious else 0.0

    @property
    def precision(self) -> float:
        return self.n_matched / max(self.n_pred, 1)

    @property
    def recall(self) -> float:
        return self.n_matched / max(self.n_gt, 1)



def episode_iou(pred: Episode, gt: Episode) -> float:
    overlap_start = max(pred.start_s, gt.start_s)
    overlap_end = min(pred.end_s, gt.end_s)
    intersection = max(0.0, overlap_end - overlap_start)
    union = pred.duration_s + gt.duration_s - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def match_episodes(
    predicted: list[Episode],
    ground_truth: list[Episode],
    min_iou: float = 0.3,
) -> EpisodeMatchResult:
    result = EpisodeMatchResult(n_gt=len(ground_truth), n_pred=len(predicted))
    if not predicted or not ground_truth:
        return result

    iou_matrix = np.zeros((len(ground_truth), len(predicted)), dtype=np.float64)
    for gi, gt_ep in enumerate(ground_truth):
        for pi, pred_ep in enumerate(predicted):
            iou_matrix[gi, pi] = episode_iou(pred_ep, gt_ep)

    matched_pred: set[int] = set()
    matched_gt_to_pred: dict[int, int] = {}
    while True:
        best_iou = float(iou_matrix.max()) if iou_matrix.size else -1.0
        if best_iou < min_iou:
            break
        gi, pi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        gi, pi = int(gi), int(pi)
        if pi in matched_pred:
            iou_matrix[gi, pi] = -1.0
            continue
        matched_pred.add(pi)
        matched_gt_to_pred[gi] = pi
        result.n_matched += 1
        result.ious.append(best_iou)
        result.start_errors_s.append(abs(predicted[pi].start_s - ground_truth[gi].start_s))
        result.end_errors_s.append(abs(predicted[pi].end_s - ground_truth[gi].end_s))
        iou_matrix[gi, :] = -1.0
        iou_matrix[:, pi] = -1.0

    if len(ground_truth) >= 2:
        result.correctly_separated = len(set(matched_gt_to_pred.values())) >= 2
    return result


# ── Legacy activity-curve episode extraction ────────────────

def extract_episodes_from_activity_curve(
    probs: np.ndarray,
    timestamps: np.ndarray,
    threshold: float = 0.5,
    min_duration_s: float = 0.5,
    merge_gap_s: float = 0.8,
) -> list[Episode]:
    """Legacy binary active/inactive episode extraction."""
    if len(probs) == 0 or len(timestamps) == 0:
        return []
    p = np.asarray(probs, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    active = p >= threshold
    raw_episodes: list[Episode] = []
    in_episode = False
    ep_start = 0.0
    for i in range(len(active)):
        if active[i] and not in_episode:
            ep_start = float(ts[i])
            in_episode = True
        elif not active[i] and in_episode:
            raw_episodes.append(Episode(start_s=ep_start, end_s=float(ts[i]), label="keyboard"))
            in_episode = False
    if in_episode:
        raw_episodes.append(Episode(start_s=ep_start, end_s=float(ts[-1]), label="keyboard"))
    if not raw_episodes:
        return []
    return merge_close_episodes(raw_episodes, merge_gap_s=merge_gap_s, min_duration_s=min_duration_s)


# ── Password-boundary decoding ───────────────────────────────

def smooth_prob_matrix(probs: np.ndarray, smooth_n: int = 3) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or smooth_n <= 1:
        return probs
    out = np.zeros_like(probs)
    kernel = np.ones(smooth_n, dtype=np.float64) / smooth_n
    for c in range(probs.shape[1]):
        out[:, c] = np.convolve(probs[:, c], kernel, mode="same")
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 1e-12] = 1.0
    return out / row_sum



def _mask_to_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(mask) - 1))
    return ranges



def _fill_short_false_gaps(mask: np.ndarray, timestamps: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Bridge short non-password gaps so brief pauses stay inside one episode."""
    mask = np.asarray(mask, dtype=bool).copy()
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(mask) <= 2 or max_gap_s <= 0:
        return mask

    false_ranges = _mask_to_ranges(~mask)
    for s_idx, e_idx in false_ranges:
        left_idx = s_idx - 1
        right_idx = e_idx + 1
        if left_idx < 0 or right_idx >= len(mask):
            continue
        if not mask[left_idx] or not mask[right_idx]:
            continue
        gap_s = float(ts[right_idx] - ts[left_idx])
        if gap_s <= max_gap_s:
            mask[s_idx:e_idx + 1] = True
    return mask



def merge_close_episodes(
    episodes: list[Episode],
    merge_gap_s: float = 0.35,
    min_duration_s: float = 0.4,
) -> list[Episode]:
    if not episodes:
        return []
    episodes = sorted(episodes, key=lambda ep: ep.start_s)
    merged = [episodes[0]]
    for ep in episodes[1:]:
        prev = merged[-1]
        if ep.start_s - prev.end_s <= merge_gap_s:
            merged[-1] = Episode(
                start_s=prev.start_s,
                end_s=max(prev.end_s, ep.end_s),
                label=prev.label or ep.label,
                onset_count=prev.onset_count + ep.onset_count,
                score=max(prev.score, ep.score),
            )
        else:
            merged.append(ep)
    return [ep for ep in merged if ep.duration_s >= min_duration_s]



def decode_password_boundary_predictions(
    probs: np.ndarray,
    timestamps: np.ndarray,
    password_threshold: float = 0.50,
    start_end_threshold: float = 0.30,
    min_duration_s: float = 0.40,
    merge_gap_s: float = 0.60,
    smooth_n: int = 3,
) -> list[Episode]:
    """
    Decode a 4-class probability curve into refined password episodes.

    Key behaviour:
      - password episodes are driven by the combined start/active/end score
      - short non-password dips are bridged, so brief internal pauses do not
        immediately terminate the episode
      - start/end anchors prefer explicit start/end peaks, but fall back to the
        first/last active-like evidence when those peaks are weak
    """
    if len(timestamps) == 0:
        return []

    probs = np.asarray(probs, dtype=np.float64)
    ts = np.asarray(timestamps, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 4:
        raise ValueError("decode_password_boundary_predictions expects shape (N, 4)")

    p = smooth_prob_matrix(probs, smooth_n=smooth_n)
    password_score = p[:, 1] + p[:, 2] + p[:, 3]
    active_like = np.maximum(p[:, 2], np.maximum(p[:, 1], p[:, 3]))
    pred_labels = np.argmax(p, axis=1)

    mask = (password_score >= password_threshold) | (active_like >= max(start_end_threshold, 0.25)) | (pred_labels != 0)
    mask = _fill_short_false_gaps(mask, ts, max_gap_s=merge_gap_s)
    ranges = _mask_to_ranges(mask)
    if not ranges:
        return []

    raw_eps: list[Episode] = []
    for s_idx, e_idx in ranges:
        local_slice = slice(max(0, s_idx - 2), min(len(ts), e_idx + 3))
        local_idx = np.arange(local_slice.start, local_slice.stop)
        local_p = p[local_slice]

        strong_start = local_idx[local_p[:, 1] >= start_end_threshold]
        strong_end = local_idx[local_p[:, 3] >= start_end_threshold]
        active_idx = local_idx[(local_p[:, 2] >= (password_threshold * 0.75)) | (np.argmax(local_p, axis=1) == boundary_label_id("password_active"))]

        if len(strong_start):
            start_idx = int(strong_start[0])
        elif len(active_idx):
            start_idx = int(active_idx[0])
        else:
            start_idx = int(s_idx)

        if len(strong_end):
            end_idx = int(strong_end[-1])
        elif len(active_idx):
            end_idx = int(active_idx[-1])
        else:
            end_idx = int(e_idx)

        start_idx = max(0, min(start_idx, e_idx))
        end_idx = min(len(ts) - 1, max(end_idx, start_idx))
        score_slice = slice(max(0, start_idx), min(len(password_score), end_idx + 1))
        raw_eps.append(Episode(
            start_s=float(ts[start_idx]),
            end_s=float(ts[end_idx]),
            label="password",
            score=float(np.max(password_score[score_slice])) if score_slice.stop > score_slice.start else 0.0,
        ))

    return merge_close_episodes(raw_eps, merge_gap_s=merge_gap_s, min_duration_s=min_duration_s)



def label_sequence_to_password_episodes(
    labels: np.ndarray,
    timestamps: np.ndarray,
    min_duration_s: float = 0.40,
    merge_gap_s: float = 0.60,
) -> list[Episode]:
    """Convert integer boundary labels directly to password episodes with gap bridging."""
    labels = np.asarray(labels, dtype=np.int64)
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(labels) == 0 or len(ts) == 0:
        return []
    mask = labels != boundary_label_id("non_password")
    mask = _fill_short_false_gaps(mask, ts, max_gap_s=merge_gap_s)
    ranges = _mask_to_ranges(mask)
    raw_eps: list[Episode] = []
    for s_idx, e_idx in ranges:
        region = labels[s_idx:e_idx + 1]
        local_start = np.where(region == boundary_label_id("password_start"))[0]
        local_end = np.where(region == boundary_label_id("password_end"))[0]
        local_active = np.where(region == boundary_label_id("password_active"))[0]
        start_idx = s_idx + int(local_start[0]) if len(local_start) else (s_idx + int(local_active[0]) if len(local_active) else s_idx)
        end_idx = s_idx + int(local_end[-1]) if len(local_end) else (s_idx + int(local_active[-1]) if len(local_active) else e_idx)
        if end_idx < start_idx:
            start_idx, end_idx = s_idx, e_idx
        raw_eps.append(Episode(start_s=float(ts[start_idx]), end_s=float(ts[end_idx]), label="password"))
    return merge_close_episodes(raw_eps, merge_gap_s=merge_gap_s, min_duration_s=min_duration_s)


# ── Demo-protocol heuristic retained for backward compatibility ───────

def classify_episodes_by_density(
    episodes: list[Episode],
    onset_times_s: list[float],
    iki_threshold_s: float = 0.6,
    rate_threshold_hz: float = 2.5,
) -> list[Episode]:
    onset_arr = np.array(onset_times_s, dtype=np.float64) if onset_times_s else np.array([])
    classified = []
    for ep in episodes:
        if len(onset_arr) > 0:
            mask = (onset_arr >= ep.start_s) & (onset_arr <= ep.end_s)
            ep_onsets = onset_arr[mask]
        else:
            ep_onsets = np.array([])
        new_ep = Episode(start_s=ep.start_s, end_s=ep.end_s, onset_count=len(ep_onsets))
        keystroke_rate = len(ep_onsets) / max(ep.duration_s, 0.01)
        if len(ep_onsets) >= 2:
            ikis = np.diff(np.sort(ep_onsets))
            median_iki = float(np.median(ikis))
            new_ep.label = "typing_2" if (median_iki > iki_threshold_s or keystroke_rate < rate_threshold_hz) else "typing_1"
        elif len(ep_onsets) == 1:
            new_ep.label = "typing_2"
        else:
            new_ep.label = "typing_1"
        classified.append(new_ep)
    return classified


# ── Onset grouping / alignment ───────────────────────────────

def group_onsets_by_gap(
    onset_times_ns: list[int],
    n_groups: int = 0,
    default_gap_ns: int = 1_500_000_000,
) -> list[list[int]]:
    if not onset_times_ns:
        return []
    sorted_onsets = sorted(int(t) for t in onset_times_ns)
    if len(sorted_onsets) == 1:
        return [sorted_onsets]

    gaps = [(sorted_onsets[i + 1] - sorted_onsets[i], i) for i in range(len(sorted_onsets) - 1)]
    if n_groups > 1 and len(gaps) >= n_groups - 1:
        split_indices = sorted([g[1] for g in sorted(gaps, key=lambda x: -x[0])[:n_groups - 1]])
    else:
        split_indices = [i for gap_ns, i in gaps if gap_ns > default_gap_ns]

    groups: list[list[int]] = []
    prev = 0
    for idx in split_indices:
        groups.append(sorted_onsets[prev:idx + 1])
        prev = idx + 1
    groups.append(sorted_onsets[prev:])
    return groups



def _group_range_ns(group: list[int], pad_ns: int = 300_000_000) -> tuple[int, int]:
    if not group:
        return (0, 0)
    start = min(group)
    end = max(group)
    if start == end:
        start -= pad_ns
        end += pad_ns
    return int(start), int(end)



def align_groups_to_reference(
    predicted_groups: list[list[int]],
    reference_groups: list[list[int]],
    pad_ns: int = 300_000_000,
    max_center_delta_ns: int = 3_000_000_000,
) -> list[list[int]]:
    """
    Align auto-grouped predicted onset groups to reference groups for scoring.

    Grouping itself stays prediction-only; this alignment is only for evaluation.
    """
    if not reference_groups:
        return []
    if not predicted_groups:
        return [[] for _ in reference_groups]

    pred_ranges = [_group_range_ns(g, pad_ns=pad_ns) for g in predicted_groups]
    ref_ranges = [_group_range_ns(g, pad_ns=pad_ns) for g in reference_groups]

    scores = []
    for ri, (rs, re) in enumerate(ref_ranges):
        rc = 0.5 * (rs + re)
        for pi, (ps, pe) in enumerate(pred_ranges):
            pc = 0.5 * (ps + pe)
            inter = max(0, min(re, pe) - max(rs, ps))
            union = max(re, pe) - min(rs, ps)
            iou = inter / union if union > 0 else 0.0
            center_delta = abs(pc - rc)
            center_penalty = center_delta / max_center_delta_ns
            score = iou - 0.05 * center_penalty
            scores.append((score, -center_delta, ri, pi))

    scores.sort(reverse=True)
    taken_pred: set[int] = set()
    aligned: list[list[int]] = [[] for _ in reference_groups]
    assigned_ref: set[int] = set()
    for score, neg_center_delta, ri, pi in scores:
        center_delta = -neg_center_delta
        if ri in assigned_ref or pi in taken_pred:
            continue
        if score <= -0.2 or center_delta > max_center_delta_ns:
            continue
        aligned[ri] = predicted_groups[pi]
        assigned_ref.add(ri)
        taken_pred.add(pi)
    return aligned


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
    lines.append(f"    Precision: {m.precision:.3f}  Recall: {m.recall:.3f}")
    lines.append(f"    Mean IoU: {m.mean_iou:.3f}")
    lines.append(f"    Start boundary error: {m.mean_start_error_ms:.1f}ms")
    lines.append(f"    End boundary error:   {m.mean_end_error_ms:.1f}ms")
    lines.append(f"    2-episode separation: {'✓' if m.correctly_separated else '✗'}")
    return "\n".join(lines)
