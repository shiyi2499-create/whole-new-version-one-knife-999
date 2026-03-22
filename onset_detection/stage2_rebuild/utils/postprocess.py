"""
Post-processing utilities (pure numpy, no torch dependency).

These functions are used by both the models (at inference time) and the
sanity tests (without torch).
"""
import numpy as np
from scipy.signal import find_peaks, medfilt
from typing import List, Tuple, Optional


# ============================================================
# Stage 2A: Group extraction from probability signal
# ============================================================

def extract_groups_from_probs(
    probs: np.ndarray,
    sample_rate: int = 100,
    median_kernel: int = 21,
    threshold: float = 0.5,
    min_group_duration_s: float = 0.8,
    expected_groups: int = 5,
) -> List[Tuple[int, int]]:
    """
    Post-process frame-wise typing probabilities into group boundaries.

    Args:
        probs: [T] array of typing probabilities in [0, 1]
        sample_rate: Hz
        median_kernel: median filter kernel size (must be odd)
        threshold: binarization threshold
        min_group_duration_s: minimum group duration in seconds
        expected_groups: target number of groups

    Returns:
        list of (start, end) sample indices, sorted by start time
    """
    min_duration = int(min_group_duration_s * sample_rate)

    # 1. Median filter
    if median_kernel > 1:
        smoothed = medfilt(probs, kernel_size=median_kernel)
    else:
        smoothed = probs.copy()

    # 2. Try multiple thresholds
    best_groups = None
    best_diff = float('inf')

    for thresh in [threshold, 0.4, 0.3, 0.6, 0.2, 0.7]:
        binary = (smoothed >= thresh).astype(np.int32)
        groups = _extract_connected_components(binary, min_duration)

        diff = abs(len(groups) - expected_groups)
        if diff < best_diff:
            best_diff = diff
            best_groups = groups
            if diff == 0:
                break

    if best_groups is None:
        best_groups = []

    # 3. Merge if too many
    while len(best_groups) > expected_groups and len(best_groups) > 1:
        min_gap = float('inf')
        merge_idx = 0
        for i in range(len(best_groups) - 1):
            gap = best_groups[i + 1][0] - best_groups[i][1]
            if gap < min_gap:
                min_gap = gap
                merge_idx = i
        merged = (best_groups[merge_idx][0], best_groups[merge_idx + 1][1])
        best_groups = best_groups[:merge_idx] + [merged] + best_groups[merge_idx + 2:]

    # 4. Split if too few
    while len(best_groups) < expected_groups and len(best_groups) > 0:
        durations = [end - start for start, end in best_groups]
        longest_idx = int(np.argmax(durations))
        start, end = best_groups[longest_idx]

        if end - start > 2 * min_duration:
            segment_probs = smoothed[start:end]
            if len(segment_probs) > 20:
                search_start = len(segment_probs) // 5
                search_end = 4 * len(segment_probs) // 5
                search_region = segment_probs[search_start:search_end]
                if len(search_region) > 0:
                    split_point = int(np.argmin(search_region)) + search_start + start
                else:
                    split_point = (start + end) // 2
            else:
                split_point = (start + end) // 2

            best_groups = (best_groups[:longest_idx] +
                           [(start, split_point), (split_point, end)] +
                           best_groups[longest_idx + 1:])
        else:
            break

    return best_groups


def _extract_connected_components(binary: np.ndarray,
                                  min_duration: int) -> List[Tuple[int, int]]:
    """Extract runs of 1s from binary array."""
    groups = []
    in_group = False
    start = 0

    for i in range(len(binary)):
        if binary[i] == 1 and not in_group:
            start = i
            in_group = True
        elif binary[i] == 0 and in_group:
            if i - start >= min_duration:
                groups.append((start, i))
            in_group = False

    if in_group and len(binary) - start >= min_duration:
        groups.append((start, len(binary)))

    return groups


# ============================================================
# Stage 2B: Peak picking from onset probability signal
# ============================================================

def pick_onset_peaks(
    probs: np.ndarray,
    expected_onsets: int = 8,
    min_iki_samples: int = 5,
    base_threshold: float = 0.3,
    fallback_thresholds: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Constrained peak picking: extract exactly expected_onsets peaks.

    Strategy:
    1. Find peaks above threshold with min distance constraint
    2. If too many: take top-K by height
    3. If too few: progressively lower threshold, then interpolate

    Returns:
        np.ndarray of onset sample indices, sorted
    """
    if fallback_thresholds is None:
        fallback_thresholds = [0.2, 0.15, 0.1, 0.05, 0.02]

    all_thresholds = [base_threshold] + fallback_thresholds
    best_peaks = np.array([], dtype=np.int64)
    best_heights = np.array([])
    best_diff = float('inf')

    for thresh in all_thresholds:
        peaks, props = find_peaks(probs, height=thresh, distance=min_iki_samples)
        diff = abs(len(peaks) - expected_onsets)
        if diff < best_diff:
            best_diff = diff
            best_peaks = peaks
            best_heights = props['peak_heights'] if len(peaks) > 0 else np.array([])
        if len(peaks) == expected_onsets:
            return np.sort(peaks)

    # Too many → take top-K
    if len(best_peaks) > expected_onsets:
        top_idx = np.argsort(best_heights)[-expected_onsets:]
        return np.sort(best_peaks[top_idx])

    # Too few → interpolate
    if len(best_peaks) < expected_onsets:
        return _interpolate_peaks(probs, best_peaks, expected_onsets, min_iki_samples)

    return np.sort(best_peaks)


def _interpolate_peaks(probs: np.ndarray,
                       detected: np.ndarray,
                       target: int,
                       min_iki: int) -> np.ndarray:
    """Fill in missing peaks via gap splitting at local maxima."""
    T = len(probs)

    if len(detected) == 0:
        positions = np.linspace(min_iki, T - min_iki, target).astype(np.int64)
        snapped = []
        for p in positions:
            ws = max(0, p - min_iki // 2)
            we = min(T, p + min_iki // 2 + 1)
            snapped.append(ws + int(np.argmax(probs[ws:we])))
        return np.array(sorted(set(snapped)), dtype=np.int64)[:target]

    peaks = sorted(detected.tolist())

    while len(peaks) < target:
        extended = [0] + peaks + [T - 1]
        gaps = [(extended[i + 1] - extended[i], extended[i], extended[i + 1])
                for i in range(len(extended) - 1)]
        gaps.sort(reverse=True)

        inserted = False
        for gap_size, gs, ge in gaps:
            if gap_size <= min_iki * 2:
                continue
            ms = gs + min_iki
            me = ge - min_iki
            if ms >= me:
                ms = (gs + ge) // 2
                me = ms + 1
            seg = probs[ms:me]
            if len(seg) > 0:
                peaks.append(ms + int(np.argmax(seg)))
                peaks.sort()
                inserted = True
                break

        if not inserted:
            remaining = target - len(peaks)
            if remaining > 0:
                extra = np.linspace(0, T - 1, target).astype(np.int64)
                peaks = sorted(set(peaks + extra.tolist()))
            break

    return np.array(peaks[:target], dtype=np.int64)
