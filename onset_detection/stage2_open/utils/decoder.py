"""
Open decoder: frame-level predictions → password groups + onset positions.

No fixed assumptions about number of passwords or password length.
Pure numpy, no torch dependency.

Algorithm:
1. Median-filter the predicted labels for stability.
2. Find all separator (class 2) runs → these split the stream into candidate groups.
3. Within each candidate group, find keystroke (class 1) runs → each run = one onset.
4. Merge tiny gaps, filter spurious detections.
5. Output: list of groups, each with a list of onset positions.
"""
import numpy as np
from scipy.signal import medfilt
from typing import List, Dict


def decode_frame_labels(
    preds: np.ndarray,
    sample_rate: int = 100,
    median_kernel: int = 11,
    min_keystroke_run: int = 2,
    min_separator_run_ms: float = 150.0,
    min_gap_between_onsets_ms: float = 40.0,
    min_group_keys: int = 1,
) -> Dict:
    """
    Decode frame-level class predictions into groups and onsets.

    Args:
        preds: [T] int array, values in {0, 1, 2}
            0 = gap, 1 = keystroke, 2 = separator
        sample_rate: Hz
        median_kernel: smoothing kernel (must be odd)
        min_keystroke_run: minimum consecutive '1' frames to count as an onset
        min_separator_run_ms: minimum separator duration to split groups
        min_gap_between_onsets_ms: merge onsets closer than this
        min_group_keys: discard groups with fewer onsets than this

    Returns:
        {
            'groups': [
                {'start': int, 'end': int, 'onsets': [int, ...], 'num_keys': int},
                ...
            ],
            'num_passwords': int,
            'total_onsets': int,
            'smoothed_preds': np.ndarray,  # for visualization
        }
    """
    T = len(preds)
    if T == 0:
        return {'groups': [], 'num_passwords': 0, 'total_onsets': 0,
                'smoothed_preds': preds}

    min_sep_samples = int(min_separator_run_ms / 1000.0 * sample_rate)
    min_onset_gap = int(min_gap_between_onsets_ms / 1000.0 * sample_rate)

    # 1. Smooth predictions
    smoothed = medfilt(preds.astype(np.float64), kernel_size=median_kernel)
    smoothed = np.round(smoothed).astype(np.int64)

    # 2. Find separator runs
    separators = _find_runs(smoothed, value=2, min_len=min_sep_samples)

    # 3. Split into candidate group regions
    #    Groups are the regions BETWEEN separators (and before first / after last)
    group_regions = []

    if len(separators) == 0:
        # No separators found → entire signal is one group
        group_regions.append((0, T))
    else:
        # Before first separator
        if separators[0][0] > 0:
            group_regions.append((0, separators[0][0]))
        # Between separators
        for i in range(len(separators) - 1):
            g_start = separators[i][1]
            g_end = separators[i + 1][0]
            if g_end > g_start:
                group_regions.append((g_start, g_end))
        # After last separator
        if separators[-1][1] < T:
            group_regions.append((separators[-1][1], T))

    # 4. Within each group region, find keystroke runs → onsets
    groups = []
    for g_start, g_end in group_regions:
        region_preds = smoothed[g_start:g_end]

        # Find keystroke runs
        ks_runs = _find_runs(region_preds, value=1, min_len=min_keystroke_run)

        # Each run's center is an onset position (in global coordinates)
        onsets = []
        for run_start, run_end in ks_runs:
            center = g_start + (run_start + run_end) // 2
            onsets.append(center)

        # Merge onsets that are too close
        onsets = _merge_close(onsets, min_onset_gap)

        if len(onsets) >= min_group_keys:
            groups.append({
                'start': g_start,
                'end': g_end,
                'onsets': onsets,
                'num_keys': len(onsets),
            })

    total_onsets = sum(g['num_keys'] for g in groups)

    return {
        'groups': groups,
        'num_passwords': len(groups),
        'total_onsets': total_onsets,
        'smoothed_preds': smoothed,
    }


def _find_runs(arr: np.ndarray, value: int, min_len: int = 1) -> List[tuple]:
    """Find all runs of `value` in `arr` with length >= min_len.
    Returns list of (start, end) exclusive."""
    runs = []
    in_run = False
    start = 0
    for i in range(len(arr)):
        if arr[i] == value and not in_run:
            start = i
            in_run = True
        elif arr[i] != value and in_run:
            if i - start >= min_len:
                runs.append((start, i))
            in_run = False
    if in_run and len(arr) - start >= min_len:
        runs.append((start, len(arr)))
    return runs


def _merge_close(positions: List[int], min_gap: int) -> List[int]:
    """Merge onset positions that are closer than min_gap."""
    if len(positions) <= 1:
        return positions
    merged = [positions[0]]
    for p in positions[1:]:
        if p - merged[-1] >= min_gap:
            merged.append(p)
        else:
            # Keep the average
            merged[-1] = (merged[-1] + p) // 2
    return merged
