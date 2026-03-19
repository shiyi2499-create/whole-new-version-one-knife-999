"""
Episode decoder: frame-level 2-class predictions -> password episodes + onsets.

Phase 1 (episode detection — unchanged):
  frame model predicts typing/silence, then we merge typing runs by time gap.

Phase 2 (onset detection — REDESIGNED):
  Priority order for onset detection inside each episode:
    1. onset_probs  — model-predicted Gaussian impulse probs (dual-head model, NEW)
       These are directly peak-pickable because the model was trained to produce
       a narrow Gaussian at each key center.
    2. energy_env   — raw IMU diff energy heuristic (fallback for old checkpoints)
    3. typing_probs — positive derivative of the typing plateau (last fallback)
    4. uniform      — last resort only

The key insight:
  Previously, onset detection was entirely post-hoc heuristic. The new onset_head
  in the dual-head model predicts a Gaussian-smoothed impulse at each key center,
  giving a direct, learned signal for peak picking. Energy envelope and typing
  prob derivative are kept as fallbacks for backward compatibility with old
  single-head checkpoints.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, medfilt


def decode_episodes(
    preds: np.ndarray,
    raw_imu: Optional[np.ndarray] = None,
    typing_probs: Optional[np.ndarray] = None,
    onset_probs: Optional[np.ndarray] = None,
    sample_rate: int = 100,
    median_kernel: int = 7,
    min_typing_run_ms: float = 30.0,
    episode_gap_ms: float = 600.0,
    min_onset_gap_ms: float = 50.0,
    min_episode_keys: int = 2,
    min_episode_duration_ms: float = 200.0,
) -> Dict:
    """
    Decode 2-class predictions into variable-length password episodes.

    Args:
        preds:        [T] int array with 0=silence, 1=typing.
        raw_imu:      [T, 6] float array. Used as heuristic fallback when
                      onset_probs is not available.
        typing_probs: [T] float array of P(typing). Last-resort fallback.
        onset_probs:  [T] float array of P(onset) from dual-head model onset_head.
                      When provided, this is the PRIMARY signal for onset detection.
    """
    T = len(preds)
    if T == 0:
        return _empty_result(preds)

    min_typing_run = max(1, int(min_typing_run_ms / 1000.0 * sample_rate))
    episode_gap = int(episode_gap_ms / 1000.0 * sample_rate)
    min_onset_gap = int(min_onset_gap_ms / 1000.0 * sample_rate)
    min_episode_dur = int(min_episode_duration_ms / 1000.0 * sample_rate)

    smoothed = medfilt(preds.astype(np.float64), kernel_size=median_kernel)
    smoothed = np.round(smoothed).astype(np.int64)

    typing_runs = _find_runs(smoothed, value=1, min_len=min_typing_run)
    if not typing_runs:
        return _empty_result(smoothed)

    episodes_raw = _merge_runs_into_episodes(typing_runs, episode_gap)

    # Even with onset_probs, energy is still useful as a ranking prior inside
    # an episode: it helps demote broad onset plateaus and noisy ripples.
    energy_env = None
    if raw_imu is not None and len(raw_imu) == T:
        energy_env = _compute_energy_envelope(raw_imu, sample_rate)

    episodes = []
    for ep_start, ep_end, member_runs in episodes_raw:
        if ep_end - ep_start < min_episode_dur:
            continue

        duration_s = max((ep_end - ep_start) / max(sample_rate, 1), 1e-6)
        onsets = _detect_onsets_in_episode(
            ep_start,
            ep_end,
            member_runs,
            onset_probs=onset_probs,
            energy_env=energy_env,
            typing_probs=typing_probs,
            min_onset_gap=min_onset_gap,
            sample_rate=sample_rate,
            duration_s=duration_s,
        )

        if len(onsets) < min_episode_keys:
            continue

        onset_scores = _score_onsets(
            onsets,
            onset_probs=onset_probs,
            energy_env=energy_env,
            typing_probs=typing_probs,
        )
        duration_ms = (ep_end - ep_start) / sample_rate * 1000.0
        episodes.append(
            {
                "start": ep_start,
                "end": ep_end,
                "onsets": onsets,
                "onset_scores": onset_scores,
                "num_keys": len(onsets),
                "duration_ms": duration_ms,
            }
        )

    total_onsets = sum(ep["num_keys"] for ep in episodes)
    return {
        "episodes": episodes,
        "num_episodes": len(episodes),
        "total_onsets": total_onsets,
        "smoothed_preds": smoothed,
        "typing_runs": typing_runs,
    }


def _score_onsets(onsets: List[int], onset_probs: Optional[np.ndarray], energy_env: Optional[np.ndarray], typing_probs: Optional[np.ndarray]) -> List[float]:
    scores = []
    for o in onsets:
        vals = []
        if onset_probs is not None and 0 <= o < len(onset_probs):
            vals.append(float(onset_probs[o]))
        if energy_env is not None and 0 <= o < len(energy_env):
            vals.append(float(energy_env[o]))
        if typing_probs is not None and 0 <= o < len(typing_probs):
            vals.append(float(typing_probs[o]))
        if not vals:
            scores.append(1.0)
        else:
            # Keep onset_probs dominant when available; fall back to average scale otherwise.
            scores.append(float(vals[0] if onset_probs is not None else np.mean(vals)))
    return scores


def _compute_energy_envelope(raw_imu: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Fallback: per-frame transient-energy from raw IMU diff.
    Used only when onset_probs is unavailable (old single-head checkpoint).
    """
    T = raw_imu.shape[0]
    if T < 2:
        return np.zeros(T, dtype=np.float64)

    diff = np.diff(raw_imu, axis=0, prepend=raw_imu[:1, :])
    energy = np.sum(diff ** 2, axis=1)

    win = max(2, int(0.03 * sample_rate))
    kernel = np.ones(win, dtype=np.float64) / win
    smooth = np.convolve(energy, kernel, mode="same")
    return smooth.astype(np.float64)


def _detect_onsets_in_episode(
    ep_start: int,
    ep_end: int,
    member_runs: List[Tuple[int, int]],
    onset_probs: Optional[np.ndarray],
    energy_env: Optional[np.ndarray],
    typing_probs: Optional[np.ndarray],
    min_onset_gap: int,
    sample_rate: int,
    duration_s: float,
) -> List[int]:
    # Priority 1: onset_head model output (directly learned, peak-pickable)
    if onset_probs is not None:
        onsets = _onsets_from_onset_probs(
            onset_probs,
            ep_start,
            ep_end,
            min_onset_gap,
            sample_rate,
            duration_s=duration_s,
            energy_env=energy_env,
        )
        if len(onsets) >= 2:
            return onsets

    # Priority 2: IMU energy envelope (heuristic, fallback for old checkpoints)
    if energy_env is not None:
        onsets = _onsets_from_energy(
            energy_env, ep_start, ep_end, member_runs, min_onset_gap, sample_rate
        )
        if len(onsets) >= 2:
            return onsets

    # Priority 3: typing probability derivative
    if typing_probs is not None:
        onsets = _onsets_from_typing_probs(
            typing_probs, ep_start, ep_end, member_runs, min_onset_gap, sample_rate
        )
        if len(onsets) >= 2:
            return onsets

    # Priority 4: uniform fallback
    return _onsets_uniform_fallback(ep_start, ep_end, min_onset_gap, sample_rate)


EMPIRICAL_KEYS_PER_SEC = 0.43
PEAK_LOG_GAP_SIGMA = 0.35


def _select_peak_subset(peaks: np.ndarray, scores: np.ndarray, sample_rate: int,
                        duration_s: float) -> np.ndarray:
    if len(peaks) <= 2:
        return peaks

    expected = int(round(duration_s * EMPIRICAL_KEYS_PER_SEC))
    expected = int(np.clip(expected, 2, 12))
    lower = max(1, expected - 2)
    upper = min(len(peaks), expected + 2)
    if upper <= lower and len(peaks) <= upper:
        return peaks

    times_s = peaks.astype(np.float64) / max(sample_rate, 1)
    node = np.log(np.clip(scores.astype(np.float64), 1e-8, None))
    target_gap = max(duration_s / max(expected - 1, 1), 1e-3)
    mu = np.log(target_gap)

    best_total = -1e18
    best_path = None
    for k in range(lower, upper + 1):
        dp = np.full((k, len(peaks)), -1e18, dtype=np.float64)
        prev = np.full((k, len(peaks)), -1, dtype=np.int32)
        dp[0, :] = node
        for m in range(1, k):
            for j in range(m, len(peaks)):
                best = -1e18
                best_i = -1
                for i in range(m - 1, j):
                    dt_s = max(times_s[j] - times_s[i], 1e-6)
                    z = (np.log(dt_s) - mu) / PEAK_LOG_GAP_SIGMA
                    trans = -0.5 * z * z
                    cur = dp[m - 1, i] + node[j] + trans
                    if cur > best:
                        best = cur
                        best_i = i
                dp[m, j] = best
                prev[m, j] = best_i
        count_prior = -0.45 * ((k - expected) ** 2)
        end = int(np.argmax(dp[k - 1]))
        total = float(dp[k - 1, end] + count_prior)
        if total > best_total:
            idxs=[end]
            cur=end
            ok=True
            for m in range(k-1,0,-1):
                cur=int(prev[m,cur])
                if cur < 0:
                    ok=False
                    break
                idxs.append(cur)
            if ok:
                best_total=total
                best_path=np.array(list(reversed(idxs)), dtype=np.int64)
    if best_path is None:
        return peaks
    return peaks[best_path]



def _onsets_from_onset_probs(
    onset_probs: np.ndarray,
    ep_start: int,
    ep_end: int,
    min_onset_gap: int,
    sample_rate: int,
    duration_s: float,
    energy_env: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Primary onset detection: peak-pick the model's Gaussian impulse output.

    The onset_head was trained to produce a Gaussian bump (sigma ~ 20ms) at
    each key center. We just find_peaks with appropriate minimum distance.
    No dynamic-range normalization needed — the model's own confidence score
    is meaningful across episodes.
    """
    region = onset_probs[ep_start:ep_end].astype(np.float64)
    ep_len = len(region)
    if ep_len < 3:
        return [(ep_start + ep_end) // 2]

    distance = max(4, min_onset_gap)
    # Light smoothing stabilizes narrow Gaussian bumps without erasing them.
    if ep_len >= 5:
        kernel = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        region_s = np.convolve(region, kernel, mode="same")
    else:
        region_s = region

    q50 = float(np.quantile(region_s, 0.50))
    q90 = float(np.quantile(region_s, 0.90))
    q98 = float(np.quantile(region_s, 0.98))
    prominence = max(0.015, (q90 - q50) * 0.35)
    height_floor = max(0.05, min(0.35, q50 + (q98 - q50) * 0.25))

    peaks, props = find_peaks(
        region_s,
        distance=distance,
        prominence=prominence,
        height=height_floor,
    )
    if len(peaks) == 0:
        peaks, props = find_peaks(
            region_s,
            distance=distance,
            prominence=max(0.01, prominence * 0.5),
        )
    if len(peaks) == 0:
        peaks, props = find_peaks(region_s, distance=distance)
    if len(peaks) >= 1:
        heights = props.get("peak_heights", region_s[peaks])
        scores = np.asarray(heights, dtype=np.float64)
        if energy_env is not None:
            e_region = energy_env[ep_start:ep_end].astype(np.float64)
            e_med = float(np.median(e_region))
            e_hi = float(np.quantile(e_region, 0.95))
            denom = max(e_hi - e_med, 1e-8)
            e_norm = np.clip((e_region[peaks] - e_med) / denom, 0.0, 1.0)
            # onset head drives the decision; energy only nudges the ranking.
            scores = scores * (0.75 + 0.25 * e_norm)

        # Use a sequence-level peak subset selector instead of a raw top-K cap.
        peaks = _select_peak_subset(peaks, scores, sample_rate, duration_s)
        onsets = [ep_start + int(p) for p in peaks]
        return _merge_close(sorted(set(onsets)), min_onset_gap)

    return []


def _onsets_from_energy(
    energy_env: np.ndarray,
    ep_start: int,
    ep_end: int,
    member_runs: List[Tuple[int, int]],
    min_onset_gap: int,
    sample_rate: int,
) -> List[int]:
    region = energy_env[ep_start:ep_end].copy()
    ep_len = len(region)
    if ep_len < 3:
        return [(ep_start + ep_end) // 2]

    energy_max = np.max(region)
    energy_median = np.median(region)
    dynamic_range = energy_max - energy_median
    if dynamic_range < 1e-10:
        return _onsets_from_run_centers(member_runs, min_onset_gap)

    distance = max(3, min_onset_gap)
    prominence = dynamic_range * 0.10
    peaks, _ = find_peaks(region, distance=distance, prominence=prominence)
    if len(peaks) == 0:
        peaks, _ = find_peaks(region, distance=distance, prominence=dynamic_range * 0.05)
    if len(peaks) == 0:
        return _onsets_from_run_centers(member_runs, min_onset_gap)

    onsets = [ep_start + int(p) for p in peaks]
    onsets = _filter_onsets_by_typing_context(
        onsets, member_runs, margin=max(3, int(0.05 * sample_rate))
    )
    if not onsets:
        return _onsets_from_run_centers(member_runs, min_onset_gap)
    return _merge_close(sorted(set(onsets)), min_onset_gap)


def _filter_onsets_by_typing_context(
    onsets: List[int], member_runs: List[Tuple[int, int]], margin: int = 5
) -> List[int]:
    if not member_runs:
        return onsets
    filtered = []
    for o in onsets:
        for rs, re in member_runs:
            if rs - margin <= o <= re + margin:
                filtered.append(o)
                break
    return filtered


def _onsets_from_typing_probs(
    typing_probs: np.ndarray,
    ep_start: int,
    ep_end: int,
    member_runs: List[Tuple[int, int]],
    min_onset_gap: int,
    sample_rate: int,
) -> List[int]:
    """Fallback: use positive derivative peaks of the typing plateau."""
    all_onsets = []
    for run_start, run_end in member_runs:
        run_len = run_end - run_start
        if run_len < 3:
            all_onsets.append((run_start + run_end) // 2)
            continue

        region = typing_probs[run_start:run_end]
        deriv = np.diff(region, prepend=region[0])
        deriv = np.maximum(deriv, 0)

        if np.max(deriv) < 1e-6:
            all_onsets.append((run_start + run_end) // 2)
            continue

        distance = max(3, min_onset_gap)
        prominence = np.max(deriv) * 0.15
        peaks, _ = find_peaks(deriv, distance=distance, prominence=prominence)
        if len(peaks) == 0:
            all_onsets.append((run_start + run_end) // 2)
        else:
            for p in peaks:
                all_onsets.append(run_start + int(p))

    all_onsets = sorted(set(all_onsets))
    return _merge_close(all_onsets, min_onset_gap)


def _onsets_from_run_centers(
    member_runs: List[Tuple[int, int]], min_onset_gap: int
) -> List[int]:
    onsets = [(rs + re) // 2 for rs, re in member_runs]
    return _merge_close(onsets, min_onset_gap)


def _onsets_uniform_fallback(
    ep_start: int, ep_end: int, min_onset_gap: int, sample_rate: int
) -> List[int]:
    """Last resort only."""
    ep_len = ep_end - ep_start
    duration_s = ep_len / sample_rate
    est_keys = max(2, int(round(duration_s * 4.5)))
    if est_keys <= 1:
        return [(ep_start + ep_end) // 2]

    step = ep_len / (est_keys + 1)
    return [ep_start + int(step * (i + 1)) for i in range(est_keys)]


def _empty_result(preds):
    return {
        "episodes": [],
        "num_episodes": 0,
        "total_onsets": 0,
        "smoothed_preds": preds,
        "typing_runs": [],
    }


def _find_runs(arr: np.ndarray, value: int, min_len: int = 1) -> List[Tuple[int, int]]:
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


def _merge_runs_into_episodes(
    typing_runs: List[Tuple[int, int]], max_gap: int
) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    if not typing_runs:
        return []

    episodes = []
    current_start = typing_runs[0][0]
    current_end = typing_runs[0][1]
    current_members = [typing_runs[0]]

    for i in range(1, len(typing_runs)):
        run_start, run_end = typing_runs[i]
        gap = run_start - current_end
        if gap <= max_gap:
            current_end = run_end
            current_members.append(typing_runs[i])
        else:
            episodes.append((current_start, current_end, current_members))
            current_start = run_start
            current_end = run_end
            current_members = [typing_runs[i]]

    episodes.append((current_start, current_end, current_members))
    return episodes


def _merge_close(positions: List[int], min_gap: int) -> List[int]:
    if len(positions) <= 1:
        return positions
    merged = [positions[0]]
    for p in positions[1:]:
        if p - merged[-1] >= min_gap:
            merged.append(p)
        else:
            merged[-1] = (merged[-1] + p) // 2
    return merged


def episodes_to_groups(episodes: List[Dict]) -> List[Dict]:
    return [
        {
            "start": ep["start"],
            "end": ep["end"],
            "onsets": ep["onsets"],
            "num_keys": ep["num_keys"],
        }
        for ep in episodes
    ]
