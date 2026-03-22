from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from onset_detection.stage2_segmental.data import build_password_episodes
from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope


IGNORE_KEYS = {
    'shift', 'capslock', 'ctrl', 'alt', 'cmd', 'tab', 'esc',
    'left', 'right', 'up', 'down', 'delete'
}


def normalize_text(s: str) -> str:
    return ''.join(ch.lower() for ch in str(s) if ch.isalnum())


def read_csv_rows(path: str | Path):
    with open(path, 'r', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def valid_attempt_rows(session_prefix: str) -> list[dict[str, Any]]:
    rows = read_csv_rows(session_prefix + '_attempts.csv')
    out = []
    for r in rows:
        prompt = normalize_text(r.get('prompt_text', ''))
        typed = normalize_text(r.get('typed_text', ''))
        match = str(r.get('match') or r.get('match_status') or r.get('status') or '').upper()
        ok = bool(typed) and (match == 'YES' or (not match and (not prompt or prompt == typed)))
        if ok:
            rr = dict(r)
            rr['prompt_text'] = prompt
            rr['typed_text'] = typed
            out.append(rr)
    return out


def read_press_events(session_prefix: str):
    rows = read_csv_rows(session_prefix + '_events.csv')
    out = []
    for r in rows:
        if str(r.get('event_type', '')).lower() != 'press':
            continue
        key = str(r.get('key') or r.get('key_name') or '').lower()
        if key in IGNORE_KEYS:
            continue
        ts = int(r.get('timestamp_ns'))
        out.append((ts, key))
    return out


def read_sensor_array(session_prefix: str):
    rows = read_csv_rows(session_prefix + '_sensor.csv')
    cols = list(rows[0].keys())
    time_col = 'timestamp_ns' if 'timestamp_ns' in cols else cols[0]
    chs = [c for c in cols if c != time_col][:6]
    ts = np.asarray([int(r[time_col]) for r in rows], dtype=np.int64)
    xs = np.asarray([[float(r[c]) for c in chs] for r in rows], dtype=np.float32)
    return ts, xs


def estimate_sample_rate_hz(ts_ns: np.ndarray) -> float:
    if len(ts_ns) < 2:
        return 200.0
    dt = np.diff(ts_ns.astype(np.float64)) / 1e9
    med = float(np.median(dt)) if len(dt) else 0.0
    if med <= 0:
        return 200.0
    return 1.0 / med


def compute_region_length_features(
    raw_imu: np.ndarray,
    ts_ns: np.ndarray,
    feature_mode: str = "no_time",
) -> np.ndarray:
    acc = np.linalg.norm(raw_imu[:, :3], axis=1)
    gyr = np.linalg.norm(raw_imu[:, 3:6], axis=1)
    energy = acc + 0.5 * gyr
    peak_thr = float(np.mean(energy) + 0.5 * np.std(energy))
    peaks = [
        i for i in range(1, len(energy) - 1)
        if energy[i] > peak_thr and energy[i] >= energy[i - 1] and energy[i] >= energy[i + 1]
    ]
    peak_vals = energy[peaks] if len(peaks) else np.asarray([], dtype=np.float32)
    feat = [
        float(np.mean(energy)),
        float(np.std(energy)),
        float(np.max(energy)),
        float(np.percentile(energy, 75)),
        float(np.percentile(energy, 90)),
        float(np.percentile(energy, 95)),
        float(np.mean(np.abs(np.diff(energy)))) if len(energy) >= 2 else 0.0,
        float(np.std(np.diff(energy))) if len(energy) >= 2 else 0.0,
        float(len(peaks)),
    ]
    if len(peak_vals):
        feat += [
            float(np.mean(peak_vals)),
            float(np.std(peak_vals)),
            float(np.max(peak_vals)),
            float(np.percentile(peak_vals, 75)),
            float(np.percentile(peak_vals, 90)),
        ]
        topk = np.sort(peak_vals)[-5:]
        topk = np.pad(topk, (5 - len(topk), 0), mode="constant")
        feat.extend([float(x) for x in topk.tolist()])
    else:
        feat += [0.0] * 10

    # Count macro-peaks under a few smoothing / separation settings.
    # These counts track the latent "number of keystroke-like bumps"
    # without exposing explicit duration / sample-count shortcuts.
    sr = estimate_sample_rate_hz(ts_ns)
    for smooth_s, dist_s in ((0.15, 0.35), (0.15, 0.5), (0.15, 0.7), (0.25, 0.5), (0.35, 0.5)):
        win = max(1, int(round(sr * smooth_s)))
        if win <= 1:
            smoothed = energy.astype(np.float64)
        else:
            kernel = np.ones(win, dtype=np.float64) / float(win)
            smoothed = np.convolve(energy.astype(np.float64), kernel, mode="same")
        sq50, sq90, sq98 = np.quantile(smoothed, [0.50, 0.90, 0.98])
        macro_peaks, _ = find_peaks(
            smoothed,
            distance=max(1, int(round(sr * dist_s))),
            prominence=max(1e-6, (sq90 - sq50) * 0.10),
            height=sq50 + (sq98 - sq50) * 0.05,
        )
        feat.append(float(len(macro_peaks)))

    if feature_mode == "legacy_time":
        duration_s = float((ts_ns[-1] - ts_ns[0]) / 1e9) if len(ts_ns) >= 2 else 0.0
        feat = [
            float(len(raw_imu)),
            float(sr),
            float(duration_s),
            *feat,
        ]
        if len(peaks) >= 2:
            peak_gaps = np.diff([ts_ns[i] for i in peaks]) / 1e9
            feat += [float(np.median(peak_gaps)), float(np.mean(peak_gaps)), float(np.std(peak_gaps))]
        else:
            feat += [0.0, 0.0, 0.0]
    elif feature_mode != "no_time":
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    return np.asarray(feat, dtype=np.float32)


def _cluster_peaks_by_gap(
    peaks: np.ndarray,
    scores: np.ndarray,
    sample_rate: float,
    max_gap_s: float = 2.0,
):
    if len(peaks) == 0:
        return []
    max_gap_frames = max(1, int(round(sample_rate * max_gap_s)))
    groups = []
    cur = [0]
    for i in range(1, len(peaks)):
        if int(peaks[i]) - int(peaks[i - 1]) <= max_gap_frames:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    out = []
    for g in groups:
        idx = np.asarray(g, dtype=np.int64)
        p = peaks[idx]
        s = scores[idx]
        out.append({
            "start_frame": int(p[0]),
            "end_frame": int(p[-1]),
            "num_peaks": int(len(p)),
            "score_sum": float(np.sum(s)),
            "score_mean": float(np.mean(s)),
        })
    return out


def build_length_subregion_from_energy(
    raw_imu: np.ndarray,
    sample_rate: float,
    cluster_gap_s: float = 2.0,
    context_pad_s: float = 1.5,
):
    energy = _compute_energy_envelope(raw_imu, int(round(sample_rate)))
    if len(energy) < 3:
        return None, {"used_cluster_subregion": False, "reason": "short_crop"}

    region = energy.astype(np.float64)
    q50 = float(np.quantile(region, 0.50))
    q90 = float(np.quantile(region, 0.90))
    q98 = float(np.quantile(region, 0.98))
    prominence = max(1e-6, (q90 - q50) * 0.15)
    height = q50 + (q98 - q50) * 0.15
    distance = max(6, int(round(sample_rate * 0.04)))

    peaks = np.asarray([], dtype=np.int64)
    props = {}
    for d_mul, p_mul, h_mul in [(1.0, 1.0, 1.0), (0.5, 0.5, 1.0), (0.33, 0.0, 0.0)]:
        kwargs = {"distance": max(3, int(round(distance * d_mul)))}
        if p_mul > 0:
            kwargs["prominence"] = max(1e-6, prominence * p_mul)
        if h_mul > 0:
            kwargs["height"] = height * h_mul
        peaks, props = find_peaks(region, **kwargs)
        if len(peaks) >= 6:
            break
    if len(peaks) == 0:
        return None, {"used_cluster_subregion": False, "reason": "no_peaks"}

    heights = np.asarray(props.get("peak_heights", region[peaks]), dtype=np.float64)
    scores = heights / max(float(np.max(heights)), 1e-8)
    clusters = _cluster_peaks_by_gap(peaks, scores, sample_rate, max_gap_s=cluster_gap_s)
    if not clusters:
        return None, {"used_cluster_subregion": False, "reason": "no_clusters"}

    best = sorted(clusters, key=lambda x: (x["score_sum"], x["num_peaks"]), reverse=True)[0]
    pad_frames = int(round(sample_rate * context_pad_s))
    lo = max(0, int(best["start_frame"]) - pad_frames)
    hi = min(len(raw_imu) - 1, int(best["end_frame"]) + pad_frames)
    if hi - lo < 3:
        return None, {"used_cluster_subregion": False, "reason": "tiny_cluster_crop"}
    return (lo, hi + 1), {
        "used_cluster_subregion": True,
        "cluster_gap_s": float(cluster_gap_s),
        "context_pad_s": float(context_pad_s),
        "num_raw_peaks": int(len(peaks)),
        "cluster_start_frame": int(best["start_frame"]),
        "cluster_end_frame": int(best["end_frame"]),
        "cluster_num_peaks": int(best["num_peaks"]),
        "cluster_score_sum": float(best["score_sum"]),
        "cluster_score_mean": float(best["score_mean"]),
        "subregion_start_frame": int(lo),
        "subregion_end_frame": int(hi),
    }


def extract_attempt_length_examples(
    session_prefix: str,
    true_len: int,
    pre_margin_ms: float = 300.0,
    post_margin_ms: float = 300.0,
    feature_mode: str = "no_time",
):
    attempts = valid_attempt_rows(session_prefix)
    presses = read_press_events(session_prefix)
    sensor_ts, sensor = read_sensor_array(session_prefix)
    enter_indices = [i for i, (_, k) in enumerate(presses) if k in ('enter', 'return')]
    out_x, out_y = [], []
    start_idx = 0
    pre_ns = int(round(pre_margin_ms * 1e6))
    post_ns = int(round(post_margin_ms * 1e6))
    for att_idx, row in enumerate(attempts):
        if att_idx >= len(enter_indices):
            break
        end_idx = enter_indices[att_idx]
        seq = presses[start_idx:end_idx + 1]
        start_idx = end_idx + 1
        char_ts = [ts for ts, key in seq if key not in ('enter', 'return')]
        if len(char_ts) != true_len:
            continue
        lo = char_ts[0] - pre_ns
        hi = seq[-1][0] + post_ns
        l = np.searchsorted(sensor_ts, lo, side='left')
        r = np.searchsorted(sensor_ts, hi, side='right')
        crop_ts = sensor_ts[max(0, l):min(len(sensor_ts), r)]
        crop_imu = sensor[max(0, l):min(len(sensor), r)]
        if len(crop_imu) < 10:
            continue
        out_x.append(compute_region_length_features(crop_imu, crop_ts, feature_mode=feature_mode))
        out_y.append(int(true_len))
    return out_x, out_y


def extract_mixed_episode_length_examples(
    input_dir: str | Path,
    true_len: int,
    pre_context_ms: float = 1500.0,
    post_context_ms: float = 1500.0,
    feature_mode: str = "no_time",
    use_cluster_subregion: bool = False,
):
    episodes = build_password_episodes(str(input_dir), pre_pad_ms=pre_context_ms, post_pad_ms=post_context_ms, min_len=1)
    out_x, out_y = [], []
    for ep in episodes:
        if len(ep.chars) != int(true_len):
            continue
        raw_imu = ep.imu
        raw_ts = ep.timestamps_ns
        if use_cluster_subregion and len(raw_imu) >= 10:
            sr = estimate_sample_rate_hz(raw_ts)
            subregion, _ = build_length_subregion_from_energy(raw_imu, sr)
            if subregion is not None:
                lo, hi = subregion
                raw_imu = raw_imu[lo:hi]
                raw_ts = raw_ts[lo:hi]
        if len(raw_imu) < 10:
            continue
        out_x.append(compute_region_length_features(raw_imu, raw_ts, feature_mode=feature_mode))
        out_y.append(int(true_len))
    return out_x, out_y


def save_length_model(model, labels: list[int], path: str | Path, feature_mode: str = "no_time"):
    payload = {'model': model, 'labels': list(labels), 'feature_mode': str(feature_mode)}
    with open(path, 'wb') as fh:
        pickle.dump(payload, fh)


def load_length_model(path: str | Path):
    with open(path, 'rb') as fh:
        payload = pickle.load(fh)
    return payload['model'], payload['labels'], {'feature_mode': payload.get('feature_mode', 'legacy_time')}
