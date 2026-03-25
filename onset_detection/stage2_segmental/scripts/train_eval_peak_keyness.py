#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_prominences, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope
from onset_detection.stage2_segmental.data import build_password_episodes, estimate_sample_rate_hz
from onset_detection.stage2_segmental.length_model import valid_attempt_rows, read_press_events, read_sensor_array
from phase3_password_inception.run_password_closure_inception import supported_key

IGNORE_KEYS = {
    'shift', 'capslock', 'ctrl', 'alt', 'cmd', 'tab', 'esc',
    'left', 'right', 'up', 'down', 'delete', 'enter', 'return', 'space', 'backspace'
}


def _smooth(values: np.ndarray, win_frames: int) -> np.ndarray:
    if win_frames <= 1:
        return values.astype(np.float64, copy=False)
    kernel = np.ones(int(win_frames), dtype=np.float64) / float(win_frames)
    return np.convolve(values.astype(np.float64), kernel, mode='same')


def _normalize_text(s: str) -> str:
    return ''.join(ch.lower() for ch in str(s) if ch.isalnum())


def _load_password_attempt_episodes(input_dir: str, pre_pad_ms: float = 250.0, post_pad_ms: float = 350.0):
    episodes = []
    for attempts_path in sorted(Path(input_dir).glob('*_attempts.csv')):
        prefix = str(attempts_path)[:-13]
        try:
            attempts = valid_attempt_rows(prefix)
            presses = read_press_events(prefix)
            sensor_ts, sensor = read_sensor_array(prefix)
        except Exception:
            continue
        by_prompt = {}
        for row in attempts:
            idx = int(row.get('prompt_index', 0))
            by_prompt[idx] = row
        sample_rate_hz = estimate_sample_rate_hz(sensor_ts)
        pre_pad_ns = int(round(pre_pad_ms * 1e6))
        post_pad_ns = int(round(post_pad_ms * 1e6))
        session_id = Path(prefix).name
        for prompt_idx, row in sorted(by_prompt.items()):
            start_ns = int(row['attempt_start_ns'])
            end_ns = int(row['submit_ns'])
            prompt = _normalize_text(row.get('prompt_text', ''))
            typed = _normalize_text(row.get('typed_text', ''))
            if not typed or typed != prompt:
                continue
            keys = []
            for ts, key in presses:
                if ts < start_ns or ts > end_ns:
                    continue
                key = str(key).lower()
                if key in IGNORE_KEYS or not supported_key(key):
                    continue
                keys.append((ts, key))
            if len(keys) != len(typed):
                continue
            lo_ns = start_ns - pre_pad_ns
            hi_ns = end_ns + post_pad_ns
            idx = np.where((sensor_ts >= lo_ns) & (sensor_ts <= hi_ns))[0]
            if len(idx) < 8:
                continue
            ep_ts = sensor_ts[idx].astype(np.int64)
            ep_imu = sensor[idx].astype(np.float32)
            key_timestamps = np.asarray([ts for ts, _ in keys], dtype=np.int64)
            key_frames = np.searchsorted(ep_ts, key_timestamps, side='left')
            key_frames = np.clip(key_frames, 0, len(ep_ts) - 1).astype(np.int64)
            episodes.append({
                'session_id': session_id,
                'episode_id': f'{session_id}::attempt{prompt_idx:02d}',
                'password': typed,
                'chars': list(typed),
                'imu': ep_imu,
                'timestamps_ns': ep_ts,
                'key_timestamps_ns': key_timestamps,
                'key_frames': key_frames,
                'sample_rate_hz': sample_rate_hz,
            })
    return episodes


def _load_mixed_episodes(input_dir: str):
    eps = []
    for ep in build_password_episodes(input_dir):
        eps.append({
            'session_id': ep.session_id,
            'episode_id': ep.episode_id,
            'password': ep.password,
            'chars': ep.chars,
            'imu': ep.imu,
            'timestamps_ns': ep.timestamps_ns,
            'key_timestamps_ns': ep.key_timestamps_ns,
            'key_frames': ep.key_frames,
            'sample_rate_hz': ep.sample_rate_hz,
        })
    return eps


def _proposal_env_window(sr: float) -> int:
    # Keep the envelope short enough to preserve slow password tap pulses.
    return max(1, int(round(float(sr) * 0.10)))


def _propose_peaks(ep, smooth_s=0.08, min_dist_s=0.035):
    sr = float(ep['sample_rate_hz'])
    energy = _compute_energy_envelope(ep['imu'], _proposal_env_window(sr)).astype(np.float64)
    sm = _smooth(energy, max(1, int(round(sr * smooth_s))))
    q50, q90 = np.quantile(sm, [0.50, 0.90])
    prominence = max(1e-6, (q90 - q50) * 0.08)
    peaks, props = find_peaks(
        sm,
        distance=max(3, int(round(sr * min_dist_s))),
        prominence=prominence,
    )
    if len(peaks) == 0:
        peaks, props = find_peaks(sm, distance=max(2, int(round(sr * 0.02))))
    return peaks.astype(np.int64), sm, props


def _peak_feature_vector(smoothed: np.ndarray, peaks: np.ndarray, peak_idx: int, sr: float, out_len: int = 48) -> np.ndarray:
    p = int(peaks[peak_idx])
    lo = max(0, p - int(round(sr * 0.20)))
    hi = min(len(smoothed), p + int(round(sr * 0.20)))
    patch = smoothed[lo:hi]
    if len(patch) < 4:
        patch = np.pad(patch, (0, max(0, 4 - len(patch))), mode='edge')
    patch = resample(patch, out_len)
    if np.iscomplexobj(patch):
        patch = np.real(patch)
    patch = np.asarray(patch, dtype=np.float64)
    patch -= patch.min()
    mx = float(np.max(patch))
    if mx > 1e-8:
        patch /= mx
    left_gap = float((p - peaks[peak_idx - 1]) / sr) if peak_idx > 0 else 1.0
    right_gap = float((peaks[peak_idx + 1] - p) / sr) if peak_idx + 1 < len(peaks) else 1.0
    promin = peak_prominences(smoothed, np.asarray([p]))[0][0] if len(smoothed) >= 3 else 0.0
    stats = np.array([
        float(smoothed[p]),
        float(promin),
        left_gap,
        right_gap,
        float((left_gap + right_gap) * 0.5),
    ], dtype=np.float32)
    return np.concatenate([stats, patch.astype(np.float32)], axis=0)


def _assign_peak_labels(peaks: np.ndarray, peak_ts_ns: np.ndarray, gt_ts_ns: np.ndarray, tol_ms: float = 80.0):
    tol_ns = tol_ms * 1e6
    labels = np.zeros(len(peaks), dtype=np.int64)
    matched_gt = np.full(len(gt_ts_ns), -1, dtype=np.int64)
    for gi, g in enumerate(gt_ts_ns):
        if len(peak_ts_ns) == 0:
            break
        j = int(np.argmin(np.abs(peak_ts_ns - g)))
        if abs(int(peak_ts_ns[j]) - int(g)) <= tol_ns:
            labels[j] = 1
            matched_gt[gi] = j
    return labels, matched_gt


def _build_dataset(episodes):
    X=[]; y=[]; meta=[]
    for ep in episodes:
        peaks, sm, _ = _propose_peaks(ep)
        peak_ts = ep['timestamps_ns'][np.clip(peaks, 0, len(ep['timestamps_ns'])-1)] if len(peaks) else np.asarray([], dtype=np.int64)
        labels, matched = _assign_peak_labels(peaks, peak_ts, ep['key_timestamps_ns'])
        for i in range(len(peaks)):
            feat = _peak_feature_vector(sm, peaks, i, ep['sample_rate_hz'])
            X.append(feat)
            y.append(int(labels[i]))
            meta.append({
                'session_id': ep['session_id'],
                'episode_id': ep['episode_id'],
                'peak_frame': int(peaks[i]),
                'label': int(labels[i]),
            })
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), meta


def _select_k_peaks(peaks: np.ndarray, scores: np.ndarray, k: int, sr: float, gap_prior_s: float = 1.3):
    if len(peaks) == 0:
        return np.asarray([], dtype=np.int64)
    if len(peaks) <= k:
        return peaks.astype(np.int64)
    times_s = peaks.astype(np.float64) / max(sr, 1.0)
    node = np.log(np.clip(scores.astype(np.float64), 1e-8, 1.0))
    mu = np.log(max(gap_prior_s, 1e-3))
    sigma = 0.45
    dp = np.full((k, len(peaks)), -1e18, dtype=np.float64)
    prev = np.full((k, len(peaks)), -1, dtype=np.int32)
    dp[0, :] = node
    for m in range(1, k):
        for j in range(m, len(peaks)):
            best=-1e18; best_i=-1
            for i in range(m-1, j):
                dt = max(times_s[j]-times_s[i], 1e-6)
                z = (np.log(dt)-mu)/sigma
                cur = dp[m-1, i] + node[j] - 0.5*z*z
                if cur > best:
                    best = cur; best_i = i
            dp[m,j]=best; prev[m,j]=best_i
    end = int(np.argmax(dp[k-1]))
    idxs=[end]; cur=end
    for m in range(k-1,0,-1):
        cur=int(prev[m,cur])
        if cur<0: break
        idxs.append(cur)
    idxs=np.array(list(reversed(idxs)), dtype=np.int64)
    if len(idxs) != k:
        order=np.argsort(-scores)[:k]
        return np.sort(peaks[order])
    return peaks[idxs]


def _evaluate_peak_model(model, episodes):
    rows=[]
    for ep in episodes:
        peaks, sm, _ = _propose_peaks(ep)
        if len(peaks) == 0:
            rows.append({'episode_id': ep['episode_id'], 'session_id': ep['session_id'], 'peak_top1': 0.0, 'peak_recall': 0.0, 'peak_precision': 0.0, 'num_candidates': 0, 'num_gt': len(ep['key_frames'])})
            continue
        X = np.stack([_peak_feature_vector(sm, peaks, i, ep['sample_rate_hz']) for i in range(len(peaks))]).astype(np.float32)
        probs = model.predict_proba(X)[:,1]
        chosen = _select_k_peaks(peaks, probs, len(ep['key_frames']), ep['sample_rate_hz'])
        peak_ts = ep['timestamps_ns'][np.clip(chosen, 0, len(ep['timestamps_ns'])-1)]
        labels,_ = _assign_peak_labels(chosen, peak_ts, ep['key_timestamps_ns'])
        tp = int(labels.sum())
        recall = tp / max(1, len(ep['key_frames']))
        precision = tp / max(1, len(chosen))
        exact = 1.0 if tp == len(ep['key_frames']) and len(chosen) == len(ep['key_frames']) else 0.0
        rows.append({'episode_id': ep['episode_id'], 'session_id': ep['session_id'], 'peak_top1': exact, 'peak_recall': recall, 'peak_precision': precision, 'num_candidates': int(len(peaks)), 'num_gt': int(len(ep['key_frames']))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--password_dirs', nargs='*', default=[])
    ap.add_argument('--mixed_dirs', nargs='*', default=[])
    ap.add_argument('--output_dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes=[]
    for d in args.password_dirs:
        episodes.extend(_load_password_attempt_episodes(d))
    for d in args.mixed_dirs:
        episodes.extend(_load_mixed_episodes(d))
    episodes.sort(key=lambda x: (x['session_id'], x['episode_id']))
    session_ids = sorted({ep['session_id'] for ep in episodes})

    X, y, meta = _build_dataset(episodes)
    summary = {
        'num_peak_candidates': int(len(X)),
        'num_positive': int(np.sum(y==1)),
        'num_negative': int(np.sum(y==0)),
        'num_sessions': len(session_ids),
        'sessions': session_ids,
    }
    with open(out_dir / 'dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    cv=[]; all_rows=[]
    for holdout in session_ids:
        train_mask = np.array([m['session_id'] != holdout for m in meta], dtype=bool)
        val_mask = ~train_mask
        if len(np.unique(y[train_mask])) < 2:
            continue
        model = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42)
        model.fit(X[train_mask], y[train_mask])
        probs = model.predict_proba(X[val_mask])[:,1]
        pred = (probs >= 0.5).astype(np.int64)
        cv.append({
            'holdout_session': holdout,
            'candidate_accuracy': float(accuracy_score(y[val_mask], pred)),
            'candidate_auc': float(roc_auc_score(y[val_mask], probs)) if len(np.unique(y[val_mask])) > 1 else None,
            'num_val_candidates': int(np.sum(val_mask)),
            'pos_rate_true': float(np.mean(y[val_mask])),
            'pos_rate_pred': float(np.mean(pred)),
        })
        val_eps = [ep for ep in episodes if ep['session_id'] == holdout]
        all_rows.extend(_evaluate_peak_model(model, val_eps))

    report = {
        'mode': 'peak_keyness_within_password_segment',
        'candidate_cv': cv,
        'episode_metrics': {
            'num_episodes': len(all_rows),
            'exact_all_keys': float(np.mean([r['peak_top1'] for r in all_rows])) if all_rows else 0.0,
            'mean_peak_recall': float(np.mean([r['peak_recall'] for r in all_rows])) if all_rows else 0.0,
            'mean_peak_precision': float(np.mean([r['peak_precision'] for r in all_rows])) if all_rows else 0.0,
        },
    }
    with open(out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'episode_rows.json', 'w', encoding='utf-8') as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
