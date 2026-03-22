#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks, resample
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, roc_auc_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader
from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope
from onset_detection.stage2_segmental.data import build_password_episodes, estimate_sample_rate_hz
from onset_detection.stage2_segmental.length_model import load_length_model
from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from onset_detection.stage2_segmental.scripts.eval_fullstream_clusterfirst import (
    _evaluate_fixed_from_frames,
    _evaluate_overlap_from_frames,
)
from onset_detection.stage2_segmental.scripts.train_eval_peak_keyness import (
    _build_dataset as _build_peak_dataset,
    _load_password_attempt_episodes,
    _peak_feature_vector,
    _propose_peaks,
    _select_k_peaks,
    _smooth,
)


def resolve_device(name: str) -> torch.device:
    req = name.lower()
    if req == 'auto':
        if torch.cuda.is_available():
            req = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            req = 'mps'
        else:
            req = 'cpu'
    return torch.device(req)


def _normalized_curve(values: np.ndarray, out_len: int = 96) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) < 4:
        vals = np.pad(vals, (0, max(0, 4 - len(vals))), mode='edge')
    curve = resample(vals, out_len)
    if np.iscomplexobj(curve):
        curve = np.real(curve)
    curve = np.asarray(curve, dtype=np.float32)
    curve -= float(curve.min())
    mx = float(curve.max())
    if mx > 1e-8:
        curve /= mx
    return curve


def _segment_imu_shape(crop_imu: np.ndarray, out_len: int = 64) -> np.ndarray:
    seg = np.asarray(crop_imu, dtype=np.float64)
    if len(seg) < 4:
        seg = np.pad(seg, ((0, max(0, 4 - len(seg))), (0, 0)), mode='edge')
    seg = resample(seg, out_len, axis=0)
    if np.iscomplexobj(seg):
        seg = np.real(seg)
    seg = np.asarray(seg, dtype=np.float32)
    seg = seg - np.mean(seg, axis=0, keepdims=True)
    std = np.std(seg, axis=0, keepdims=True)
    seg = seg / np.maximum(std, 1e-6)
    return seg.reshape(-1).astype(np.float32)


def _peak_keyness_profile(peak_frames: np.ndarray, peak_probs: np.ndarray, length: int, out_len: int = 48) -> np.ndarray:
    profile = np.zeros(max(length, 1), dtype=np.float32)
    if len(peak_frames):
        idx = np.clip(peak_frames.astype(np.int64), 0, len(profile) - 1)
        np.maximum.at(profile, idx, peak_probs.astype(np.float32))
    profile = _smooth(profile.astype(np.float64), max(1, len(profile) // 32)).astype(np.float32)
    return _normalized_curve(profile, out_len=out_len)


def _iki_regularity(peaks: np.ndarray, sample_rate_hz: float) -> float:
    if len(peaks) < 3:
        return 0.0
    peak_times = peaks.astype(np.float64) / max(sample_rate_hz, 1.0)
    ikis = np.diff(peak_times)
    mean_iki = float(np.mean(ikis))
    if mean_iki <= 1e-8:
        return 0.0
    cv = float(np.std(ikis) / mean_iki)
    return float(np.exp(-cv / 0.6))


def _cluster_macro_peaks(peaks: np.ndarray, scores: np.ndarray, sample_rate: float, gap_s: float):
    if len(peaks) == 0:
        return []
    max_gap_frames = max(1, int(round(sample_rate * gap_s)))
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
            'start_frame': int(p[0]),
            'end_frame': int(p[-1]),
            'num_peaks': int(len(p)),
            'score_sum': float(np.sum(s)),
            'score_mean': float(np.mean(s)),
            'peaks': p.astype(np.int64),
            'scores': s.astype(np.float64),
        })
    return out


def _propose_candidates_fullstream(imu: np.ndarray, ts: np.ndarray, sample_rate_hz: float):
    energy_raw = _compute_energy_envelope(imu, int(round(sample_rate_hz))).astype(np.float64)
    seen = set()
    candidates = []
    for smooth_s in (0.10, 0.15, 0.22, 0.30, 0.40):
        smoothed = _smooth(energy_raw, max(1, int(round(sample_rate_hz * smooth_s))))
        q50, q90, q98 = np.quantile(smoothed, [0.50, 0.90, 0.98])
        prominence = max(1e-6, (q90 - q50) * 0.08)
        height = q50 + (q98 - q50) * 0.03
        for dist_s in (0.25, 0.35, 0.50, 0.70, 0.90):
            peaks, props = find_peaks(
                smoothed,
                distance=max(1, int(round(sample_rate_hz * dist_s))),
                prominence=prominence,
                height=height,
            )
            if len(peaks) == 0:
                continue
            heights = np.asarray(props.get('peak_heights', smoothed[peaks]), dtype=np.float64)
            peak_scores = heights / max(float(np.max(heights)), 1e-8)
            for gap_s in (0.85, 1.1, 1.35, 1.7, 2.1):
                for cluster in _cluster_macro_peaks(peaks, peak_scores, sample_rate_hz, gap_s=gap_s):
                    if not (3 <= cluster['num_peaks'] <= 18):
                        continue
                    pad_frames = int(round(sample_rate_hz * 0.80))
                    lo = max(0, int(cluster['start_frame']) - pad_frames)
                    hi = min(len(imu), int(cluster['end_frame']) + pad_frames + 1)
                    key = (lo, hi, cluster['num_peaks'])
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        'crop_start': int(lo),
                        'crop_end': int(hi),
                        'cluster_num_peaks': int(cluster['num_peaks']),
                        'cluster_score_mean': float(cluster['score_mean']),
                        'cluster_peaks': cluster['peaks'].astype(np.int64),
                        'smooth_s': float(smooth_s),
                        'dist_s': float(dist_s),
                        'gap_s': float(gap_s),
                    })
    return candidates


def _candidate_label(candidate: dict, gt_eps_in_session: list) -> tuple[float, dict]:
    a0 = float(candidate['crop_start_ns']) * 1e-9
    a1 = float(candidate['crop_end_ns']) * 1e-9
    best = {'iou': 0.0, 'key_frac': 0.0, 'quality': 0.0, 'matched_episode_id': None}
    for ep in gt_eps_in_session:
        b0 = float(ep.key_timestamps_ns[0]) * 1e-9
        b1 = float(ep.key_timestamps_ns[-1]) * 1e-9
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        union = max(a1, b1) - min(a0, b0)
        iou = inter / union if union > 1e-8 else 0.0
        inside = np.sum((ep.key_timestamps_ns >= candidate['crop_start_ns']) & (ep.key_timestamps_ns <= candidate['crop_end_ns']))
        key_frac = float(inside) / float(len(ep.key_timestamps_ns)) if len(ep.key_timestamps_ns) else 0.0
        quality = 0.65 * iou + 0.35 * key_frac
        if quality > best['quality']:
            best = {'iou': float(iou), 'key_frac': float(key_frac), 'quality': float(quality), 'matched_episode_id': ep.episode_id}
    return float(best['quality']), best


def _train_peak_model(train_peak_episodes):
    normalized = []
    for ep in train_peak_episodes:
        if isinstance(ep, dict):
            normalized.append(ep)
        else:
            normalized.append({
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
    X, y, _ = _build_peak_dataset(normalized)
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42,
    )
    model.fit(X, y)
    return model


def _candidate_feature_vector(candidate: dict, crop_imu: np.ndarray, crop_ts: np.ndarray, sr: float, peak_model, length_model=None) -> np.ndarray:
    energy = _compute_energy_envelope(crop_imu, int(round(sr))).astype(np.float64)
    smoothed = _smooth(energy, max(1, int(round(sr * 0.10))))
    peaks, sm, _ = _propose_peaks({'imu': crop_imu, 'sample_rate_hz': sr, 'timestamps_ns': crop_ts})
    peak_probs = np.zeros(len(peaks), dtype=np.float32)
    if len(peaks):
        Xp = np.stack([_peak_feature_vector(sm, peaks, i, sr) for i in range(len(peaks))]).astype(np.float32)
        peak_probs = peak_model.predict_proba(Xp)[:, 1].astype(np.float32)
    profile = _peak_keyness_profile(peaks, peak_probs, len(sm), out_len=48)
    energy_shape = _normalized_curve(smoothed, out_len=64)
    imu_shape = _segment_imu_shape(crop_imu, out_len=64)
    hi = float(np.mean(peak_probs >= 0.8)) if len(peak_probs) else 0.0
    mid = float(np.mean(peak_probs >= 0.5)) if len(peak_probs) else 0.0
    top3 = float(np.mean(np.sort(peak_probs)[-min(3, len(peak_probs)):])) if len(peak_probs) else 0.0
    top5 = float(np.mean(np.sort(peak_probs)[-min(5, len(peak_probs)):])) if len(peak_probs) else 0.0
    mass = float(np.sum(peak_probs))
    n_pk = float(len(peaks))
    regularity = _iki_regularity(peaks, sr)
    span_s = max(float(len(crop_imu)) / max(sr, 1.0), 1e-3)
    density = n_pk / span_s
    base = np.array([
        float(candidate['cluster_num_peaks']),
        float(candidate['cluster_score_mean']),
        n_pk,
        mass,
        top3,
        top5,
        hi,
        mid,
        regularity,
        density,
    ], dtype=np.float32)
    return np.concatenate([base, profile.astype(np.float32), energy_shape.astype(np.float32), imu_shape], axis=0)


def _recover_from_candidate(candidate: dict, full_imu: np.ndarray, full_ts: np.ndarray, sr: float, peak_model, length_model, classifier, overlap, device, ep_eval):
    lo = int(candidate['crop_start'])
    hi = int(candidate['crop_end'])
    crop_imu = full_imu[lo:hi]
    crop_ts = full_ts[lo:hi]
    peaks, sm, _ = _propose_peaks({'imu': crop_imu, 'sample_rate_hz': sr, 'timestamps_ns': crop_ts})
    if len(peaks) == 0:
        return None, None, {'error': 'no_peaks'}
    Xp = np.stack([_peak_feature_vector(sm, peaks, i, sr) for i in range(len(peaks))]).astype(np.float32)
    peak_probs = peak_model.predict_proba(Xp)[:, 1].astype(np.float32)
    if length_model is not None:
        model, labels, meta = length_model
        from onset_detection.stage2_segmental.length_model import compute_region_length_features
        feat = compute_region_length_features(crop_imu, crop_ts, feature_mode=str(meta.get('feature_mode', 'no_time'))).reshape(1, -1)
        k = int(model.predict(feat)[0])
    else:
        k = len(ep_eval.chars)
    chosen = _select_k_peaks(peaks, peak_probs, k, sr, gap_prior_s=1.3)
    pred_global_frames = chosen + lo
    pred_global_ts = full_ts[np.clip(pred_global_frames, 0, len(full_ts)-1)]
    local_frames = np.searchsorted(ep_eval.timestamps_ns, pred_global_ts, side='left')
    local_frames = np.clip(local_frames, 0, len(ep_eval.timestamps_ns)-1).astype(np.int64)
    if len(local_frames) != len(ep_eval.chars):
        xs = np.linspace(0, len(local_frames)-1, len(ep_eval.chars))
        local_frames = np.interp(xs, np.arange(len(local_frames)), local_frames.astype(np.float64)).round().astype(np.int64)
    base = _evaluate_fixed_from_frames(classifier, ep_eval, local_frames)
    ov, ov_debug = _evaluate_overlap_from_frames(overlap, ep_eval, local_frames, device)
    dbg = {
        'pred_len': int(k),
        'peak_probs_top': np.sort(peak_probs)[-min(8, len(peak_probs)):].tolist(),
        'mapped_local_frames': local_frames.tolist(),
        'chosen_global_frames': pred_global_frames.tolist(),
    }
    if ov_debug is not None:
        dbg['overlap_debug'] = ov_debug
    return base, ov, dbg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--password_dirs', nargs='*', default=[])
    ap.add_argument('--train_mixed_dirs', nargs='+', required=True)
    ap.add_argument('--eval_dirs', nargs='+', required=True)
    ap.add_argument('--classifier_checkpoint', required=True)
    ap.add_argument('--classifier_scaler', required=True)
    ap.add_argument('--overlap_checkpoint', required=True)
    ap.add_argument('--length_model', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    password_peak_eps = []
    for d in args.password_dirs:
        password_peak_eps.extend(_load_password_attempt_episodes(d))

    train_mixed_eps = []
    for d in args.train_mixed_dirs:
        train_mixed_eps.extend(build_password_episodes(d))
    eval_eps = []
    for d in args.eval_dirs:
        eval_eps.extend(build_password_episodes(d))
    classifier = load_external_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    classifier.eval()
    overlap = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap.eval()
    overlap.freeze_classifier(True)
    length_model = load_length_model(args.length_model)

    baseline_rows = []
    overlap_rows = []
    debug_rows = []
    train_peak_eps = password_peak_eps + train_mixed_eps
    peak_model = _train_peak_model(train_peak_eps)

    by_train_session = {}
    for ep in train_mixed_eps:
        by_train_session.setdefault(ep.session_id, []).append(ep)

    X = []
    y = []
    for session_id, session_eps in by_train_session.items():
        loader = SessionLoader(session_eps[0].session_path)
        ts, imu = loader.get_imu()
        sr = estimate_sample_rate_hz(ts)
        cands = _propose_candidates_fullstream(imu, ts, sr)
        for cand in cands:
            crop_imu = imu[int(cand['crop_start']):int(cand['crop_end'])]
            crop_ts = ts[int(cand['crop_start']):int(cand['crop_end'])]
            cand = dict(cand)
            cand['crop_start_ns'] = int(ts[min(int(cand['crop_start']), len(ts)-1)])
            cand['crop_end_ns'] = int(ts[min(max(int(cand['crop_end'])-1,0), len(ts)-1)])
            feat = _candidate_feature_vector(cand, crop_imu, crop_ts, sr, peak_model, length_model)
            quality, _label_dbg = _candidate_label(cand, session_eps)
            X.append(feat)
            y.append(float(_label_dbg.get('key_frac', 0.0)))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ranker = RandomForestRegressor(n_estimators=500, max_depth=12, min_samples_leaf=2, random_state=42)
    ranker.fit(X, y)
    train_pred = ranker.predict(X)

    by_eval_session = {}
    for ep in eval_eps:
        by_eval_session.setdefault(ep.session_id, []).append(ep)

    fold_reports = [{
        'mode': 'train_on_train_mixed_only',
        'num_train_sessions': int(len(by_train_session)),
        'num_eval_sessions': int(len(by_eval_session)),
        'train_candidate_r2': float(r2_score(y, train_pred)) if len(X) else None,
        'num_train_candidates': int(len(X)),
        'num_train_positive_like': int(np.sum(y >= 0.5)),
    }]

    for holdout, holdout_eps in sorted(by_eval_session.items()):
        loader = SessionLoader(holdout_eps[0].session_path)
        ts, imu = loader.get_imu()
        sr = estimate_sample_rate_hz(ts)
        cands = _propose_candidates_fullstream(imu, ts, sr)
        scored = []
        for cand in cands:
            crop_imu = imu[int(cand['crop_start']):int(cand['crop_end'])]
            crop_ts = ts[int(cand['crop_start']):int(cand['crop_end'])]
            cand = dict(cand)
            cand['crop_start_ns'] = int(ts[min(int(cand['crop_start']), len(ts)-1)])
            cand['crop_end_ns'] = int(ts[min(max(int(cand['crop_end'])-1,0), len(ts)-1)])
            feat = _candidate_feature_vector(cand, crop_imu, crop_ts, sr, peak_model, length_model).reshape(1, -1)
            score = float(ranker.predict(feat)[0])
            quality, label_dbg = _candidate_label(cand, holdout_eps)
            scored.append((score, quality, cand, label_dbg))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_candidates = []
        for score, quality, cand, label_dbg in scored[:5]:
            top_candidates.append({'rank_score': float(score), 'label_quality': float(quality), **label_dbg, **cand})
        for ep in holdout_eps:
            if not scored:
                debug_rows.append({'session_id': holdout, 'episode_id': ep.episode_id, 'error': 'no_candidates'})
                continue
            best_score, _best_quality, best_cand, best_label_dbg = scored[0]
            base, ov, rec_dbg = _recover_from_candidate(best_cand, imu, ts, sr, peak_model, length_model, classifier, overlap, device, ep)
            if base is None or ov is None:
                debug_rows.append({'session_id': holdout, 'episode_id': ep.episode_id, 'error': 'recover_failed', 'top_candidates': top_candidates})
                continue
            base['session_id'] = holdout; base['episode_id'] = ep.episode_id
            ov['session_id'] = holdout; ov['episode_id'] = ep.episode_id
            baseline_rows.append(base)
            overlap_rows.append(ov)
            debug_rows.append({
                'session_id': holdout,
                'episode_id': ep.episode_id,
                'selected_candidate_score': float(best_score),
                'selected_candidate_quality_vs_this_ep': float(0.65 * best_label_dbg['iou'] + 0.35 * best_label_dbg['key_frac']) if best_label_dbg['matched_episode_id'] == ep.episode_id else 0.0,
                'selected_candidate_label_debug': best_label_dbg,
                'selected_candidate': best_cand,
                'top_candidates': top_candidates,
                **rec_dbg,
            })

    report = {
        'mode': 'segment_passwordness_with_peak_keyness',
        'folds': fold_reports,
        'baseline_fixed_window': aggregate_episode_results(baseline_rows),
        'overlap_refine': aggregate_episode_results(overlap_rows),
    }
    (out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'debug_rows.json').write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
