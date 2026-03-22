#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.signal import find_peaks, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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
from onset_detection.stage2_segmental.length_model import compute_region_length_features, load_length_model
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits
from onset_detection.stage2_segmental.scripts.eval_fullstream_clusterfirst import (
    _classifier_score_from_frames,
    _cluster_macro_peaks,
    _evaluate_fixed_from_frames,
    _evaluate_overlap_from_frames,
    _extract_window_from_signal,
    _predict_length,
    _select_exact_k_peaks,
    _smooth,
)


def resolve_device(name: str) -> torch.device:
    req = name.lower()
    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"
    return torch.device(req)


def _normalized_energy_shape(crop_imu: np.ndarray, sample_rate_hz: float, out_len: int = 64) -> np.ndarray:
    energy = _compute_energy_envelope(crop_imu, int(round(sample_rate_hz))).astype(np.float64)
    smoothed = _smooth(energy, max(1, int(round(sample_rate_hz * 0.15))))
    if len(smoothed) < 4:
        smoothed = np.pad(smoothed, (0, max(0, 4 - len(smoothed))), mode="edge")
    shape = resample(smoothed, out_len)
    if np.iscomplexobj(shape):
        shape = np.real(shape)
    shape = np.asarray(shape, dtype=np.float64)
    shape -= shape.min()
    mx = float(np.max(shape))
    if mx > 1e-8:
        shape /= mx
    return shape.astype(np.float32)


def _coarse_activity_scores(crop_imu: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    energy = _compute_energy_envelope(crop_imu, int(round(sample_rate_hz))).astype(np.float64)
    if len(energy) < 4:
        return np.zeros_like(energy, dtype=np.float64)
    smooth_short = _smooth(energy, max(1, int(round(sample_rate_hz * 0.08))))
    smooth_long = _smooth(energy, max(1, int(round(sample_rate_hz * 0.35))))
    diff = np.maximum(smooth_short - smooth_long, 0.0)
    q = np.quantile(diff, 0.95) if len(diff) else 0.0
    if q > 1e-8:
        diff = diff / q
    return np.clip(diff, 0.0, 3.0)


def _cluster_iou_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 1e-8 else 0.0


def _contains_key_fraction(start_ns: int, end_ns: int, key_ts_ns: np.ndarray) -> float:
    if len(key_ts_ns) == 0:
        return 0.0
    inside = np.sum((key_ts_ns >= start_ns) & (key_ts_ns <= end_ns))
    return float(inside) / float(len(key_ts_ns))


def _candidate_feature_vector(candidate: dict, crop_imu: np.ndarray, crop_ts: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    shape = _normalized_energy_shape(crop_imu, sample_rate_hz, out_len=64)
    len_debug = candidate.get("length_debug", {})
    probs = len_debug.get("length_probs", {}) if isinstance(len_debug, dict) else {}
    prob8 = float(probs.get("8", 0.0))
    prob9 = float(probs.get("9", 0.0))
    prob10 = float(probs.get("10", 0.0))
    base = np.array([
        float(candidate.get("cluster_num_peaks", 0)),
        float(candidate.get("cluster_score_mean", 0.0)),
        float(candidate.get("cluster_score_sum", 0.0)),
        float(candidate.get("pred_len", 0)),
        float(candidate.get("len_conf", 0.0)),
        float(candidate.get("peak_match", 0.0)),
        float(candidate.get("cls_score", 0.0)),
        prob8,
        prob9,
        prob10,
    ], dtype=np.float32)
    return np.concatenate([base, shape], axis=0)


def _propose_candidates_fullstream(
    imu: np.ndarray,
    ts: np.ndarray,
    sample_rate_hz: float,
    classifier,
    device: torch.device,
    length_model,
    min_keys: int,
    max_keys: int,
    gap_prior_s: float,
):
    energy_raw = _compute_energy_envelope(imu, int(round(sample_rate_hz))).astype(np.float64)
    activity = _coarse_activity_scores(imu, sample_rate_hz)
    candidates = []
    seen = set()
    for smooth_s in (0.10, 0.15, 0.25, 0.35):
        smoothed = _smooth(energy_raw, max(1, int(round(sample_rate_hz * smooth_s))))
        q50, q90, q98 = np.quantile(smoothed, [0.50, 0.90, 0.98])
        prominence = max(1e-6, (q90 - q50) * 0.08)
        height = q50 + (q98 - q50) * 0.04
        for dist_s in (0.30, 0.40, 0.50, 0.70, 0.90):
            peaks, props = find_peaks(
                smoothed,
                distance=max(1, int(round(sample_rate_hz * dist_s))),
                prominence=prominence,
                height=height,
            )
            if len(peaks) == 0:
                continue
            heights = np.asarray(props.get("peak_heights", smoothed[peaks]), dtype=np.float64)
            peak_scores = heights / max(float(np.max(heights)), 1e-8)
            for gap_s in (1.35, 1.6, 1.9, 2.2):
                for cluster in _cluster_macro_peaks(peaks, peak_scores, sample_rate_hz, gap_s=gap_s):
                    key = (int(cluster["start_frame"]), int(cluster["end_frame"]), int(cluster["num_peaks"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    if not (3 <= cluster["num_peaks"] <= max(20, max_keys + 4)):
                        continue
                    pad_frames = int(round(sample_rate_hz * 1.25))
                    lo = max(0, int(cluster["start_frame"]) - pad_frames)
                    hi = min(len(imu), int(cluster["end_frame"]) + pad_frames + 1)
                    crop_imu = imu[lo:hi]
                    crop_ts = ts[lo:hi]
                    pred_len, len_conf, len_debug = _predict_length(length_model, crop_imu, crop_ts)
                    if pred_len is None:
                        continue
                    if not (min_keys <= pred_len <= max_keys):
                        continue
                    local_macro = (cluster["peaks"] - lo).astype(np.int64)
                    chosen = _select_exact_k_peaks(local_macro, cluster["scores"], sample_rate_hz, pred_len, gap_prior_s=gap_prior_s)
                    cls_score, pred_text = _classifier_score_from_frames(classifier, crop_imu, chosen, sample_rate_hz, device)
                    peak_match = math.exp(-abs(cluster["num_peaks"] - pred_len) / 1.5)
                    crop_activity = activity[lo:hi]
                    middle = crop_activity[max(0, len(crop_activity)//4): min(len(crop_activity), 3*len(crop_activity)//4)]
                    side = np.concatenate([crop_activity[: max(1, len(crop_activity)//8)], crop_activity[-max(1, len(crop_activity)//8):]])
                    burst_ratio = float((middle.mean() + 1e-6) / (side.mean() + 1e-6)) if len(middle) and len(side) else 1.0
                    candidates.append(
                        {
                            "smooth_s": float(smooth_s),
                            "dist_s": float(dist_s),
                            "gap_s": float(gap_s),
                            "cluster_start_frame": int(cluster["start_frame"]),
                            "cluster_end_frame": int(cluster["end_frame"]),
                            "crop_start": int(lo),
                            "crop_end": int(hi),
                            "cluster_num_peaks": int(cluster["num_peaks"]),
                            "cluster_score_sum": float(cluster["score_sum"]),
                            "cluster_score_mean": float(cluster["score_mean"]),
                            "pred_len": int(pred_len),
                            "len_conf": float(len_conf),
                            "peak_match": float(peak_match),
                            "cls_score": float(cls_score),
                            "burst_ratio": float(burst_ratio),
                            "pred_text_preview": pred_text,
                            "chosen_local_frames": chosen.astype(np.int64),
                            "length_debug": len_debug,
                        }
                    )
    return candidates


def _label_candidate(candidate: dict, ep, session_ts: np.ndarray) -> tuple[int, dict]:
    start_idx = int(candidate["crop_start"])
    end_idx = max(start_idx + 1, int(candidate["crop_end"]) - 1)
    cand_start_ns = int(session_ts[min(start_idx, len(session_ts)-1)])
    cand_end_ns = int(session_ts[min(end_idx, len(session_ts)-1)])
    gt_start_ns = int(ep.key_timestamps_ns[0])
    gt_end_ns = int(ep.key_timestamps_ns[-1])
    iou = _cluster_iou_seconds(cand_start_ns * 1e-9, cand_end_ns * 1e-9, gt_start_ns * 1e-9, gt_end_ns * 1e-9)
    key_frac = _contains_key_fraction(cand_start_ns, cand_end_ns, ep.key_timestamps_ns)
    is_pos = int((iou >= 0.30 and key_frac >= 0.75) or (iou >= 0.45) or (key_frac >= 0.875))
    return is_pos, {"iou": float(iou), "key_frac": float(key_frac)}


def _build_ranker_dataset(episodes, classifier, device, length_model, min_keys, max_keys, gap_prior_s):
    X=[]; y=[]; meta=[]
    for ep in episodes:
        loader = SessionLoader(ep.session_path)
        ts, imu = loader.get_imu()
        sr = estimate_sample_rate_hz(ts)
        cands = _propose_candidates_fullstream(imu, ts, sr, classifier, device, length_model, min_keys, max_keys, gap_prior_s)
        for cand in cands:
            crop_imu = imu[int(cand['crop_start']):int(cand['crop_end'])]
            crop_ts = ts[int(cand['crop_start']):int(cand['crop_end'])]
            feat = _candidate_feature_vector(cand, crop_imu, crop_ts, sr)
            label, dbg = _label_candidate(cand, ep, ts)
            X.append(feat)
            y.append(label)
            meta.append({
                'session_id': ep.session_id,
                'episode_id': ep.episode_id,
                'label': int(label),
                **dbg,
                'pred_len': int(cand['pred_len']),
                'len_conf': float(cand['len_conf']),
                'cluster_num_peaks': int(cand['cluster_num_peaks']),
                'cls_score': float(cand['cls_score']),
                'burst_ratio': float(cand['burst_ratio']),
                'crop_start': int(cand['crop_start']),
                'crop_end': int(cand['crop_end']),
            })
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), meta


def _score_candidate_with_ranker(model, candidate: dict, crop_imu: np.ndarray, crop_ts: np.ndarray, sample_rate_hz: float) -> float:
    feat = _candidate_feature_vector(candidate, crop_imu, crop_ts, sample_rate_hz).reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        return float(model.predict_proba(feat)[0, 1])
    return float(model.predict(feat)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dirs', nargs='+', required=True)
    ap.add_argument('--classifier_checkpoint', required=True)
    ap.add_argument('--classifier_scaler', required=True)
    ap.add_argument('--overlap_checkpoint', required=True)
    ap.add_argument('--length_model', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--min_keys', type=int, default=6)
    ap.add_argument('--max_keys', type=int, default=12)
    ap.add_argument('--gap_prior_s', type=float, default=1.3)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    episodes = []
    for d in args.input_dirs:
        episodes.extend(build_password_episodes(d))
    episodes.sort(key=lambda x: (x.session_id, x.episode_id))
    session_ids = sorted({ep.session_id for ep in episodes})

    classifier = load_external_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    classifier.eval()
    overlap = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap.eval()
    overlap.freeze_classifier(True)
    length_model = load_length_model(args.length_model)

    X, y, meta = _build_ranker_dataset(episodes, classifier, device, length_model, args.min_keys, args.max_keys, args.gap_prior_s)
    if len(X) == 0:
        raise RuntimeError('No cluster candidates found for ranker dataset')

    with open(out_dir / 'ranker_dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'num_candidates': int(len(X)),
            'num_positive': int(np.sum(y==1)),
            'num_negative': int(np.sum(y==0)),
            'sessions': session_ids,
        }, f, ensure_ascii=False, indent=2)

    fold_reports=[]
    baseline_rows=[]
    overlap_rows=[]
    debug_rows=[]
    pred_records=[]

    for holdout in session_ids:
        train_mask = np.array([m['session_id'] != holdout for m in meta], dtype=bool)
        val_mask = ~train_mask
        if len(np.unique(y[train_mask])) < 2:
            continue
        ranker = RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=42,
        )
        ranker.fit(X[train_mask], y[train_mask])
        val_probs = ranker.predict_proba(X[val_mask])[:,1]
        val_pred = (val_probs >= 0.5).astype(np.int64)
        fold_reports.append({
            'holdout_session': holdout,
            'candidate_accuracy': float(accuracy_score(y[val_mask], val_pred)),
            'candidate_auc': float(roc_auc_score(y[val_mask], val_probs)) if len(np.unique(y[val_mask])) > 1 else None,
            'num_val_candidates': int(np.sum(val_mask)),
        })

        holdout_eps = [ep for ep in episodes if ep.session_id == holdout]
        for ep in holdout_eps:
            loader = SessionLoader(ep.session_path)
            ts, imu = loader.get_imu()
            sr = estimate_sample_rate_hz(ts)
            cands = _propose_candidates_fullstream(imu, ts, sr, classifier, device, length_model, args.min_keys, args.max_keys, args.gap_prior_s)
            if not cands:
                debug_rows.append({'session_id': holdout, 'episode_id': ep.episode_id, 'error': 'no_cluster_candidate'})
                continue
            scored=[]
            for cand in cands:
                crop_imu = imu[int(cand['crop_start']):int(cand['crop_end'])]
                crop_ts = ts[int(cand['crop_start']):int(cand['crop_end'])]
                rank_score = _score_candidate_with_ranker(ranker, cand, crop_imu, crop_ts, sr)
                scored.append((rank_score, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best = scored[0]
            pred_global_frames = best['chosen_local_frames'] + int(best['crop_start'])
            pred_global_ts = ts[np.clip(pred_global_frames, 0, len(ts)-1)]
            local_frames = np.searchsorted(ep.timestamps_ns, pred_global_ts, side='left')
            local_frames = np.clip(local_frames, 0, len(ep.timestamps_ns)-1).astype(np.int64)
            k = len(ep.chars)
            if len(local_frames) != k:
                xs = np.linspace(0, len(local_frames)-1, k)
                local_frames = np.interp(xs, np.arange(len(local_frames)), local_frames.astype(np.float64)).round().astype(np.int64)
            base = _evaluate_fixed_from_frames(classifier, ep, local_frames)
            ov, ov_debug = _evaluate_overlap_from_frames(overlap, ep, local_frames, device)
            if base is None or ov is None:
                debug_rows.append({'session_id': holdout, 'episode_id': ep.episode_id, 'error': 'eval_failed'})
                continue
            base['episode_id']=ep.episode_id; base['session_id']=holdout
            ov['episode_id']=ep.episode_id; ov['session_id']=holdout
            baseline_rows.append(base); overlap_rows.append(ov)
            label_dbg = _label_candidate(best, ep, ts)[1]
            debug_rows.append({
                'session_id': holdout,
                'episode_id': ep.episode_id,
                'rank_score': float(best_score),
                'best_candidate': {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in best.items() if k!='chosen_local_frames'},
                'candidate_label_debug': label_dbg,
                'mapped_local_frames': local_frames.tolist(),
                'gt_local_frames': ep.key_frames.tolist(),
                'overlap_debug': ov_debug,
                'top_candidates': [
                    {
                        'rank_score': float(s),
                        **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in cand.items() if k!='chosen_local_frames'}
                    } for s,cand in scored[:5]
                ],
            })
            pred_records.append({'session_id': holdout, 'episode_id': ep.episode_id, 'reference': ep.password, 'pred_baseline': base['prediction'], 'pred_overlap': ov['prediction'], 'rank_score': float(best_score)})

    report = {
        'mode': 'cluster_ranker_fullstream',
        'candidate_cv': fold_reports,
        'baseline_cluster_ranker_fixed_window': aggregate_episode_results(baseline_rows),
        'overlap_cluster_ranker_refine': aggregate_episode_results(overlap_rows),
    }
    with open(out_dir / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'debug_rows.json', 'w', encoding='utf-8') as f:
        json.dump(debug_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'predictions.json', 'w', encoding='utf-8') as f:
        json.dump(pred_records, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
