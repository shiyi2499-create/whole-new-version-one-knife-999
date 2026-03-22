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

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.password_segment_detector import load_segment_detector, run_binary_inference, extract_coarse_regions
from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope
from onset_detection.stage2_segmental.data import build_password_episodes, extract_fixed_window, estimate_sample_rate_hz
from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.length_model import compute_region_length_features, load_length_model


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


def select_primary_regions(regions, max_regions: int = 1):
    if not regions:
        return []

    def score(r):
        return 0.45 * float(r.mean_prob) + 0.55 * float(r.max_prob)

    ranked = sorted(regions, key=score, reverse=True)
    return ranked[:max_regions]


def coarse_region_support_score(region) -> float:
    return float(0.45 * float(region.mean_prob) + 0.55 * float(region.max_prob))


def _extract_window_from_signal(signal: np.ndarray, center_frame: int, sample_rate_hz: float, target_len: int, pre_ms: float = 100.0, post_ms: float = 200.0):
    pre_frames = int(round(pre_ms / 1000.0 * sample_rate_hz))
    post_frames = int(round(post_ms / 1000.0 * sample_rate_hz))
    lo = max(0, int(center_frame) - pre_frames)
    hi = min(len(signal), int(center_frame) + post_frames)
    if hi - lo < 3:
        return None
    out = resample(signal[lo:hi], target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)


def _select_exact_k_peaks(peaks: np.ndarray, scores: np.ndarray, sample_rate: float, duration_s: float, k: int, fixed_gap_prior_s: float | None = None, return_score: bool = False):
    if len(peaks) == 0:
        out = np.asarray([], dtype=np.int64)
        return (out, float("-inf")) if return_score else out
    if len(peaks) <= k:
        out = peaks.astype(np.int64)
        return (out, float(np.mean(scores)) if len(scores) else float("-inf")) if return_score else out

    times_s = peaks.astype(np.float64) / max(sample_rate, 1)
    node = np.log(np.clip(scores.astype(np.float64), 1e-8, None))
    target_gap = max((fixed_gap_prior_s if fixed_gap_prior_s is not None else duration_s / max(k - 1, 1)), 1e-3)
    mu = np.log(target_gap)
    sigma = 0.40

    dp = np.full((k, len(peaks)), -1e18, dtype=np.float64)
    prev = np.full((k, len(peaks)), -1, dtype=np.int32)
    dp[0, :] = node
    for m in range(1, k):
        for j in range(m, len(peaks)):
            best = -1e18
            best_i = -1
            for i in range(m - 1, j):
                dt_s = max(times_s[j] - times_s[i], 1e-6)
                z = (np.log(dt_s) - mu) / sigma
                trans = -0.5 * z * z
                cur = dp[m - 1, i] + node[j] + trans
                if cur > best:
                    best = cur
                    best_i = i
            dp[m, j] = best
            prev[m, j] = best_i

    end = int(np.argmax(dp[k - 1]))
    idxs = [end]
    cur = end
    for m in range(k - 1, 0, -1):
        cur = int(prev[m, cur])
        if cur < 0:
            break
        idxs.append(cur)
    idxs = np.array(list(reversed(idxs)), dtype=np.int64)
    if len(idxs) != k:
        order = np.argsort(-scores)[:k]
        out = np.sort(peaks[order].astype(np.int64))
        return (out, float(np.mean(scores[order]))) if return_score else out
    out = peaks[idxs].astype(np.int64)
    return (out, float(dp[k - 1, end])) if return_score else out


def _search_best_k_sequence(
    peaks: np.ndarray,
    scores: np.ndarray,
    sample_rate: float,
    min_keys: int,
    max_keys: int,
    gap_prior_s: float,
    count_prior_center: float | None = 8.0,
    count_prior_weight: float = 0.08,
):
    best_seq = None
    best_score = -1e18
    upper = min(max_keys, len(peaks))
    lower = min(min_keys, upper)
    for k in range(lower, upper + 1):
        seq, raw_score = _select_exact_k_peaks(peaks, scores, sample_rate, duration_s=0.0, k=k, fixed_gap_prior_s=gap_prior_s, return_score=True)
        count_pen = 0.0 if count_prior_center is None else -float(count_prior_weight) * abs(k - float(count_prior_center))
        total = float(raw_score + count_pen)
        if total > best_score:
            best_score = total
            best_seq = seq
    if best_seq is None:
        return np.asarray([], dtype=np.int64), float('-inf'), lower
    return np.asarray(best_seq, dtype=np.int64), float(best_score), int(len(best_seq))


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


def _build_length_subregion_from_energy(
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

    peaks, props = find_peaks(region, distance=distance, prominence=prominence, height=height)
    if len(peaks) < 6:
        peaks, props = find_peaks(region, distance=max(4, distance // 2), prominence=max(1e-6, prominence * 0.5))
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


def propose_energy_classifier_anchors(
    raw_imu: np.ndarray,
    sample_rate: float,
    classifier,
    device: torch.device,
    expected_keys: int = 8,
    min_keys: int = 6,
    max_keys: int = 12,
    gap_prior_s: float = 1.5,
    count_prior_center: float | None = 8.0,
    count_prior_weight: float = 0.08,
):
    energy = _compute_energy_envelope(raw_imu, int(round(sample_rate)))
    if len(energy) < 3:
        return np.asarray([], dtype=np.int64), {"reason": "short_crop"}

    region = energy.astype(np.float64)
    q50 = float(np.quantile(region, 0.50))
    q90 = float(np.quantile(region, 0.90))
    q98 = float(np.quantile(region, 0.98))
    prominence = max(1e-6, (q90 - q50) * 0.15)
    height = q50 + (q98 - q50) * 0.15
    distance = max(6, int(round(sample_rate * 0.04)))

    target_k = expected_keys if expected_keys > 0 else max_keys
    peaks, props = find_peaks(region, distance=distance, prominence=prominence, height=height)
    if len(peaks) < target_k:
        peaks, props = find_peaks(region, distance=max(4, distance // 2), prominence=max(1e-6, prominence * 0.5))
    if len(peaks) < target_k:
        peaks, props = find_peaks(region, distance=max(3, distance // 3))
    if len(peaks) == 0:
        return np.asarray([], dtype=np.int64), {"reason": "no_peaks"}

    energy_heights = np.asarray(props.get("peak_heights", region[peaks]), dtype=np.float64)
    e_norm = energy_heights / max(float(np.max(energy_heights)), 1e-8)

    windows = []
    valid_mask = []
    for p in peaks:
        win = _extract_window_from_signal(raw_imu, int(p), sample_rate, classifier.target_len)
        if win is None:
            valid_mask.append(False)
            windows.append(np.zeros((classifier.target_len, raw_imu.shape[1]), dtype=np.float32))
        else:
            valid_mask.append(True)
            windows.append(win)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = classifier(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    max_prob = probs.max(axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2]
    margin = np.clip(max_prob - top2, 0.0, 1.0)
    cls_score = 0.7 * max_prob + 0.3 * margin
    cls_score[~valid_mask] = 0.0

    scores = 0.35 * e_norm + 0.65 * cls_score
    duration_s = len(region) / max(sample_rate, 1.0)
    if expected_keys > 0:
        used_expected = int(expected_keys)
        chosen = _select_exact_k_peaks(peaks, scores, sample_rate, duration_s, used_expected, fixed_gap_prior_s=gap_prior_s)
        search_score = None
    else:
        chosen, search_score, used_expected = _search_best_k_sequence(
            peaks,
            scores,
            sample_rate,
            min_keys=min_keys,
            max_keys=min(max_keys, len(peaks)),
            gap_prior_s=gap_prior_s,
            count_prior_center=count_prior_center,
            count_prior_weight=count_prior_weight,
        )
    return np.sort(chosen.astype(np.int64)), {
        "num_raw_peaks": int(len(peaks)),
        "num_chosen": int(len(chosen)),
        "used_expected_keys": int(used_expected),
        "search_score": None if search_score is None else float(search_score),
        "q50": q50,
        "q90": q90,
        "q98": q98,
        "distance": distance,
        "prominence": prominence,
        "height": height,
        "chosen_scores": [float(x) for x in scores[np.isin(peaks, chosen)]],
    }




def infer_expected_keys_from_region(
    crop_imu: np.ndarray,
    crop_ts: np.ndarray,
    length_model,
    sample_rate: float,
) -> tuple[int | None, dict, tuple[int, int] | None]:
    if length_model is None:
        return None, {"used_length_model": False}, None
    model, labels, meta = length_model
    subregion, sub_debug = _build_length_subregion_from_energy(crop_imu, sample_rate)
    if subregion is not None:
        lo, hi = subregion
        feat_imu = crop_imu[lo:hi]
        feat_ts = crop_ts[lo:hi]
    else:
        feat_imu = crop_imu
        feat_ts = crop_ts
    feature_mode = str(meta.get("feature_mode", "legacy_time"))
    feat = compute_region_length_features(feat_imu, feat_ts, feature_mode=feature_mode).reshape(1, -1)
    pred = int(model.predict(feat)[0])
    debug = {
        "used_length_model": True,
        "predicted_length": pred,
        "candidate_labels": [int(x) for x in labels],
        "feature_num_frames": int(len(feat_imu)),
        "feature_duration_s": float((feat_ts[-1] - feat_ts[0]) / 1e9) if len(feat_ts) > 1 else 0.0,
        "feature_mode": feature_mode,
        "subregion_debug": sub_debug,
    }
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(feat)[0]
        debug['length_probs'] = {str(int(lbl)): float(p) for lbl, p in zip(model.classes_, proba)}
    return pred, debug, subregion

def evaluate_fixed_from_frames(classifier, ep, local_frames):
    labels = np.asarray([classifier.class_to_idx[c] for c in ep.chars if c in classifier.class_to_idx], dtype=np.int64)
    if len(labels) != len(ep.chars):
        return None
    windows = []
    for frame in local_frames.tolist():
        win = extract_fixed_window(ep, int(frame), target_len=classifier.target_len)
        if win is None:
            return None
        windows.append(win)
    if len(windows) != len(labels):
        return None
    with torch.no_grad():
        xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=next(classifier.parameters()).device)
        logits = classifier(xb).cpu().numpy()
    return {
        "reference": ep.password,
        "prediction": "".join(classifier.classes[int(i)] for i in logits.argmax(axis=1).tolist()),
        **char_topk_from_logits(logits, labels),
    }


def score_candidate_region_from_frames(classifier, raw_imu, local_frames, sample_rate_hz: float, device: torch.device):
    if len(local_frames) == 0:
        return None
    windows = []
    for frame in np.asarray(local_frames, dtype=np.int64).tolist():
        win = _extract_window_from_signal(raw_imu, int(frame), sample_rate_hz, classifier.target_len)
        if win is None:
            return None
        windows.append(win)
    xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = classifier(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    max_prob = probs.max(axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2]
    margin = np.clip(max_prob - top2, 0.0, 1.0)
    pred_text = "".join(classifier.classes[int(i)] for i in probs.argmax(axis=1).tolist())
    score = float(np.mean(max_prob) + 0.35 * np.mean(margin))
    return {
        "score": score,
        "mean_max_prob": float(np.mean(max_prob)),
        "mean_margin": float(np.mean(margin)),
        "pred_text": pred_text,
        "num_windows": int(len(windows)),
    }


def evaluate_overlap_from_frames(model, ep, local_frames, device):
    labels = np.asarray([model.classifier.class_to_idx[c] for c in ep.chars if c in model.classifier.class_to_idx], dtype=np.int64)
    if len(labels) != len(ep.chars):
        return None, None
    with torch.no_grad():
        imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
        key_frames = torch.tensor(local_frames, dtype=torch.long, device=device)
        out = model.forward_episode(imu, key_frames, ep.sample_rate_hz)
        logits = out["logits"].detach().cpu().numpy()
    return {
        "reference": ep.password,
        "prediction": "".join(model.classifier.classes[int(i)] for i in logits.argmax(axis=1).tolist()),
        **char_topk_from_logits(logits, labels),
    }, {
        "offsets": out["offsets"].detach().cpu().tolist(),
        "width_scales": out["width_scales"].detach().cpu().tolist(),
        "starts": out["starts"].detach().cpu().tolist(),
        "ends": out["ends"].detach().cpu().tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--segment_checkpoint", required=True)
    ap.add_argument("--segment_scaler", required=True)
    ap.add_argument("--overlap_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--segment_threshold", type=float, default=0.3)
    ap.add_argument("--sample_rate", type=float, default=0.0, help="Override sample rate; <=0 means infer from timestamps")
    ap.add_argument("--expected_duration_s", type=float, default=15.0)
    ap.add_argument("--expected_keys", type=int, default=8, help="Expected key count; <=0 means infer softly from anchor scores")
    ap.add_argument("--min_keys", type=int, default=6)
    ap.add_argument("--max_keys", type=int, default=12)
    ap.add_argument("--gap_prior_s", type=float, default=1.3)
    ap.add_argument("--classifier_checkpoint", default="")
    ap.add_argument("--classifier_scaler", default="")
    ap.add_argument("--length_model", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    seg_model, seg_means, seg_stds, seg_meta = load_segment_detector(args.segment_checkpoint, args.segment_scaler, device)
    overlap_model = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap_model.eval()
    overlap_model.freeze_classifier(True)
    runtime_classifier = overlap_model.classifier
    classifier_source = "overlap_checkpoint"
    length_model = load_length_model(args.length_model) if args.length_model else None
    if args.classifier_checkpoint and args.classifier_scaler:
        runtime_classifier = load_external_inception(args.classifier_checkpoint, args.classifier_scaler, device)
        runtime_classifier.eval()
        classifier_source = "external_classifier"

    episodes = build_password_episodes(args.input_dir)
    by_session = {}
    for ep in episodes:
        by_session.setdefault(ep.session_id, []).append(ep)
    for sess_eps in by_session.values():
        sess_eps.sort(key=lambda x: x.episode_index)

    baseline_rows = []
    overlap_rows = []
    debug_rows = []

    for session_path in sorted({ep.session_path for ep in episodes}):
        session_id = Path(session_path).name
        full_loader = __import__("onset_detection.stage2_episode.data.loaders", fromlist=["SessionLoader"]).SessionLoader(session_path)
        ts, imu = full_loader.get_imu()
        sensor = np.column_stack([ts, imu])
        probs, times = run_binary_inference(
            seg_model, sensor, seg_means, seg_stds,
            seg_meta["window_ms"], seg_meta["stride_ms"], seg_meta["target_rate_hz"], device,
        )
        coarse_all = extract_coarse_regions(probs, times, threshold=args.segment_threshold, merge_gap_s=1.5, min_duration_s=2.0, margin_s=1.0)
        coarse = select_primary_regions(coarse_all, max_regions=5)
        session_eps = by_session[session_id]
        if not coarse:
            for ep in session_eps:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "error": "no_coarse_region"})
            continue

        candidate_debug = []
        best_candidate = None
        best_score = float("-inf")
        for ridx, region in enumerate(coarse):
            region_mask = (ts >= region.start_s * 1e9) & (ts <= region.end_s * 1e9)
            region_idx = np.where(region_mask)[0]
            if len(region_idx) < 10:
                candidate_debug.append({"candidate_rank": ridx, "error": "empty_coarse_region"})
                continue
            crop_start = int(region_idx[0])
            crop_imu = imu[region_idx]
            crop_ts = ts[region_idx]
            effective_sr = float(args.sample_rate) if args.sample_rate and args.sample_rate > 0 else estimate_sample_rate_hz(crop_ts)
            inferred_keys, length_debug, length_subregion = infer_expected_keys_from_region(crop_imu, crop_ts, length_model, effective_sr)
            expected_keys = int(inferred_keys) if inferred_keys is not None else int(args.expected_keys)
            work_crop_start = crop_start
            work_imu = crop_imu
            work_ts = crop_ts
            if length_subregion is not None:
                sub_lo, sub_hi = length_subregion
                if sub_hi - sub_lo >= 10:
                    work_crop_start = crop_start + int(sub_lo)
                    work_imu = crop_imu[sub_lo:sub_hi]
                    work_ts = crop_ts[sub_lo:sub_hi]
            local_anchor_frames, anchor_debug = propose_energy_classifier_anchors(
                work_imu, effective_sr, runtime_classifier, device, expected_keys, args.min_keys, args.max_keys, args.gap_prior_s
            )
            anchor_debug['expected_keys_used'] = int(expected_keys)
            anchor_debug['length_debug'] = length_debug
            candidate_score = score_candidate_region_from_frames(
                runtime_classifier, work_imu, local_anchor_frames, effective_sr, device
            )
            entry = {
                "candidate_rank": ridx,
                "coarse_region": {"start_s": region.start_s, "end_s": region.end_s, "duration_s": region.duration_s},
                "effective_sample_rate_hz": effective_sr,
                "anchor_debug": anchor_debug,
                "candidate_score": candidate_score,
            }
            candidate_debug.append(entry)
            if candidate_score is None:
                continue
            score = float(candidate_score["score"])
            region_support = coarse_region_support_score(region)
            total_score = score + 0.8 * region_support
            entry["region_support_score"] = float(region_support)
            entry["total_score"] = float(total_score)
            if total_score > best_score:
                best_score = total_score
                best_candidate = {
                    "region": region,
                    "crop_start": crop_start,
                    "crop_imu": crop_imu,
                    "crop_ts": crop_ts,
                    "effective_sr": effective_sr,
                    "local_anchor_frames": np.asarray(local_anchor_frames, dtype=np.int64),
                    "anchor_debug": anchor_debug,
                    "candidate_score": candidate_score,
                    "region_support_score": float(region_support),
                    "total_score": float(total_score),
                }

        if best_candidate is None:
            for ep in session_eps:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "error": "no_candidate_region_scored", "candidate_debug": candidate_debug})
            continue

        region = best_candidate["region"]
        crop_start = best_candidate["crop_start"]
        crop_imu = best_candidate["crop_imu"]
        crop_ts = best_candidate["crop_ts"]
        effective_sr = best_candidate["effective_sr"]
        local_anchor_frames = best_candidate["local_anchor_frames"]
        anchor_debug = best_candidate["anchor_debug"]
        pred_global_frames = local_anchor_frames + crop_start if len(local_anchor_frames) else np.asarray([], dtype=np.int64)
        pred_global_ts = ts[np.clip(pred_global_frames, 0, len(ts) - 1)] if len(pred_global_frames) else np.asarray([], dtype=np.int64)

        for ep in session_eps:
            if len(pred_global_ts) == 0:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "coarse_region": {"start_s": region.start_s, "end_s": region.end_s}, "anchor_debug": anchor_debug, "candidate_debug": candidate_debug, "error": "no_anchors"})
                continue
            local_frames = np.searchsorted(ep.timestamps_ns, pred_global_ts, side="left")
            local_frames = np.clip(local_frames, 0, len(ep.timestamps_ns) - 1).astype(np.int64)
            k = len(ep.chars)
            if len(local_frames) != k:
                xs = np.linspace(0, len(local_frames) - 1, k)
                local_frames = np.interp(xs, np.arange(len(local_frames)), local_frames.astype(np.float64)).round().astype(np.int64)

            base = evaluate_fixed_from_frames(runtime_classifier, ep, local_frames)
            ov, ov_debug = evaluate_overlap_from_frames(overlap_model, ep, local_frames, device)
            if base is None or ov is None:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "error": "eval_failed"})
                continue
            base["episode_id"] = ep.episode_id
            base["session_id"] = ep.session_id
            ov["episode_id"] = ep.episode_id
            ov["session_id"] = ep.session_id
            baseline_rows.append(base)
            overlap_rows.append(ov)
            debug_rows.append({
                "episode_id": ep.episode_id,
                "session_id": session_id,
                "coarse_region": {"start_s": region.start_s, "end_s": region.end_s, "duration_s": region.duration_s},
                "effective_sample_rate_hz": effective_sr,
                "anchor_debug": anchor_debug,
                "candidate_debug": candidate_debug,
                "chosen_candidate_score": best_candidate["candidate_score"],
                "chosen_region_support_score": best_candidate["region_support_score"],
                "chosen_total_score": best_candidate["total_score"],
                "pred_global_frames": pred_global_frames.tolist(),
                "mapped_local_frames": local_frames.tolist(),
                "gt_local_frames": ep.key_frames.tolist(),
                "overlap_debug": ov_debug,
            })

    report = {
        "classifier_source": classifier_source,
        "baseline_fullstream_coarse_energy_cls_fixed_window": aggregate_episode_results(baseline_rows),
        "overlap_fullstream_coarse_energy_cls_refine": aggregate_episode_results(overlap_rows),
        "delta_top1": aggregate_episode_results(overlap_rows)["char_top1"] - aggregate_episode_results(baseline_rows)["char_top1"] if baseline_rows and overlap_rows else 0.0,
        "delta_top3": aggregate_episode_results(overlap_rows)["char_top3"] - aggregate_episode_results(baseline_rows)["char_top3"] if baseline_rows and overlap_rows else 0.0,
        "delta_top5": aggregate_episode_results(overlap_rows)["char_top5"] - aggregate_episode_results(baseline_rows)["char_top5"] if baseline_rows and overlap_rows else 0.0,
        "delta_cer": aggregate_episode_results(overlap_rows)["cer"] - aggregate_episode_results(baseline_rows)["cer"] if baseline_rows and overlap_rows else 0.0,
        "segment_threshold": args.segment_threshold,
        "expected_keys": args.expected_keys,
        "min_keys": args.min_keys,
        "max_keys": args.max_keys,
        "gap_prior_s": args.gap_prior_s,
    }

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "baseline_rows.json", "w", encoding="utf-8") as f:
        json.dump(baseline_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "overlap_rows.json", "w", encoding="utf-8") as f:
        json.dump(overlap_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "debug_rows.json", "w", encoding="utf-8") as f:
        json.dump(debug_rows, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
