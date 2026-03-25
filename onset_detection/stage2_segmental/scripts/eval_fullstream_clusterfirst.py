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

from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope
from onset_detection.stage2_segmental.data import build_password_episodes, estimate_sample_rate_hz
from onset_detection.stage2_segmental.length_model import compute_region_length_features, load_length_model
from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint


def _proposal_env_window(sr: float) -> int:
    return max(1, int(round(float(sr) * 0.10)))


def _proposal_energy_envelope(imu: np.ndarray, sr: float) -> np.ndarray:
    return _compute_energy_envelope(imu, _proposal_env_window(sr)).astype(np.float64)


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


def _smooth(values: np.ndarray, win_frames: int) -> np.ndarray:
    if win_frames <= 1:
        return values.astype(np.float64, copy=False)
    kernel = np.ones(int(win_frames), dtype=np.float64) / float(win_frames)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


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


def _extract_window_from_signal(
    signal: np.ndarray,
    center_frame: int,
    sample_rate_hz: float,
    target_len: int,
    pre_ms: float = 100.0,
    post_ms: float = 200.0,
):
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


def _select_exact_k_peaks(
    peaks: np.ndarray,
    scores: np.ndarray,
    sample_rate: float,
    k: int,
    gap_prior_s: float = 1.3,
):
    if len(peaks) <= k:
        return np.asarray(peaks, dtype=np.int64)
    times_s = peaks.astype(np.float64) / max(sample_rate, 1.0)
    node = np.log(np.clip(scores.astype(np.float64), 1e-8, None))
    mu = np.log(max(gap_prior_s, 1e-3))
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
        return np.sort(peaks[order].astype(np.int64))
    return peaks[idxs].astype(np.int64)


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
        out.append(
            {
                "start_frame": int(p[0]),
                "end_frame": int(p[-1]),
                "num_peaks": int(len(p)),
                "score_sum": float(np.sum(s)),
                "score_mean": float(np.mean(s)),
                "peaks": p.astype(np.int64),
                "scores": s.astype(np.float64),
            }
        )
    return out


def _classifier_score_from_frames(classifier, crop_imu: np.ndarray, frames: np.ndarray, sample_rate_hz: float, device: torch.device):
    if len(frames) == 0:
        return 0.0, ""
    windows = []
    for frame in frames.tolist():
        win = _extract_window_from_signal(crop_imu, int(frame), sample_rate_hz, classifier.target_len)
        if win is None:
            return 0.0, ""
        windows.append(win)
    xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.softmax(classifier(xb), dim=1).cpu().numpy()
    max_prob = probs.max(axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2]
    margin = np.clip(max_prob - top2, 0.0, 1.0)
    score = float(np.mean(max_prob) + 0.20 * np.mean(margin))
    pred_text = "".join(classifier.classes[int(i)] for i in probs.argmax(axis=1).tolist())
    return score, pred_text


def _predict_length(length_model, crop_imu: np.ndarray, crop_ts: np.ndarray) -> tuple[int | None, float, dict]:
    if length_model is None:
        return None, 0.0, {"used_length_model": False}
    model, labels, meta = length_model
    feat = compute_region_length_features(crop_imu, crop_ts, feature_mode=str(meta.get("feature_mode", "no_time"))).reshape(1, -1)
    pred = int(model.predict(feat)[0])
    conf = 0.0
    probs_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
        conf = float(np.max(probs))
        probs_dict = {str(int(lbl)): float(p) for lbl, p in zip(model.classes_, probs)}
    return pred, conf, {"used_length_model": True, "predicted_length": pred, "length_probs": probs_dict}


def _evaluate_fixed_from_frames(classifier, ep, local_frames):
    labels = np.asarray([classifier.class_to_idx[c] for c in ep.chars if c in classifier.class_to_idx], dtype=np.int64)
    if len(labels) != len(ep.chars):
        return None
    windows = []
    for frame in local_frames.tolist():
        win = _extract_window_from_signal(ep.imu, int(frame), ep.sample_rate_hz, classifier.target_len)
        if win is None:
            return None
        windows.append(win)
    xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=next(classifier.parameters()).device)
    with torch.no_grad():
        logits = classifier(xb).cpu().numpy()
    return {
        "reference": ep.password,
        "prediction": "".join(classifier.classes[int(i)] for i in logits.argmax(axis=1).tolist()),
        **char_topk_from_logits(logits, labels),
    }


def _evaluate_overlap_from_frames(model, ep, local_frames, device):
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


def _cluster_candidates_from_stream(
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
    energy_raw = _proposal_energy_envelope(imu, sample_rate_hz)
    candidates = []
    for smooth_s in (0.15, 0.25, 0.35, 0.5):
        smoothed = _smooth(energy_raw, max(1, int(round(sample_rate_hz * smooth_s))))
        q50, q90 = np.quantile(smoothed, [0.50, 0.90])
        prominence = max(1e-6, (q90 - q50) * 0.10)
        for dist_s in (0.35, 0.5, 0.7, 0.9, 1.1):
            peaks, props = find_peaks(
                smoothed,
                distance=max(1, int(round(sample_rate_hz * dist_s))),
                prominence=prominence,
            )
            if len(peaks) == 0:
                continue
            heights = np.asarray(props.get("peak_heights", smoothed[peaks]), dtype=np.float64)
            peak_scores = heights / max(float(np.max(heights)), 1e-8)
            for cluster in _cluster_macro_peaks(peaks, peak_scores, sample_rate_hz, gap_s=1.6):
                if not (min_keys <= cluster["num_peaks"] <= max_keys):
                    continue
                pad_frames = int(round(sample_rate_hz * 1.5))
                lo = max(0, int(cluster["start_frame"]) - pad_frames)
                hi = min(len(imu), int(cluster["end_frame"]) + pad_frames + 1)
                crop_imu = imu[lo:hi]
                crop_ts = ts[lo:hi]
                pred_len, len_conf, len_debug = _predict_length(length_model, crop_imu, crop_ts)
                if pred_len is None:
                    continue
                local_macro = (cluster["peaks"] - lo).astype(np.int64)
                chosen = _select_exact_k_peaks(local_macro, cluster["scores"], sample_rate_hz, pred_len, gap_prior_s=gap_prior_s)
                cls_score, pred_text = _classifier_score_from_frames(classifier, crop_imu, chosen, sample_rate_hz, device)
                peak_match = math.exp(-abs(cluster["num_peaks"] - pred_len) / 1.5)
                cluster_mean = float(cluster["score_mean"])
                regularity = _iki_regularity(chosen, sample_rate_hz)
                total_score = (
                    0.55 * float(cls_score)
                    + 0.30 * float(regularity)
                    + 0.20 * float(peak_match)
                    + 0.15 * float(cluster_mean)
                    + 0.10 * float(len_conf)
                )
                candidates.append(
                    {
                        "smooth_s": float(smooth_s),
                        "dist_s": float(dist_s),
                        "cluster_start_frame": int(cluster["start_frame"]),
                        "cluster_end_frame": int(cluster["end_frame"]),
                        "crop_start": int(lo),
                        "crop_end": int(hi),
                        "cluster_num_peaks": int(cluster["num_peaks"]),
                        "cluster_score_mean": float(cluster["score_mean"]),
                        "pred_len": int(pred_len),
                        "len_conf": float(len_conf),
                        "peak_match": float(peak_match),
                        "regularity": float(regularity),
                        "cls_score": float(cls_score),
                        "pred_text_preview": pred_text,
                        "total_score": float(total_score),
                        "chosen_local_frames": chosen.astype(np.int64),
                        "length_debug": len_debug,
                    }
                )
    candidates.sort(key=lambda x: x["total_score"], reverse=True)
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--overlap_checkpoint", required=True)
    ap.add_argument("--classifier_checkpoint", required=True)
    ap.add_argument("--classifier_scaler", required=True)
    ap.add_argument("--length_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--min_keys", type=int, default=6)
    ap.add_argument("--max_keys", type=int, default=12)
    ap.add_argument("--gap_prior_s", type=float, default=1.3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    overlap_model = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap_model.eval()
    overlap_model.freeze_classifier(True)
    runtime_classifier = load_external_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    runtime_classifier.eval()
    length_model = load_length_model(args.length_model)

    episodes = build_password_episodes(args.input_dir)
    by_session = {}
    for ep in episodes:
        by_session.setdefault(ep.session_id, []).append(ep)
    for sess_eps in by_session.values():
        sess_eps.sort(key=lambda x: x.episode_index)

    baseline_rows = []
    overlap_rows = []
    debug_rows = []

    for session_id, session_eps in sorted(by_session.items()):
        session_path = session_eps[0].session_path
        loader = __import__("onset_detection.stage2_episode.data.loaders", fromlist=["SessionLoader"]).SessionLoader(session_path)
        ts, imu = loader.get_imu()
        sr = estimate_sample_rate_hz(ts)
        candidates = _cluster_candidates_from_stream(
            imu=imu,
            ts=ts,
            sample_rate_hz=sr,
            classifier=runtime_classifier,
            device=device,
            length_model=length_model,
            min_keys=args.min_keys,
            max_keys=args.max_keys,
            gap_prior_s=args.gap_prior_s,
        )
        if not candidates:
            for ep in session_eps:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "error": "no_cluster_candidate"})
            continue
        best = candidates[0]
        pred_global_frames = best["chosen_local_frames"] + int(best["crop_start"])
        pred_global_ts = ts[np.clip(pred_global_frames, 0, len(ts) - 1)]

        for ep in session_eps:
            local_frames = np.searchsorted(ep.timestamps_ns, pred_global_ts, side="left")
            local_frames = np.clip(local_frames, 0, len(ep.timestamps_ns) - 1).astype(np.int64)
            k = len(ep.chars)
            if len(local_frames) != k:
                xs = np.linspace(0, len(local_frames) - 1, k)
                local_frames = np.interp(xs, np.arange(len(local_frames)), local_frames.astype(np.float64)).round().astype(np.int64)

            base = _evaluate_fixed_from_frames(runtime_classifier, ep, local_frames)
            ov, ov_debug = _evaluate_overlap_from_frames(overlap_model, ep, local_frames, device)
            if base is None or ov is None:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "error": "eval_failed"})
                continue
            base["episode_id"] = ep.episode_id
            base["session_id"] = ep.session_id
            ov["episode_id"] = ep.episode_id
            ov["session_id"] = ep.session_id
            baseline_rows.append(base)
            overlap_rows.append(ov)
            debug_rows.append(
                {
                    "episode_id": ep.episode_id,
                    "session_id": session_id,
                    "best_candidate": {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in best.items()
                        if k != "chosen_local_frames"
                    },
                    "pred_global_frames": pred_global_frames.tolist(),
                    "mapped_local_frames": local_frames.tolist(),
                    "gt_local_frames": ep.key_frames.tolist(),
                    "overlap_debug": ov_debug,
                    "top_candidates": [
                        {
                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in c.items()
                            if k != "chosen_local_frames"
                        }
                        for c in candidates[:5]
                    ],
                }
            )

    report = {
        "classifier_source": "external_classifier",
        "mode": "cluster_first_fullstream",
        "baseline_clusterfirst_fixed_window": aggregate_episode_results(baseline_rows),
        "overlap_clusterfirst_refine": aggregate_episode_results(overlap_rows),
        "delta_top1": aggregate_episode_results(overlap_rows)["char_top1"] - aggregate_episode_results(baseline_rows)["char_top1"] if baseline_rows and overlap_rows else 0.0,
        "delta_top3": aggregate_episode_results(overlap_rows)["char_top3"] - aggregate_episode_results(baseline_rows)["char_top3"] if baseline_rows and overlap_rows else 0.0,
        "delta_top5": aggregate_episode_results(overlap_rows)["char_top5"] - aggregate_episode_results(baseline_rows)["char_top5"] if baseline_rows and overlap_rows else 0.0,
        "delta_cer": aggregate_episode_results(overlap_rows)["cer"] - aggregate_episode_results(baseline_rows)["cer"] if baseline_rows and overlap_rows else 0.0,
        "min_keys": args.min_keys,
        "max_keys": args.max_keys,
        "gap_prior_s": args.gap_prior_s,
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "debug_rows.json", "w", encoding="utf-8") as f:
        json.dump(debug_rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
