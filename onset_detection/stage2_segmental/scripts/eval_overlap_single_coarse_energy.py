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
from scipy.signal import find_peaks

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.password_segment_detector import (
    load_segment_detector,
    run_binary_inference,
    extract_coarse_regions,
)
from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope
from onset_detection.stage2_segmental.data import build_password_episodes, extract_fixed_window
from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint


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


def select_duration_aware_region(regions, expected_s: float, max_regions: int = 1):
    if not regions:
        return []

    def score(r):
        quality = 0.45 * float(r.mean_prob) + 0.55 * float(r.max_prob)
        dur_pen = math.exp(-abs(float(r.duration_s) - expected_s) / max(expected_s, 1e-6))
        long_pen = 1.0 if float(r.duration_s) <= expected_s * 1.5 else math.exp(-(float(r.duration_s) - expected_s * 1.5) / 8.0)
        return quality * dur_pen * long_pen

    ranked = sorted(regions, key=score, reverse=True)
    return ranked[:max_regions]


def _select_exact_k_peaks(peaks: np.ndarray, scores: np.ndarray, sample_rate: int, duration_s: float, k: int) -> np.ndarray:
    if len(peaks) == 0:
        return np.asarray([], dtype=np.int64)
    if len(peaks) <= k:
        return peaks.astype(np.int64)

    times_s = peaks.astype(np.float64) / max(sample_rate, 1)
    node = np.log(np.clip(scores.astype(np.float64), 1e-8, None))
    target_gap = max(duration_s / max(k - 1, 1), 1e-3)
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
        return np.sort(peaks[order].astype(np.int64))
    return peaks[idxs].astype(np.int64)


def propose_energy_anchors(raw_imu: np.ndarray, sample_rate: int, expected_keys: int = 8) -> tuple[np.ndarray, dict]:
    energy = _compute_energy_envelope(raw_imu, sample_rate)
    if len(energy) < 3:
        return np.asarray([], dtype=np.int64), {"reason": "short_crop"}

    region = energy.astype(np.float64)
    q50 = float(np.quantile(region, 0.50))
    q90 = float(np.quantile(region, 0.90))
    q98 = float(np.quantile(region, 0.98))
    prominence = max(1e-6, (q90 - q50) * 0.15)
    height = q50 + (q98 - q50) * 0.15
    distance = max(6, int(round(sample_rate * 0.04)))

    peaks, props = find_peaks(region, distance=distance, prominence=prominence, height=height)
    if len(peaks) < expected_keys:
        peaks, props = find_peaks(region, distance=max(4, distance // 2), prominence=max(1e-6, prominence * 0.5))
    if len(peaks) < expected_keys:
        peaks, props = find_peaks(region, distance=max(3, distance // 3))
    if len(peaks) == 0:
        return np.asarray([], dtype=np.int64), {
            "reason": "no_peaks",
            "q50": q50,
            "q90": q90,
            "q98": q98,
        }

    heights = props.get("peak_heights", region[peaks])
    scores = np.asarray(heights, dtype=np.float64)
    duration_s = len(region) / max(sample_rate, 1)
    chosen = _select_exact_k_peaks(peaks, scores, sample_rate, duration_s, expected_keys)
    return np.sort(chosen.astype(np.int64)), {
        "num_raw_peaks": int(len(peaks)),
        "num_chosen": int(len(chosen)),
        "q50": q50,
        "q90": q90,
        "q98": q98,
        "distance": distance,
        "prominence": prominence,
        "height": height,
    }


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
    ap.add_argument("--sample_rate", type=int, default=200)
    ap.add_argument("--expected_duration_s", type=float, default=15.0)
    ap.add_argument("--expected_keys", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    seg_model, seg_means, seg_stds, seg_meta = load_segment_detector(args.segment_checkpoint, args.segment_scaler, device)
    overlap_model = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap_model.eval()
    overlap_model.freeze_classifier(True)

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
        coarse_all = extract_coarse_regions(
            probs, times,
            threshold=args.segment_threshold,
            merge_gap_s=1.5,
            min_duration_s=2.0,
            margin_s=1.0,
        )
        coarse = select_duration_aware_region(coarse_all, expected_s=args.expected_duration_s, max_regions=1)
        session_eps = by_session[session_id]
        if not coarse:
            for ep in session_eps:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "error": "no_coarse_region"})
            continue

        region = coarse[0]
        region_mask = (ts >= region.start_s * 1e9) & (ts <= region.end_s * 1e9)
        region_idx = np.where(region_mask)[0]
        if len(region_idx) < 10:
            for ep in session_eps:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "error": "empty_coarse_region"})
            continue
        crop_start = int(region_idx[0])
        crop_imu = imu[region_idx]
        local_anchor_frames, anchor_debug = propose_energy_anchors(crop_imu, args.sample_rate, args.expected_keys)
        pred_global_frames = local_anchor_frames + crop_start if len(local_anchor_frames) else np.asarray([], dtype=np.int64)
        pred_global_ts = ts[np.clip(pred_global_frames, 0, len(ts) - 1)] if len(pred_global_frames) else np.asarray([], dtype=np.int64)

        for ep in session_eps:
            if len(pred_global_ts) == 0:
                debug_rows.append({
                    "episode_id": ep.episode_id,
                    "session_id": session_id,
                    "coarse_region": {"start_s": region.start_s, "end_s": region.end_s},
                    "anchor_debug": anchor_debug,
                    "error": "no_anchors",
                })
                continue
            local_frames = np.searchsorted(ep.timestamps_ns, pred_global_ts, side="left")
            local_frames = np.clip(local_frames, 0, len(ep.timestamps_ns) - 1).astype(np.int64)
            k = len(ep.chars)
            if len(local_frames) != k:
                xs = np.linspace(0, len(local_frames) - 1, k)
                local_frames = np.interp(xs, np.arange(len(local_frames)), local_frames.astype(np.float64)).round().astype(np.int64)

            base = evaluate_fixed_from_frames(overlap_model.classifier, ep, local_frames)
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
                "anchor_debug": anchor_debug,
                "pred_global_frames": pred_global_frames.tolist(),
                "mapped_local_frames": local_frames.tolist(),
                "gt_local_frames": ep.key_frames.tolist(),
                "overlap_debug": ov_debug,
            })

    report = {
        "baseline_fullstream_coarse_energy_fixed_window": aggregate_episode_results(baseline_rows),
        "overlap_fullstream_coarse_energy_refine": aggregate_episode_results(overlap_rows),
        "delta_top1": aggregate_episode_results(overlap_rows)["char_top1"] - aggregate_episode_results(baseline_rows)["char_top1"] if baseline_rows and overlap_rows else 0.0,
        "delta_top3": aggregate_episode_results(overlap_rows)["char_top3"] - aggregate_episode_results(baseline_rows)["char_top3"] if baseline_rows and overlap_rows else 0.0,
        "delta_top5": aggregate_episode_results(overlap_rows)["char_top5"] - aggregate_episode_results(baseline_rows)["char_top5"] if baseline_rows and overlap_rows else 0.0,
        "delta_cer": aggregate_episode_results(overlap_rows)["cer"] - aggregate_episode_results(baseline_rows)["cer"] if baseline_rows and overlap_rows else 0.0,
        "segment_threshold": args.segment_threshold,
        "expected_duration_s": args.expected_duration_s,
        "expected_keys": args.expected_keys,
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
