#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
for p in (PROJECT_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_segmental.data import (
    PasswordEpisode,
    build_password_episodes,
    estimate_sample_rate_hz,
)
from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from onset_detection.stage2_segmental.scripts.train_gt_overlap import evaluate_fixed_window, evaluate_overlap
from onset_detection.stage2_episode.data.loaders import SessionLoader


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


def median_iki_frames(episodes: List[PasswordEpisode]) -> float:
    vals = []
    for ep in episodes:
        k = ep.key_frames.astype(np.int64)
        if len(k) > 1:
            vals.extend(np.diff(k).tolist())
    if not vals:
        return 250.0
    return float(np.median(vals))


def _recursive_split(onsets: np.ndarray, max_keys: int, gap_thresh: int) -> List[np.ndarray]:
    out: List[np.ndarray] = []

    def rec(xs: np.ndarray):
        if len(xs) <= max_keys:
            out.append(xs)
            return
        gaps = np.diff(xs)
        if len(gaps) == 0 or int(gaps.max()) < gap_thresh:
            out.append(xs)
            return
        idx = int(np.argmax(gaps))
        rec(xs[: idx + 1])
        rec(xs[idx + 1 :])

    rec(np.asarray(onsets, dtype=np.int64))
    return out


def _select_monotonic_subset(cands: np.ndarray, k: int, left: float, right: float) -> np.ndarray:
    cands = np.asarray(sorted(set(int(x) for x in cands)), dtype=np.int64)
    if len(cands) == 0:
        return np.round(np.linspace(left, right, k)).astype(np.int64)
    if len(cands) <= k:
        if len(cands) == 1:
            return np.full(k, int(cands[0]), dtype=np.int64)
        xs = np.linspace(0, len(cands) - 1, k)
        return np.interp(xs, np.arange(len(cands)), cands.astype(np.float64)).round().astype(np.int64)

    targets = np.linspace(left, right, k).astype(np.float64)
    n = len(cands)
    dp = np.full((k, n), np.inf, dtype=np.float64)
    prev = np.full((k, n), -1, dtype=np.int32)
    dp[0, :] = (cands.astype(np.float64) - targets[0]) ** 2
    for i in range(1, k):
        for j in range(i, n):
            costs = dp[i - 1, i - 1:j] + (cands[j].astype(np.float64) - targets[i]) ** 2
            best_rel = int(np.argmin(costs))
            dp[i, j] = costs[best_rel]
            prev[i, j] = i - 1 + best_rel
    end = int(np.argmin(dp[k - 1, k - 1:]) + (k - 1))
    out = [cands[end]]
    cur = end
    for i in range(k - 1, 0, -1):
        cur = int(prev[i, cur])
        out.append(cands[cur])
    out.reverse()
    return np.asarray(out, dtype=np.int64)


def _cluster_predictions(pred_episodes: List[dict], median_iki: float, min_keys_keep: int, max_keys: int) -> List[np.ndarray]:
    split_gap = int(round(max(380.0, median_iki * 1.7)))
    clusters: List[np.ndarray] = []
    for ep in pred_episodes:
        onsets = np.asarray(ep.get("onsets", []), dtype=np.int64)
        if len(onsets) == 0:
            continue
        for chunk in _recursive_split(onsets, max_keys=max_keys, gap_thresh=split_gap):
            if len(chunk) >= min_keys_keep:
                clusters.append(chunk)
    clusters.sort(key=lambda x: int(x[0]))
    return clusters


def _estimate_k(chunk: np.ndarray, median_iki: float, min_k: int, max_k: int) -> int:
    if len(chunk) <= min_k:
        return len(chunk)
    duration = max(int(chunk[-1]) - int(chunk[0]), 1)
    est = int(round(duration / max(median_iki, 1.0))) + 1
    est = max(min_k, min(max_k, est))
    est = min(est, len(chunk))
    return est


def _build_pseudo_episode(session_path: str, session_id: str, pseudo_index: int, cluster: np.ndarray, pad_ms: float) -> Tuple[PasswordEpisode, Tuple[int, int]]:
    loader = SessionLoader(session_path)
    ts_all, imu_all = loader.get_imu()
    sr = estimate_sample_rate_hz(ts_all)
    pad_frames = int(round(pad_ms / 1000.0 * sr))
    lo = max(0, int(cluster[0]) - pad_frames)
    hi = min(len(imu_all) - 1, int(cluster[-1]) + pad_frames)
    region_imu = imu_all[lo : hi + 1].astype(np.float32)
    region_ts = ts_all[lo : hi + 1].astype(np.int64)
    local_frames = (cluster - lo).astype(np.int64)
    ep = PasswordEpisode(
        session_path=session_path,
        session_id=session_id,
        episode_index=pseudo_index,
        episode_id=f"{session_id}::pred{pseudo_index:02d}",
        prompt="",
        password="",
        imu=region_imu,
        timestamps_ns=region_ts,
        key_frames=local_frames,
        key_timestamps_ns=np.zeros(len(local_frames), dtype=np.int64),
        chars=["?"] * len(local_frames),
        sample_rate_hz=sr,
    )
    return ep, (int(cluster[0]), int(cluster[-1]))


def _iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0:
        return 0.0
    return inter / union


def _match_to_gt(pred_ranges: List[Tuple[int, int]], gt_eps: List[PasswordEpisode]) -> List[int]:
    gt_ranges = []
    for ep in gt_eps:
        ep_start_ns = int(ep.timestamps_ns[0])
        global_frames = []
        for ts in ep.key_timestamps_ns.tolist():
            gf = int(np.searchsorted(ep.timestamps_ns, ts, side="left"))
            global_frames.append(gf)
        # Convert local episode frames to session-global-like range using timestamps.
        # The key timestamps are absolute, so we use them directly on the local episode clock.
        # For matching we only need the absolute temporal span, not exact frame units.
        gt_ranges.append((int(ep.key_timestamps_ns[0]), int(ep.key_timestamps_ns[-1])))
    matches = [-1] * len(pred_ranges)
    used = set()
    for i, pr in enumerate(pred_ranges):
        best_j = -1
        best_iou = 0.0
        for j, gr in enumerate(gt_ranges):
            if j in used:
                continue
            val = _iou(pr, gr)
            if val > best_iou:
                best_iou = val
                best_j = j
        if best_j >= 0 and best_iou > 0.05:
            matches[i] = best_j
            used.add(best_j)
    return matches


def evaluate_session(model, gt_eps: List[PasswordEpisode], result_json: dict, pad_ms: float, median_iki: float, min_keys_keep: int, max_keys: int, device: torch.device):
    clusters = _cluster_predictions(result_json.get("pred_episodes", []), median_iki, min_keys_keep, max_keys)
    pseudo_eps = []
    pred_ranges = []
    for i, cluster in enumerate(clusters):
        k = _estimate_k(cluster, median_iki, min_keys_keep, max_keys)
        chosen = _select_monotonic_subset(cluster, k, float(cluster[0]), float(cluster[-1]))
        ep, gr = _build_pseudo_episode(gt_eps[0].session_path, gt_eps[0].session_id, i, chosen, pad_ms)
        pseudo_eps.append(ep)
        # Convert predicted global frame span to absolute ns span using the session timestamps.
        loader = SessionLoader(gt_eps[0].session_path)
        ts_all, _ = loader.get_imu()
        pred_ranges.append((int(ts_all[int(chosen[0])]), int(ts_all[int(chosen[-1])])))

    matches = _match_to_gt(pred_ranges, gt_eps)

    rows_base = []
    rows_ov = []
    debug = []
    for pred_ep, match_idx in zip(pseudo_eps, matches):
        if match_idx < 0:
            continue
        gt_ep = gt_eps[match_idx]
        labels = np.asarray(
            [model.classifier.class_to_idx[c] for c in gt_ep.chars if c in model.classifier.class_to_idx],
            dtype=np.int64,
        )
        if len(labels) != len(gt_ep.chars):
            continue
        if len(pred_ep.key_frames) != len(labels):
            # full-auto count estimate still missed the length
            continue

        base_metrics, base_rows = evaluate_fixed_window(model.classifier, [pred_ep])
        ov_metrics, ov_rows, ov_debug = evaluate_overlap(model, [pred_ep], device)

        base_logits = None
        ov_logits = None
        # Re-run once to get logits for char_topk against GT labels.
        windows = []
        for frame in pred_ep.key_frames.tolist():
            win = __import__("onset_detection.stage2_segmental.data", fromlist=["extract_fixed_window"]).extract_fixed_window(
                pred_ep, int(frame), target_len=model.classifier.target_len
            )
            if win is None:
                windows = []
                break
            windows.append(win)
        if len(windows) != len(labels):
            continue
        with torch.no_grad():
            xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
            base_logits = model.classifier(xb).cpu().numpy()
            imu = torch.tensor(pred_ep.imu, dtype=torch.float32, device=device)
            key_frames = torch.tensor(pred_ep.key_frames, dtype=torch.long, device=device)
            ov_out = model.forward_episode(imu, key_frames, pred_ep.sample_rate_hz)
            ov_logits = ov_out["logits"].detach().cpu().numpy()

        rows_base.append({
            "episode_id": gt_ep.episode_id,
            "session_id": gt_ep.session_id,
            "reference": gt_ep.password,
            "prediction": "".join(model.classifier.classes[int(i)] for i in base_logits.argmax(axis=1).tolist()),
            **char_topk_from_logits(base_logits, labels),
        })
        rows_ov.append({
            "episode_id": gt_ep.episode_id,
            "session_id": gt_ep.session_id,
            "reference": gt_ep.password,
            "prediction": "".join(model.classifier.classes[int(i)] for i in ov_logits.argmax(axis=1).tolist()),
            **char_topk_from_logits(ov_logits, labels),
        })
        debug.append({
            "pred_episode_id": pred_ep.episode_id,
            "matched_gt_episode_id": gt_ep.episode_id,
            "pred_key_frames": pred_ep.key_frames.tolist(),
            "gt_key_frames": gt_ep.key_frames.tolist(),
            "pred_len": len(pred_ep.key_frames),
            "gt_len": len(gt_ep.key_frames),
            "pred_range": pred_ranges[pseudo_eps.index(pred_ep)],
            "gt_range": (int(gt_ep.key_timestamps_ns[0]), int(gt_ep.key_timestamps_ns[-1])),
        })
    return rows_base, rows_ov, debug, len(clusters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--overlap_checkpoint", required=True)
    ap.add_argument("--episode_results_dir", required=True)
    ap.add_argument("--holdout_sessions", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pad_ms", type=float, default=250.0)
    ap.add_argument("--min_keys_keep", type=int, default=4)
    ap.add_argument("--max_keys", type=int, default=12)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    holdouts = [s.strip() for s in args.holdout_sessions.split(",") if s.strip()]

    episodes = build_password_episodes(args.input_dir)
    by_sess = {}
    for ep in episodes:
        by_sess.setdefault(ep.session_id, []).append(ep)
    for sess_eps in by_sess.values():
        sess_eps.sort(key=lambda x: x.episode_index)

    train_eps = [ep for ep in episodes if ep.session_id not in set(holdouts)]
    med_iki = median_iki_frames(train_eps)

    model = load_overlap_checkpoint(args.overlap_checkpoint, device)
    model.eval()
    model.freeze_classifier(True)

    base_rows_all = []
    ov_rows_all = []
    debug_all = []
    clusters_per_session = {}
    for session_id in holdouts:
        if session_id not in by_sess:
            continue
        result_path = Path(args.episode_results_dir) / f"{session_id}_results.json"
        if not result_path.exists():
            continue
        result_json = json.load(open(result_path))
        rows_b, rows_o, debug_rows, n_clusters = evaluate_session(
            model,
            by_sess[session_id],
            result_json,
            pad_ms=args.pad_ms,
            median_iki=med_iki,
            min_keys_keep=args.min_keys_keep,
            max_keys=args.max_keys,
            device=device,
        )
        base_rows_all.extend(rows_b)
        ov_rows_all.extend(rows_o)
        debug_all.extend(debug_rows)
        clusters_per_session[session_id] = n_clusters

    report = {
        "median_iki_frames_from_train": med_iki,
        "clusters_per_session": clusters_per_session,
        "baseline_fullautoish": aggregate_episode_results(base_rows_all),
        "overlap_fullautoish": aggregate_episode_results(ov_rows_all),
        "delta_top1": aggregate_episode_results(ov_rows_all)["char_top1"] - aggregate_episode_results(base_rows_all)["char_top1"],
        "delta_top3": aggregate_episode_results(ov_rows_all)["char_top3"] - aggregate_episode_results(base_rows_all)["char_top3"],
        "delta_top5": aggregate_episode_results(ov_rows_all)["char_top5"] - aggregate_episode_results(base_rows_all)["char_top5"],
        "delta_cer": aggregate_episode_results(ov_rows_all)["cer"] - aggregate_episode_results(base_rows_all)["cer"],
        "holdout_sessions": holdouts,
    }
    with open(out_dir / "fullautoish_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "fullautoish_baseline_rows.json", "w", encoding="utf-8") as f:
        json.dump(base_rows_all, f, ensure_ascii=False, indent=2)
    with open(out_dir / "fullautoish_overlap_rows.json", "w", encoding="utf-8") as f:
        json.dump(ov_rows_all, f, ensure_ascii=False, indent=2)
    with open(out_dir / "fullautoish_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_all, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
