#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
for p in (PROJECT_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_segmental.data import (
    build_password_episodes,
    extract_fixed_window,
)
from onset_detection.stage2_segmental.metrics import (
    aggregate_episode_results,
    char_topk_from_logits,
)
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


def _episode_candidates_from_json(
    result_json: dict,
    gt_index: int,
    pad_frames: int,
    include_gt_boundaries: bool = True,
) -> np.ndarray:
    gt_ep = result_json["gt_episodes"][gt_index]
    lo = int(gt_ep["start"]) - pad_frames
    hi = int(gt_ep["end"]) + pad_frames
    cands: List[int] = []
    for pred_ep in result_json.get("pred_episodes", []):
        for onset in pred_ep.get("onsets", []):
            oi = int(onset)
            if lo <= oi <= hi:
                cands.append(oi)
    if include_gt_boundaries:
        cands.extend([int(gt_ep["start"]), int(gt_ep["end"])])
    return np.asarray(sorted(set(cands)), dtype=np.int64)


def _map_region_to_local(region_frames: np.ndarray, gt_region_onsets: List[int], local_gt_frames: np.ndarray) -> np.ndarray:
    if len(region_frames) == 0:
        return np.asarray([], dtype=np.int64)
    # Same raw session samples, different clipped subarrays: offset mapping is enough.
    offset = int(local_gt_frames[0]) - int(gt_region_onsets[0])
    return (region_frames.astype(np.int64) + offset).astype(np.int64)


def evaluate_with_candidate_anchors(
    overlap_model,
    episodes,
    result_dir: str,
    device: torch.device,
    pad_ms: float,
):
    per_session = {}
    for ep in episodes:
        per_session.setdefault(ep.session_id, []).append(ep)
    for sess_eps in per_session.values():
        sess_eps.sort(key=lambda x: x.episode_index)

    baseline_results = []
    overlap_results = []
    debug_rows = []

    for session_id, sess_eps in sorted(per_session.items()):
        p = Path(result_dir) / f"{session_id}_results.json"
        if not p.exists():
            continue
        result_json = json.load(open(p))
        if len(result_json.get("gt_episodes", [])) < len(sess_eps):
            continue

        for ep in sess_eps:
            gt_region = result_json["gt_episodes"][ep.episode_index]
            k = len(ep.chars)
            pad_frames = int(round(pad_ms / 1000.0 * ep.sample_rate_hz))
            region_cands = _episode_candidates_from_json(result_json, ep.episode_index, pad_frames)
            if len(region_cands) == 0:
                region_cands = np.asarray([gt_region["start"], gt_region["end"]], dtype=np.int64)
            chosen_region = _select_monotonic_subset(
                region_cands,
                k,
                float(gt_region["start"]),
                float(gt_region["end"]),
            )
            chosen_local = _map_region_to_local(chosen_region, gt_region["onsets"], ep.key_frames)
            chosen_local = np.clip(chosen_local, 0, len(ep.imu) - 1).astype(np.int64)

            labels = np.asarray(
                [overlap_model.classifier.class_to_idx[c] for c in ep.chars if c in overlap_model.classifier.class_to_idx],
                dtype=np.int64,
            )
            if len(labels) != k:
                continue

            windows = []
            for frame in chosen_local.tolist():
                win = extract_fixed_window(ep, int(frame), target_len=overlap_model.classifier.target_len)
                if win is None:
                    windows = []
                    break
                windows.append(win)
            if len(windows) != k:
                continue

            with torch.no_grad():
                xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
                base_logits = overlap_model.classifier(xb).cpu().numpy()
                imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
                key_frames = torch.tensor(chosen_local, dtype=torch.long, device=device)
                ov_out = overlap_model.forward_episode(imu, key_frames, ep.sample_rate_hz)
                ov_logits = ov_out["logits"].detach().cpu().numpy()

            baseline_results.append({
                "episode_id": ep.episode_id,
                "session_id": ep.session_id,
                "reference": ep.password,
                "prediction": "".join(overlap_model.classifier.classes[int(i)] for i in base_logits.argmax(axis=1).tolist()),
                **char_topk_from_logits(base_logits, labels),
            })
            overlap_results.append({
                "episode_id": ep.episode_id,
                "session_id": ep.session_id,
                "reference": ep.password,
                "prediction": "".join(overlap_model.classifier.classes[int(i)] for i in ov_logits.argmax(axis=1).tolist()),
                **char_topk_from_logits(ov_logits, labels),
            })
            debug_rows.append({
                "episode_id": ep.episode_id,
                "session_id": ep.session_id,
                "reference": ep.password,
                "num_raw_candidates": int(len(region_cands)),
                "region_candidates": region_cands.tolist(),
                "chosen_region": chosen_region.tolist(),
                "chosen_local": chosen_local.tolist(),
                "gt_local": ep.key_frames.tolist(),
                "overlap_offsets": ov_out["offsets"].detach().cpu().tolist(),
                "overlap_width_scales": ov_out["width_scales"].detach().cpu().tolist(),
            })

    return (
        aggregate_episode_results(baseline_results),
        baseline_results,
        aggregate_episode_results(overlap_results),
        overlap_results,
        debug_rows,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--overlap_checkpoint", required=True)
    ap.add_argument("--episode_results_dir", required=True)
    ap.add_argument("--holdout_sessions", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--pad_ms", type=float, default=250.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    holdouts = [s.strip() for s in args.holdout_sessions.split(",") if s.strip()]
    episodes = build_password_episodes(args.input_dir)
    val_eps = [ep for ep in episodes if ep.session_id in set(holdouts)]
    if not val_eps:
        raise RuntimeError("No held-out episodes found for the requested sessions.")

    model = load_overlap_checkpoint(args.overlap_checkpoint, device)
    model.eval()
    model.freeze_classifier(True)

    base_metrics, base_rows, ov_metrics, ov_rows, debug_rows = evaluate_with_candidate_anchors(
        model,
        val_eps,
        args.episode_results_dir,
        device,
        pad_ms=args.pad_ms,
    )

    report = {
        "baseline_candidate_fixed_window": base_metrics,
        "overlap_candidate_refine": ov_metrics,
        "delta_top1": ov_metrics["char_top1"] - base_metrics["char_top1"],
        "delta_top3": ov_metrics["char_top3"] - base_metrics["char_top3"],
        "delta_top5": ov_metrics["char_top5"] - base_metrics["char_top5"],
        "delta_cer": ov_metrics["cer"] - base_metrics["cer"],
        "holdout_sessions": holdouts,
        "candidate_source": args.episode_results_dir,
        "pad_ms": args.pad_ms,
    }

    with open(out_dir / "candidate_bridge_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "candidate_bridge_baseline_rows.json", "w", encoding="utf-8") as f:
        json.dump(base_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "candidate_bridge_overlap_rows.json", "w", encoding="utf-8") as f:
        json.dump(ov_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "candidate_bridge_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_rows, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
