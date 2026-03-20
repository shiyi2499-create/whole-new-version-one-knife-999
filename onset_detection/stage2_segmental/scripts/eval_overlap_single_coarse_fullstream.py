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
from onset_detection.stage2_episode.scripts.run_e2e_episode import load_model as load_episode_model
from onset_detection.stage2_episode.utils.signal_processing import preprocess
from onset_detection.stage2_episode.utils.decoder import decode_episodes
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


def choose_best_pred_episode(pred_episodes, expected_keys: int = 8):
    if not pred_episodes:
        return None

    def score(ep):
        nk = int(ep.get("num_keys", len(ep.get("onsets", []))))
        dur = max(1, int(ep.get("end", 0)) - int(ep.get("start", 0)))
        key_score = math.exp(-abs(nk - expected_keys) / max(expected_keys, 1))
        dur_score = 1.0 / (1.0 + max(0.0, abs(dur - 14 * 100) / (14 * 100)))
        return 0.8 * key_score + 0.2 * dur_score

    return max(pred_episodes, key=score)


def run_stage2_on_crop(model, imu_crop, sr, typing_threshold, ecfg, scfg, device):
    proc, _ = preprocess(imu_crop, sr, scfg.use_magnitude, scfg.normalize)
    x = torch.from_numpy(proc.T).float().unsqueeze(0).to(device)
    with torch.no_grad():
        typing_logits, onset_logits = model(x)
        typing_probs = torch.softmax(typing_logits, dim=1)[0, 1].cpu().numpy()
        if typing_threshold is None:
            preds = typing_logits.argmax(dim=1)[0].cpu().numpy()
        else:
            preds = (typing_probs >= float(typing_threshold)).astype(np.int64)
        onset_probs = None
        if onset_logits is not None:
            onset_probs = torch.sigmoid(onset_logits)[0, 0].cpu().numpy()
    dec = decode_episodes(
        preds,
        raw_imu=imu_crop,
        typing_probs=typing_probs,
        onset_probs=onset_probs,
        sample_rate=sr,
        median_kernel=ecfg.median_kernel,
        min_typing_run_ms=ecfg.min_typing_run_ms,
        episode_gap_ms=ecfg.episode_gap_ms,
        min_onset_gap_ms=ecfg.min_onset_gap_ms,
        min_episode_keys=ecfg.min_episode_keys,
        min_episode_duration_ms=ecfg.min_episode_duration_ms,
    )
    return dec


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
    ap.add_argument("--episode_checkpoint", required=True)
    ap.add_argument("--overlap_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--segment_threshold", type=float, default=0.3)
    ap.add_argument("--typing_threshold", type=float, default=0.945)
    ap.add_argument("--sample_rate", type=int, default=100)
    ap.add_argument("--expected_duration_s", type=float, default=15.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    seg_model, seg_means, seg_stds, seg_meta = load_segment_detector(args.segment_checkpoint, args.segment_scaler, device)
    ep_model, _mcfg, ecfg = load_episode_model(args.episode_checkpoint, device)
    scfg = __import__("onset_detection.stage2_episode.configs.config", fromlist=["SignalConfig"]).SignalConfig(sample_rate=args.sample_rate)
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
        probs, times = run_binary_inference(
            seg_model, np.column_stack([ts, imu]), seg_means, seg_stds,
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
        dec = run_stage2_on_crop(ep_model, crop_imu, args.sample_rate, args.typing_threshold, ecfg, scfg, device)
        best_pred = choose_best_pred_episode(dec.get("episodes", []), expected_keys=8)
        if best_pred is None:
            for ep in session_eps:
                debug_rows.append({
                    "episode_id": ep.episode_id,
                    "session_id": session_id,
                    "coarse_region": {"start_s": region.start_s, "end_s": region.end_s},
                    "error": "no_stage2_episode",
                })
            continue

        pred_global_frames = (np.asarray(best_pred["onsets"], dtype=np.int64) + crop_start).astype(np.int64)
        pred_global_ts = ts[np.clip(pred_global_frames, 0, len(ts) - 1)]

        for ep in session_eps:
            local_frames = np.searchsorted(ep.timestamps_ns, pred_global_ts, side="left")
            local_frames = np.clip(local_frames, 0, len(ep.timestamps_ns) - 1).astype(np.int64)
            k = len(ep.chars)
            if len(local_frames) == 0:
                debug_rows.append({"episode_id": ep.episode_id, "session_id": session_id, "error": "no_local_frames"})
                continue
            if len(local_frames) != k:
                # Monotonic interpolation to expected count.
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
                "stage2_pred_num_keys": int(best_pred.get("num_keys", len(best_pred.get("onsets", [])))),
                "pred_global_frames": pred_global_frames.tolist(),
                "mapped_local_frames": local_frames.tolist(),
                "gt_local_frames": ep.key_frames.tolist(),
                "overlap_debug": ov_debug,
            })

    report = {
        "baseline_fullstream_coarse_fixed_window": aggregate_episode_results(baseline_rows),
        "overlap_fullstream_coarse_refine": aggregate_episode_results(overlap_rows),
        "delta_top1": aggregate_episode_results(overlap_rows)["char_top1"] - aggregate_episode_results(baseline_rows)["char_top1"] if baseline_rows and overlap_rows else 0.0,
        "delta_top3": aggregate_episode_results(overlap_rows)["char_top3"] - aggregate_episode_results(baseline_rows)["char_top3"] if baseline_rows and overlap_rows else 0.0,
        "delta_top5": aggregate_episode_results(overlap_rows)["char_top5"] - aggregate_episode_results(baseline_rows)["char_top5"] if baseline_rows and overlap_rows else 0.0,
        "delta_cer": aggregate_episode_results(overlap_rows)["cer"] - aggregate_episode_results(baseline_rows)["cer"] if baseline_rows and overlap_rows else 0.0,
        "segment_threshold": args.segment_threshold,
        "typing_threshold": args.typing_threshold,
        "expected_duration_s": args.expected_duration_s,
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
