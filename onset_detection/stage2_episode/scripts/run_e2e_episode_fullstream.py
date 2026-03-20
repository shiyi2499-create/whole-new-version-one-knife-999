#!/usr/bin/env python3
"""
Run stage2_episode on the full raw session stream (no GT password-block crop).

This is the honest "脱离 GT" evaluator:
- inference runs on the full session IMU
- GT is used only for offline evaluation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
for p in (PKG_ROOT,):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_e2e_episode import load_model
from utils.signal_processing import preprocess
from utils.decoder import decode_episodes
from utils.metrics import full_eval, format_report
from data.loaders import SessionLoader, discover_sessions
from configs.config import SignalConfig


def load_gt_fullstream(session_dir: str):
    loader = SessionLoader(session_dir)
    ts, imu = loader.get_imu()
    if len(ts) == 0:
        return None

    groups = loader.split_password_groups_from_enters()
    if not groups:
        return None

    gt_episodes = []
    for group in groups:
        a_s, a_e = group["start_ns"], group["end_ns"]
        gs = min(int(np.searchsorted(ts, a_s)), len(ts) - 1)
        ge = min(int(np.searchsorted(ts, a_e)), len(ts))
        ep_onsets = [
            min(int(np.searchsorted(ts, p["ts"])), len(ts) - 1)
            for p in group["keys"]
        ]
        ep_chars = [p["key"] for p in group["keys"]]
        gt_episodes.append({
            "start": gs,
            "end": ge,
            "onsets": ep_onsets,
            "chars": ep_chars,
            "num_keys": len(ep_onsets),
            "label": group.get("label", "typing_2"),
        })

    return {"imu": imu, "ts": ts, "gt_episodes": gt_episodes}


def run_one_session_fullstream(model, imu, sr, scfg, ecfg, device, typing_threshold=None):
    proc, _ = preprocess(imu, sr, scfg.use_magnitude, scfg.normalize)
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
        raw_imu=imu,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sample_rate", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--typing-threshold", type=float, default=None)
    ap.add_argument("--episode_gap_ms", type=float, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = args.sample_rate
    dev = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    model, mcfg, ecfg = load_model(args.checkpoint, dev)
    scfg = SignalConfig(sample_rate=sr)
    if args.episode_gap_ms is not None:
        ecfg.episode_gap_ms = args.episode_gap_ms

    sessions = discover_sessions(args.input_dir)
    print(f"Found {len(sessions)} sessions")

    all_det = []
    all_f1 = []
    all_rec = []
    loaded = 0

    for session_dir in sessions:
        data = load_gt_fullstream(session_dir)
        if data is None:
            continue
        loaded += 1
        imu = data["imu"]
        gt_episodes = data["gt_episodes"]

        print(f"\n--- Session {loaded}: {session_dir} ---")
        print(
            f"  Full stream: {len(imu)} samples ({len(imu)/sr:.1f}s), "
            f"GT episodes: {len(gt_episodes)}, "
            f"GT keys/ep: {[ep['num_keys'] for ep in gt_episodes]}"
        )

        dec = run_one_session_fullstream(
            model, imu, sr, scfg, ecfg, dev,
            typing_threshold=args.typing_threshold,
        )
        ev = full_eval(dec["episodes"], gt_episodes, sr)
        all_det.append(ev["episode_detection_rate"])
        all_f1.append(ev["avg_onset_f1"])
        all_rec.append(ev["avg_onset_recall"])

        print(f"  Pred episodes: {len(dec['episodes'])}, keys/ep: {[ep['num_keys'] for ep in dec['episodes']]}, total onsets: {dec['total_onsets']}")
        print(format_report(ev))

        sess_id = Path(session_dir).name
        with open(out / f"{sess_id}_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "pred_episodes": dec["episodes"],
                "gt_episodes": gt_episodes,
                "eval": ev,
                "num_pred": len(dec["episodes"]),
                "num_gt": len(gt_episodes),
                "typing_threshold": args.typing_threshold,
                "episode_gap_ms": ecfg.episode_gap_ms,
                "full_stream": True,
            }, f, ensure_ascii=False, indent=2)

    agg = {
        "num_sessions": loaded,
        "avg_episode_detection_rate": float(np.mean(all_det)) if all_det else 0.0,
        "avg_onset_f1": float(np.mean(all_f1)) if all_f1 else 0.0,
        "avg_onset_recall": float(np.mean(all_rec)) if all_rec else 0.0,
        "typing_threshold": args.typing_threshold,
        "episode_gap_ms": ecfg.episode_gap_ms,
        "full_stream": True,
    }
    with open(out / "aggregate_results.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    print("\nDone!")
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
