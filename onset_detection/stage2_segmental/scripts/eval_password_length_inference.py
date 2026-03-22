#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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

from onset_detection.stage2_episode.data.loaders import SessionLoader
from onset_detection.stage2_segmental.data import estimate_sample_rate_hz
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.scripts.eval_overlap_single_coarse_energy_cls import (
    _extract_window_from_signal,
    propose_energy_classifier_anchors,
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


def discover_sessions(password_dirs: list[str]) -> list[str]:
    sessions = []
    for rd in password_dirs:
        if not os.path.isdir(rd):
            continue
        for f in sorted(os.listdir(rd)):
            if f.startswith("."):
                continue
            if "_free_type_" in f and f.endswith("_sensor.csv"):
                prefix = os.path.join(rd, f.replace("_sensor.csv", ""))
                if os.path.exists(prefix + "_events.csv") and os.path.exists(prefix + "_attempts.csv"):
                    sessions.append(prefix)
    return sorted(sessions)


def parse_attempts(session_prefix: str):
    with open(session_prefix + "_attempts.csv", newline="") as f:
        return list(csv.DictReader(f))


def parse_sequences(session_prefix: str):
    attempts = parse_attempts(session_prefix)
    sequences = []
    cur_events = []
    with open(session_prefix + "_events.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["event_type"] != "press":
                continue
            key = row["key"].lower()
            ts = int(row["timestamp_ns"])
            if key in {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
                       "left", "right", "up", "down", "delete"}:
                continue
            if key in {"enter", "return"}:
                sequences.append({"events": cur_events.copy(), "submit_ns": ts})
                cur_events = []
                continue
            cur_events.append({"key": key, "timestamp_ns": ts})
    out = []
    for idx, seq in enumerate(sequences):
        att = attempts[idx] if idx < len(attempts) else {}
        match = (att.get("match") or "").upper()
        if match and match != "YES":
            continue
        typed = (att.get("typed_text") or "").strip().lower()
        if not typed or not seq["events"]:
            continue
        out.append({
            "attempt_idx": idx,
            "reference": typed,
            "true_len": len(typed),
            "events": seq["events"],
            "submit_ns": seq["submit_ns"],
        })
    return out


def decode_chars(raw_imu: np.ndarray, sample_rate: float, frames: np.ndarray, classifier, device: torch.device) -> str:
    windows = []
    for p in frames.tolist():
        win = _extract_window_from_signal(raw_imu, int(p), sample_rate, classifier.target_len)
        if win is None:
            return ""
        windows.append(win)
    if not windows:
        return ""
    with torch.no_grad():
        xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
        logits = classifier(xb)
        idx = logits.argmax(dim=1).cpu().tolist()
    return "".join(classifier.classes[int(i)] for i in idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--password-dir", action="append", required=True)
    ap.add_argument("--classifier-checkpoint", required=True)
    ap.add_argument("--classifier-scaler", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--min-keys", type=int, default=8)
    ap.add_argument("--max-keys", type=int, default=9)
    ap.add_argument("--gap-prior-s", type=float, default=1.3)
    ap.add_argument("--count-prior-center", type=float, default=-1.0, help="Set <0 to disable hard bias toward a specific length")
    ap.add_argument("--count-prior-weight", type=float, default=0.08)
    ap.add_argument("--pre-margin-ms", type=float, default=120.0)
    ap.add_argument("--post-margin-ms", type=float, default=220.0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    classifier = load_external_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    classifier.eval()

    count_prior_center = None if args.count_prior_center < 0 else float(args.count_prior_center)

    rows = []
    for session_prefix in discover_sessions(args.password_dir):
        loader = SessionLoader(session_prefix)
        ts_ns, imu = loader.get_imu()
        seqs = parse_sequences(session_prefix)
        sr = estimate_sample_rate_hz(ts_ns)
        pre_ns = int(round(args.pre_margin_ms * 1e6))
        post_ns = int(round(args.post_margin_ms * 1e6))
        for seq in seqs:
            first_ns = int(seq["events"][0]["timestamp_ns"]) - pre_ns
            last_ns = int(seq["submit_ns"]) + post_ns
            mask = (ts_ns >= first_ns) & (ts_ns <= last_ns)
            idx = np.where(mask)[0]
            if len(idx) < 10:
                continue
            crop_imu = imu[idx]
            crop_ts = ts_ns[idx]
            crop_sr = estimate_sample_rate_hz(crop_ts) or sr
            chosen, debug = propose_energy_classifier_anchors(
                crop_imu, crop_sr, classifier, device,
                expected_keys=0,
                min_keys=args.min_keys,
                max_keys=args.max_keys,
                gap_prior_s=args.gap_prior_s,
                count_prior_center=count_prior_center,
                count_prior_weight=args.count_prior_weight,
            )
            pred_len = int(len(chosen))
            pred_text = decode_chars(crop_imu, crop_sr, chosen, classifier, device)
            rows.append({
                "session": os.path.basename(session_prefix),
                "attempt_idx": seq["attempt_idx"],
                "reference": seq["reference"],
                "true_len": seq["true_len"],
                "pred_len": pred_len,
                "len_correct": int(pred_len == seq["true_len"]),
                "pred_text": pred_text,
                "anchor_debug": debug,
            })

    true = sum(r["len_correct"] for r in rows)
    report = {
        "num_attempts": len(rows),
        "length_accuracy": true / max(len(rows), 1),
        "count_prior_center": count_prior_center,
        "count_prior_weight": args.count_prior_weight,
        "rows": rows,
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        "num_attempts": report["num_attempts"],
        "length_accuracy": report["length_accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
