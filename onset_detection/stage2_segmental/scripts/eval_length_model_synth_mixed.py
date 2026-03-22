#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader, discover_sessions
from onset_detection.stage2_segmental.length_model import (
    compute_region_length_features,
    estimate_sample_rate_hz,
    load_length_model,
    normalize_text,
    read_csv_rows,
)
from onset_detection.stage2_segmental.scripts.eval_overlap_single_coarse_energy_cls import _build_length_subregion_from_energy


def iter_success_attempt_crops(password_dir: str, true_len: int, pre_ms: float = 300.0, post_ms: float = 300.0):
    for sensor_path in sorted(Path(password_dir).glob("*_sensor.csv")):
        prefix = str(sensor_path).replace("_sensor.csv", "")
        if not (Path(prefix + "_attempts.csv").exists() and Path(prefix + "_events.csv").exists()):
            continue
        attempts = read_csv_rows(prefix + "_attempts.csv")
        events = read_csv_rows(prefix + "_events.csv")
        sensor = read_csv_rows(prefix + "_sensor.csv")
        if not attempts or not events or not sensor:
            continue

        ts = np.asarray([int(r["timestamp_ns"]) for r in sensor], dtype=np.int64)
        cols = [c for c in sensor[0].keys() if c != "timestamp_ns"][:6]
        imu = np.asarray([[float(r[c]) for c in cols] for r in sensor], dtype=np.float32)
        presses = []
        for r in events:
            if str(r.get("event_type", "")).lower() != "press":
                continue
            key = str(r.get("key") or r.get("key_name") or "").lower()
            presses.append((int(r["timestamp_ns"]), key))
        enter_idx = [i for i, (_, k) in enumerate(presses) if k in ("enter", "return")]
        start_idx = 0
        success_rows = []
        for row in attempts:
            prompt = normalize_text(row.get("prompt_text", ""))
            typed = normalize_text(row.get("typed_text", ""))
            match = str(row.get("match") or row.get("match_status") or row.get("status") or "").upper()
            ok = bool(typed) and (match == "YES" or (not match and (not prompt or prompt == typed)))
            if ok:
                success_rows.append(row)
        for att_idx, row in enumerate(success_rows):
            if att_idx >= len(enter_idx):
                break
            end_i = enter_idx[att_idx]
            seq = presses[start_idx:end_i + 1]
            start_idx = end_i + 1
            char_ts = [t for t, k in seq if k not in ("enter", "return")]
            if len(char_ts) != true_len:
                continue
            lo_ns = char_ts[0] - int(round(pre_ms * 1e6))
            hi_ns = seq[-1][0] + int(round(post_ms * 1e6))
            l = np.searchsorted(ts, lo_ns, side="left")
            r = np.searchsorted(ts, hi_ns, side="right")
            crop_ts = ts[max(0, l):min(len(ts), r)]
            crop_imu = imu[max(0, l):min(len(ts), r)]
            if len(crop_imu) < 10:
                continue
            yield {
                "prefix": prefix,
                "true_len": int(true_len),
                "crop_ts": crop_ts,
                "crop_imu": crop_imu,
            }


def collect_background_clips(mixed_dir: str, clip_s: float = 8.0):
    clips = []
    for session_path in discover_sessions(mixed_dir):
        loader = SessionLoader(session_path)
        ts, imu = loader.get_imu()
        acts = loader.get_activity_log()
        if len(ts) == 0 or not acts:
            continue
        sr = estimate_sample_rate_hz(ts)
        clip_frames = max(20, int(round(sr * clip_s)))
        for row in acts:
            label = str(row.get("label", ""))
            typing_style = str(row.get("typing_style", ""))
            act = str(row.get("activity", ""))
            if act == "keyboard" and typing_style == "password":
                continue
            start_ns = int(row.get("start_ns"))
            end_ns = int(row.get("end_ns"))
            l = np.searchsorted(ts, start_ns, side="left")
            r = np.searchsorted(ts, end_ns, side="right")
            seg = imu[l:r]
            if len(seg) < clip_frames:
                continue
            for off in (0, max(0, len(seg) // 2 - clip_frames // 2)):
                sub = seg[off:off + clip_frames]
                if len(sub) == clip_frames:
                    clips.append(np.asarray(sub, dtype=np.float32))
    return clips


def make_synthetic_region(bg_pre: np.ndarray, attempt_imu: np.ndarray, bg_post: np.ndarray, sample_rate_hz: float):
    synth = np.concatenate([bg_pre, attempt_imu, bg_post], axis=0)
    dt_ns = int(round(1e9 / max(sample_rate_hz, 1.0)))
    ts = np.arange(len(synth), dtype=np.int64) * dt_ns
    return synth, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--password-dataset", action="append", required=True, help="LEN:DIR")
    ap.add_argument("--mixed-bg-dir", required=True)
    ap.add_argument("--length-model", required=True)
    ap.add_argument("--num-samples-per-len", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model, labels, meta = load_length_model(args.length_model)
    feature_mode = str(meta.get("feature_mode", "legacy_time"))
    bg_clips = collect_background_clips(args.mixed_bg_dir)
    if len(bg_clips) < 2:
        raise RuntimeError("Not enough background clips")

    rows = []
    for spec in args.password_dataset:
        true_len, path = spec.split(":", 1)
        true_len = int(true_len)
        attempts = list(iter_success_attempt_crops(path, true_len))
        rng.shuffle(attempts)
        for ex in attempts[:args.num_samples_per_len]:
            sr = estimate_sample_rate_hz(ex["crop_ts"])
            bg_pre = rng.choice(bg_clips)
            bg_post = rng.choice(bg_clips)
            synth_imu, synth_ts = make_synthetic_region(bg_pre, ex["crop_imu"], bg_post, sr)

            pred_full = int(model.predict(compute_region_length_features(synth_imu, synth_ts, feature_mode=feature_mode).reshape(1, -1))[0])
            subregion, sub_debug = _build_length_subregion_from_energy(synth_imu, sr)
            if subregion is not None:
                lo, hi = subregion
                sub_imu = synth_imu[lo:hi]
                sub_ts = synth_ts[lo:hi]
                pred_cluster = int(model.predict(compute_region_length_features(sub_imu, sub_ts, feature_mode=feature_mode).reshape(1, -1))[0])
            else:
                pred_cluster = None
            rows.append({
                "true_len": true_len,
                "pred_full": pred_full,
                "pred_cluster": pred_cluster,
                "subregion_debug": sub_debug,
            })

    by_len = {}
    for r in rows:
        by_len.setdefault(r["true_len"], []).append(r)

    report = {"feature_mode": feature_mode, "rows": rows, "summary": {}}
    for true_len, items in sorted(by_len.items()):
        n = len(items)
        full_acc = sum(int(x["pred_full"] == true_len) for x in items) / max(n, 1)
        cluster_acc = sum(int(x["pred_cluster"] == true_len) for x in items) / max(n, 1)
        report["summary"][str(true_len)] = {
            "num_samples": n,
            "full_region_acc": full_acc,
            "cluster_region_acc": cluster_acc,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
