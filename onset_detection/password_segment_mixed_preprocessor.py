"""
Password Segment Dataset Builder (mixed-aware)
=============================================

Build a binary coarse-region dataset for full-stream password detection.

Positive windows:
  - password/len_8 full sessions
  - mixed_single / mixed_retry password blocks (typing_2, typing_3)

Negative windows:
  - onset_negative
  - single_key
  - mixed2 typing_1
  - mixed_single / mixed_retry non-password context
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from password_segment_preprocessor import (
    DEFAULT_BALANCE_SEED,
    DEFAULT_TARGET_RATE_HZ,
    MAX_WINDOWS_PER_SOURCE,
    MIN_WINDOWS_PER_SOURCE,
    SEGMENT_STRIDE_MS,
    SEGMENT_WINDOW_MS,
    N_CHANNELS,
    _empty_result,
    _merge_results,
    build_password_segment_dataset,  # noqa: F401  # kept for reference import path stability
    discover_sessions,
    extract_windows_constant_label,
    extract_windows_in_time_range,
    load_activity_log,
    load_sensor_csv,
    process_keyboard_sessions,
    process_mixed2_free_typing,
    process_negative_sessions,
    process_password_sessions,
    rebalance_by_source,
)


POS_LABELS = {"typing_2", "typing_3"}


def process_mixed_context_and_password(
    dirs,
    mode_tag: str,
    window_ms: int,
    stride_ms: int,
    target_rate_hz: int,
):
    sessions = discover_sessions(dirs, mode_filter=mode_tag, dedup=False)
    if not sessions:
        sessions = discover_sessions(dirs, mode_filter="", dedup=False)
    print(f"  Found {len(sessions)} {mode_tag} sessions")

    pos_results = []
    neg_results = []

    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        alog_path = sess + "_activity_log.csv"
        if not os.path.exists(sensor_path) or not os.path.exists(alog_path):
            continue

        sensor = load_sensor_csv(sensor_path)
        segments = load_activity_log(alog_path)
        sid = os.path.basename(sess)

        for seg in segments:
            label = seg.get("label", "")
            typing_style = seg.get("typing_style", "")
            start_ns = int(seg["start_time_ns"])
            end_ns = int(seg["end_time_ns"])

            if label in POS_LABELS or typing_style == "password":
                source_tag = f"{mode_tag}_password"
                r = extract_windows_in_time_range(
                    sensor, 1, start_ns, end_ns,
                    window_ms, stride_ms, target_rate_hz,
                    sid, source_tag,
                )
                if len(r["labels"]):
                    pos_results.append(r)
            else:
                # Keep the real mixed-stream context as negative supervision.
                source_tag = f"{mode_tag}_{label or 'context'}"
                r = extract_windows_in_time_range(
                    sensor, 0, start_ns, end_ns,
                    window_ms, stride_ms, target_rate_hz,
                    sid, source_tag,
                )
                if len(r["labels"]):
                    neg_results.append(r)

    return _merge_results(pos_results), _merge_results(neg_results)


def build_mixed_password_segment_dataset(args):
    win_ms, stride_ms = args.window_ms, args.stride_ms
    output = args.output
    if args.project_root and not os.path.isabs(output):
        output = os.path.join(os.path.abspath(args.project_root), output)

    print(f"\n{'='*60}")
    print("  PASSWORD SEGMENT MIXED-AWARE PREPROCESSOR")
    print(f"  window={win_ms}ms  stride={stride_ms}ms  rate={args.target_rate}Hz")
    print(f"{'='*60}\n")

    parts = []
    balance_max = dict(MAX_WINDOWS_PER_SOURCE)
    balance_min = dict(MIN_WINDOWS_PER_SOURCE)

    if args.password_dirs:
        print("[1/6] Password standalone sessions → POSITIVE...")
        pw = process_password_sessions(args.password_dirs, win_ms, stride_ms, args.target_rate)
        if len(pw["labels"]):
            parts.append(pw)

    if args.keyboard_dirs:
        print("\n[2/6] Keyboard sessions → NEGATIVE...")
        bg = process_keyboard_sessions(args.keyboard_dirs, "single_key", "single_key_neg", win_ms, stride_ms, args.target_rate)
        if len(bg["labels"]):
            parts.append(bg)

    if args.negative_dirs:
        print("\n[3/6] Negative sessions → NEGATIVE...")
        neg = process_negative_sessions(args.negative_dirs, win_ms, stride_ms, args.target_rate)
        if len(neg["labels"]):
            parts.append(neg)

    if args.mixed2_dirs:
        print("\n[4/6] Mixed2 typing_1 → HARD NEGATIVE...")
        ft = process_mixed2_free_typing(args.mixed2_dirs, win_ms, stride_ms, args.target_rate)
        if len(ft["labels"]):
            parts.append(ft)

    if args.mixed_single_dirs:
        print("\n[5/6] mixed_single_training → password/context...")
        pos, neg = process_mixed_context_and_password(args.mixed_single_dirs, "mixed_single_training", win_ms, stride_ms, args.target_rate)
        if len(pos["labels"]):
            parts.append(pos)
            balance_min["mixed_single_training_password"] = max(
                balance_min.get("mixed_single_training_password", 0),
                max(1500, len(pos["labels"]))
            )
        if len(neg["labels"]):
            parts.append(neg)

    if args.mixed_retry_dirs:
        print("\n[6/6] mixed_retry_training → password/context...")
        pos, neg = process_mixed_context_and_password(args.mixed_retry_dirs, "mixed_retry_training", win_ms, stride_ms, args.target_rate)
        if len(pos["labels"]):
            parts.append(pos)
            balance_min["mixed_retry_training_password"] = max(
                balance_min.get("mixed_retry_training_password", 0),
                max(2000, len(pos["labels"]))
            )
        if len(neg["labels"]):
            parts.append(neg)

    if not parts:
        print("  ❌ No data collected")
        sys.exit(1)

    merged_raw = _merge_results(parts)
    merged, balance_report = rebalance_by_source(
        merged_raw,
        balance_max,
        balance_min,
        args.balance_seed,
    )

    n_pos = int(merged["labels"].sum())
    n_neg = len(merged["labels"]) - n_pos
    src_counts = {}
    for s in merged["sources"]:
        src_counts[s] = src_counts.get(s, 0) + 1

    print(f"\n{'='*60}")
    print("  SOURCE BALANCING")
    print(f"{'='*60}")
    for src in sorted(balance_report):
        item = balance_report[src]
        print(f"  {src:34s} raw={item['raw']:7,d}  final={item['final']:7,d}  {item['action']}")

    print(f"\n{'='*60}")
    print("  DATASET SUMMARY (BALANCED)")
    print(f"{'='*60}")
    print(f"  Total:               {len(merged['labels']):,}")
    print(f"  password_typing (1): {n_pos:,}  ({100*n_pos/len(merged['labels']):.1f}%)")
    print(f"  non_password    (0): {n_neg:,}  ({100*n_neg/len(merged['labels']):.1f}%)")
    print("  Sources:")
    for src, cnt in sorted(src_counts.items()):
        print(f"    {src:34s} {cnt:8,}")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(
        output,
        windows=merged["windows"],
        labels=merged["labels"],
        times_s=merged["times_s"],
        sessions=merged["sessions"],
        sources=merged["sources"],
        window_ms=win_ms,
        stride_ms=stride_ms,
        label_radius_ms=0,
        target_rate_hz=args.target_rate,
        n_channels=N_CHANNELS,
        task="password_segment",
        label_names=np.array(["non_password", "password_typing"]),
    )
    print(f"\n  ✓ Saved → {output}\n")


def main():
    p = argparse.ArgumentParser(description="Build mixed-aware binary password segment dataset")
    p.add_argument("--project-root", default="")
    p.add_argument("--password-dirs", nargs="+", default=["data/raw/password/len_8"])
    p.add_argument("--keyboard-dirs", nargs="+", default=["data/raw/single_key"])
    p.add_argument("--negative-dirs", nargs="*", default=["data/raw/onset_negative"])
    p.add_argument("--mixed2-dirs", nargs="*", default=["data/raw/onset_mixed2"])
    p.add_argument("--mixed-single-dirs", nargs="*", default=["data/raw/mixed_single_training"])
    p.add_argument("--mixed-retry-dirs", nargs="*", default=["data/raw/mixed_retry_training"])
    p.add_argument("--window-ms", type=int, default=SEGMENT_WINDOW_MS)
    p.add_argument("--stride-ms", type=int, default=SEGMENT_STRIDE_MS)
    p.add_argument("--target-rate", type=int, default=DEFAULT_TARGET_RATE_HZ)
    p.add_argument("--balance-seed", type=int, default=DEFAULT_BALANCE_SEED)
    p.add_argument("--output", default="data/processed/password_segment_mixed_dataset.npz")
    args = p.parse_args()

    if args.project_root:
        root = os.path.abspath(args.project_root)
        for attr in [
            "password_dirs",
            "keyboard_dirs",
            "negative_dirs",
            "mixed2_dirs",
            "mixed_single_dirs",
            "mixed_retry_dirs",
        ]:
            setattr(args, attr, [os.path.join(root, d) if not os.path.isabs(d) else d for d in getattr(args, attr)])
        if not os.path.isabs(args.output):
            args.output = os.path.join(root, args.output)

    build_mixed_password_segment_dataset(args)


if __name__ == "__main__":
    main()
