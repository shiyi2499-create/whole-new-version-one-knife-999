"""
Password Segment Preprocessor (balanced sources)
================================================

Build a **binary** dataset: password_typing (1) vs non_password (0).

Positive samples:  password/len_8 sessions (known password typing)
Negative samples:
  - onset_negative (idle / trackpad / shake / freetyping)
  - single_key / boost (keyboard but not password)
  - **mixed2 typing_1** (free typing inside mixed streams)

Important:
  The main goal of this builder is not to perfectly mirror raw collection counts.
  It explicitly rebalances sources so Stage 1 learns:

    password typing  vs  free typing + other background

  instead of being dominated by single_key negative windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Optional

import numpy as np

try:
    from scipy.signal import resample as scipy_resample
except ImportError as exc:
    raise ImportError("scipy is required: pip install scipy") from exc


# ── Configuration ─────────────────────────────────────────────

DEFAULT_TARGET_RATE_HZ = 190
N_CHANNELS = 6
SEGMENT_WINDOW_MS = 500
SEGMENT_STRIDE_MS = 40
DEFAULT_BALANCE_SEED = 42

# Caps overly large background sources so they do not dominate the task.
MAX_WINDOWS_PER_SOURCE = {
    "single_key_neg": 20_000,
    "boost_neg": 8_000,
}

# Lift hard negatives so Stage 1 is forced to learn password vs free typing.
MIN_WINDOWS_PER_SOURCE = {
    "negative_freetyping": 20_000,
    "mixed2_free_typing": 4_000,
}


# ── Low-level helpers ────────────────────────────────────────

def window_samples(window_ms: int, rate_hz: int) -> int:
    return max(1, int(window_ms / 1000.0 * rate_hz))


def resample_window(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) < 2:
        return np.zeros((target_len, values.shape[1]), dtype=np.float32)
    out = scipy_resample(values, target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)


def load_sensor_csv(path: str) -> np.ndarray:
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append([
                int(row["timestamp_ns"]),
                float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"]),
                float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"]),
            ])
    return np.asarray(rows, dtype=np.float64)


def load_activity_log(path: str) -> list[dict]:
    segments = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            segments.append({
                "start_time_ns": int(row["start_time_ns"]),
                "end_time_ns": int(row["end_time_ns"]),
                "activity": row.get("activity", ""),
                "label": row.get("label", row.get("activity", "")),
                "typing_style": row.get("typing_style", ""),
            })
    return segments


PART_RE = re.compile(r"_part(\d+)_")


def _parse_part(session_prefix: str) -> int:
    m = PART_RE.search(os.path.basename(session_prefix))
    return int(m.group(1)) if m else -1


def _select_latest_complete_sessions(sessions: list[str]) -> list[str]:
    by_part: dict[int, list[str]] = {}
    no_part: list[str] = []
    for sess in sessions:
        part = _parse_part(sess)
        if part < 0:
            no_part.append(sess)
        else:
            by_part.setdefault(part, []).append(sess)
    selected = list(no_part)
    for part in sorted(by_part):
        candidates = sorted(by_part[part])
        complete = [s for s in candidates if os.path.exists(s + "_sensor.csv")]
        selected.append(complete[-1] if complete else candidates[-1])
    return selected


def discover_sessions(dirs, mode_filter="", dedup=True):
    sessions = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  ⚠ Directory not found: {d}")
            continue
        for root, _subdirs, files in os.walk(d):
            for f in sorted(files):
                if f.startswith(".") or f.startswith("._") or not f.endswith("_sensor.csv"):
                    continue
                if mode_filter and f"_{mode_filter}_" not in f:
                    continue
                sessions.append(os.path.join(root, f.replace("_sensor.csv", "")))
    if dedup and sessions:
        before = len(sessions)
        sessions = _select_latest_complete_sessions(sessions)
        if len(sessions) < before:
            print(f"    (dedup: {before} → {len(sessions)} sessions)")
    return sorted(sessions)


def _iterate_window_chunks(sensor, window_ms, stride_ms, target_rate_hz):
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]
    if len(ts_ns) < 10:
        return
    t_start_ns, t_end_ns = int(ts_ns[0]), int(ts_ns[-1])
    win_ns = int(window_ms * 1_000_000)
    stride_ns = int(stride_ms * 1_000_000)
    target_len = window_samples(window_ms, target_rate_hz)
    half = win_ns // 2
    centre = t_start_ns + half
    while centre + half <= t_end_ns:
        i0 = np.searchsorted(ts_ns, centre - half, side="left")
        i1 = np.searchsorted(ts_ns, centre + half, side="right")
        if i1 - i0 >= 4:
            yield centre, resample_window(vals[i0:i1], target_len)
        centre += stride_ns


# ── Windowing ────────────────────────────────────────────────

def _empty_result():
    return {"windows": np.zeros((0, 1, N_CHANNELS), dtype=np.float32),
            "labels": np.array([], dtype=np.int32),
            "times_s": np.array([], dtype=np.float64),
            "sessions": np.array([], dtype=str),
            "sources": np.array([], dtype=str)}


def _merge_results(results):
    if not results:
        return _empty_result()
    return {k: np.concatenate([r[k] for r in results], axis=0)
            for k in ["windows", "labels", "times_s", "sessions", "sources"]}


def _select_indices_for_source(source_indices, target_count, rng):
    if len(source_indices) == 0:
        return np.array([], dtype=np.int64)
    replace = len(source_indices) < target_count
    return rng.choice(source_indices, size=target_count, replace=replace)


def rebalance_by_source(merged, max_per_source, min_per_source, seed):
    if len(merged["labels"]) == 0:
        return merged, {}

    rng = np.random.default_rng(seed)
    sources = merged["sources"]
    unique_sources = sorted(set(sources.tolist()))
    chosen = []
    report = {}

    for src in unique_sources:
        src_idx = np.where(sources == src)[0]
        raw_count = len(src_idx)
        target = raw_count
        action = "kept"

        if src in max_per_source and raw_count > max_per_source[src]:
            target = max_per_source[src]
            action = "capped"
        if src in min_per_source and raw_count < min_per_source[src]:
            target = min_per_source[src]
            action = "oversampled" if action == "kept" else f"{action}+oversampled"

        picked = _select_indices_for_source(src_idx, target, rng)
        if len(picked):
            chosen.append(picked)
        report[src] = {
            "raw": raw_count,
            "final": int(target),
            "action": action,
        }

    if not chosen:
        return _empty_result(), report

    final_idx = np.concatenate(chosen, axis=0)
    rng.shuffle(final_idx)
    balanced = {k: merged[k][final_idx] for k in ["windows", "labels", "times_s", "sessions", "sources"]}
    return balanced, report


def extract_windows_constant_label(sensor, label, window_ms, stride_ms,
                                   target_rate_hz, session_id, source_tag):
    if len(sensor) < 10:
        return _empty_result()
    windows, labels, times_s = [], [], []
    for centre, win in _iterate_window_chunks(sensor, window_ms, stride_ms, target_rate_hz):
        windows.append(win); labels.append(label); times_s.append(centre / 1e9)
    if not windows:
        return _empty_result()
    n = len(windows)
    return {"windows": np.stack(windows), "labels": np.asarray(labels, dtype=np.int32),
            "times_s": np.asarray(times_s, dtype=np.float64),
            "sessions": np.asarray([session_id]*n), "sources": np.asarray([source_tag]*n)}


def extract_windows_in_time_range(sensor, label, start_ns, end_ns,
                                  window_ms, stride_ms, target_rate_hz,
                                  session_id, source_tag):
    """Extract windows only from sensor data within [start_ns, end_ns]."""
    ts_ns = sensor[:, 0]
    mask = (ts_ns >= start_ns) & (ts_ns <= end_ns)
    if mask.sum() < 10:
        return _empty_result()
    return extract_windows_constant_label(
        sensor[mask], label, window_ms, stride_ms,
        target_rate_hz, session_id, source_tag)


# ── Directory processors ─────────────────────────────────────

def process_password_sessions(dirs, window_ms, stride_ms, target_rate_hz):
    sessions = discover_sessions(dirs, mode_filter="free_type")
    print(f"  Found {len(sessions)} password sessions in {dirs}")
    results = []
    for sess in sessions:
        if not os.path.exists(sess + "_sensor.csv"):
            continue
        sensor = load_sensor_csv(sess + "_sensor.csv")
        r = extract_windows_constant_label(sensor, 1, window_ms, stride_ms, target_rate_hz,
                                           os.path.basename(sess), "password_typing")
        if len(r["labels"]):
            results.append(r)
            print(f"    {os.path.basename(sess)}: {len(r['labels'])} windows (password_typing)")
    return _merge_results(results)


def process_keyboard_sessions(dirs, mode_filter, source_tag, window_ms, stride_ms, target_rate_hz):
    sessions = discover_sessions(dirs, mode_filter=mode_filter)
    print(f"  Found {len(sessions)} {source_tag} sessions in {dirs}")
    results = []
    for sess in sessions:
        if not os.path.exists(sess + "_sensor.csv"):
            continue
        sensor = load_sensor_csv(sess + "_sensor.csv")
        r = extract_windows_constant_label(sensor, 0, window_ms, stride_ms, target_rate_hz,
                                           os.path.basename(sess), source_tag)
        if len(r["labels"]):
            results.append(r)
            print(f"    {os.path.basename(sess)}: {len(r['labels'])} windows ({source_tag})")
    return _merge_results(results)


def process_negative_sessions(dirs, window_ms, stride_ms, target_rate_hz):
    results = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if f.startswith(".") or f.startswith("._") or not f.endswith("_sensor.csv"):
                    continue
                sensor = load_sensor_csv(os.path.join(root, f))
                sid = f.replace("_sensor.csv", "")
                tag = f"negative_{os.path.basename(root)}"
                r = extract_windows_constant_label(sensor, 0, window_ms, stride_ms,
                                                   target_rate_hz, sid, tag)
                if len(r["labels"]):
                    results.append(r)
                    print(f"    {sid}: {len(r['labels'])} windows ({tag})")
    return _merge_results(results)


def process_mixed2_free_typing(dirs, window_ms, stride_ms, target_rate_hz):
    """Extract typing_1 (free typing) from mixed2 as HARD NEGATIVE (label=0)."""
    sessions = discover_sessions(dirs, mode_filter="mixed2", dedup=False)
    if not sessions:
        sessions = discover_sessions(dirs, mode_filter="", dedup=False)
    print(f"  Found {len(sessions)} mixed2 sessions for free-typing extraction")
    results = []
    for sess in sessions:
        alog = sess + "_activity_log.csv"
        if not os.path.exists(alog):
            print(f"    ⚠ No activity_log for {os.path.basename(sess)}, skipping")
            continue
        sensor = load_sensor_csv(sess + "_sensor.csv")
        segments = load_activity_log(alog)
        sid = os.path.basename(sess)
        for seg in segments:
            if seg.get("label", "") != "typing_1":
                continue
            r = extract_windows_in_time_range(
                sensor, 0, int(seg["start_time_ns"]), int(seg["end_time_ns"]),
                window_ms, stride_ms, target_rate_hz, sid, "mixed2_free_typing")
            if len(r["labels"]):
                results.append(r)
                print(f"    {sid} typing_1: {len(r['labels'])} windows (free_typing hard neg)")
    return _merge_results(results)


# ── Main builder ─────────────────────────────────────────────

def build_password_segment_dataset(args):
    win_ms, stride_ms = args.window_ms, args.stride_ms
    output = args.output
    if args.project_root and not os.path.isabs(output):
        output = os.path.join(os.path.abspath(args.project_root), output)

    print(f"\n{'='*60}")
    print("  PASSWORD SEGMENT BINARY PREPROCESSOR")
    print(f"  window={win_ms}ms  stride={stride_ms}ms  rate={args.target_rate}Hz")
    print(f"{'='*60}\n")
    parts = []

    if args.password_dirs:
        print("[1/4] Password sessions → POSITIVE...")
        pw = process_password_sessions(args.password_dirs, win_ms, stride_ms, args.target_rate)
        if len(pw["labels"]): parts.append(pw)

    if args.keyboard_dirs:
        print("\n[2/4] Keyboard sessions → NEGATIVE...")
        for mode, tag in [("single_key", "single_key_neg"), ("boost", "boost_neg")]:
            bg = process_keyboard_sessions(args.keyboard_dirs, mode, tag, win_ms, stride_ms, args.target_rate)
            if len(bg["labels"]): parts.append(bg)

    if args.negative_dirs:
        print("\n[3/4] Negative sessions → NEGATIVE...")
        neg = process_negative_sessions(args.negative_dirs, win_ms, stride_ms, args.target_rate)
        if len(neg["labels"]): parts.append(neg)

    if args.mixed2_dirs:
        print("\n[4/4] Mixed2 typing_1 → HARD NEGATIVE (free typing)...")
        ft = process_mixed2_free_typing(args.mixed2_dirs, win_ms, stride_ms, args.target_rate)
        if len(ft["labels"]): parts.append(ft)
    else:
        print("\n[4/4] No mixed2 dirs (skipping free-typing hard negatives)")

    if not parts:
        print("  ❌ No data collected"); sys.exit(1)

    merged_raw = _merge_results(parts)
    merged, balance_report = rebalance_by_source(
        merged_raw,
        MAX_WINDOWS_PER_SOURCE,
        MIN_WINDOWS_PER_SOURCE,
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
        print(f"  {src:30s} raw={item['raw']:7,d}  final={item['final']:7,d}  {item['action']}")

    print(f"\n{'='*60}")
    print("  DATASET SUMMARY (BALANCED)")
    print(f"{'='*60}")
    print(f"  Total:               {len(merged['labels']):,}")
    print(f"  password_typing (1): {n_pos:,}  ({100*n_pos/len(merged['labels']):.1f}%)")
    print(f"  non_password    (0): {n_neg:,}  ({100*n_neg/len(merged['labels']):.1f}%)")
    print(f"  Sources:")
    for src, cnt in sorted(src_counts.items()):
        print(f"    {src:30s} {cnt:8,}")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(output,
        windows=merged["windows"], labels=merged["labels"],
        times_s=merged["times_s"], sessions=merged["sessions"], sources=merged["sources"],
        window_ms=win_ms, stride_ms=stride_ms, label_radius_ms=0,
        target_rate_hz=args.target_rate, n_channels=N_CHANNELS,
        task="password_segment",
        label_names=np.array(["non_password", "password_typing"]))
    print(f"\n  ✓ Saved → {output}\n")


def main():
    p = argparse.ArgumentParser(description="Build binary password segment dataset")
    p.add_argument("--project-root", default="")
    p.add_argument("--password-dirs", nargs="+", default=["data/raw/password/len_8"])
    p.add_argument("--keyboard-dirs", nargs="+", default=["data/raw/single_key", "data/raw/boost"])
    p.add_argument("--negative-dirs", nargs="*", default=["data/raw/onset_negative"])
    p.add_argument("--mixed2-dirs", nargs="*", default=["data/raw/onset_mixed2"],
                   help="mixed2 dirs for extracting typing_1 as free-typing hard negative")
    p.add_argument("--window-ms", type=int, default=SEGMENT_WINDOW_MS)
    p.add_argument("--stride-ms", type=int, default=SEGMENT_STRIDE_MS)
    p.add_argument("--target-rate", type=int, default=DEFAULT_TARGET_RATE_HZ)
    p.add_argument("--balance-seed", type=int, default=DEFAULT_BALANCE_SEED)
    p.add_argument("--output", default="data/processed/password_segment_dataset.npz")
    args = p.parse_args()
    if args.project_root:
        root = os.path.abspath(args.project_root)
        for attr in ["password_dirs", "keyboard_dirs", "negative_dirs", "mixed2_dirs"]:
            setattr(args, attr, [os.path.join(root, d) if not os.path.isabs(d) else d for d in getattr(args, attr)])
        if not os.path.isabs(args.output):
            args.output = os.path.join(root, args.output)
    build_password_segment_dataset(args)


if __name__ == "__main__":
    main()
