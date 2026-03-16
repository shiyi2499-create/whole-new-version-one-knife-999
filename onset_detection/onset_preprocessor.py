"""
Onset Preprocessor
==================
Build a sliding-window binary-classification dataset for onset detection.

Positive windows: centred near a keystroke event (within ±label_radius_ms)
Negative windows: inter-key gaps, idle, trackpad, etc.

Data sources:
  1. Existing single_key / boost / password sessions (auto-label from events.csv)
  2. New negative-only sessions recorded by onset_collector.py

Output: onset_dataset.npz containing:
  - windows   (N, window_samples, 6)   float32
  - labels    (N,)                      int32   {0, 1}
  - times_s   (N,)                      float64 centre time of each window
  - sessions  (N,)                      str     source session ID
  - sources   (N,)                      str     source type tag

Run:
  python3 onset_preprocessor.py --keyboard-dirs data/raw/single_key data/raw/boost
  python3 onset_preprocessor.py --keyboard-dirs data/raw/single_key data/raw/boost \\
                                --password-dirs data/raw/password/len_8 \\
                                --negative-dirs data/raw/onset_negative
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import json
import numpy as np
from collections import defaultdict
from typing import Optional

try:
    from scipy.signal import resample as scipy_resample
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


# ── Configuration ─────────────────────────────────────────────

DEFAULT_TARGET_RATE_HZ = 190
DEFAULT_WINDOW_MS = 150         # detection window length
DEFAULT_STRIDE_MS = 25          # sliding stride
DEFAULT_LABEL_RADIUS_MS = 30    # ±ms from event to mark as positive
N_CHANNELS = 6                  # accel_xyz + gyro_xyz


def window_samples(window_ms: int, rate_hz: int) -> int:
    """Number of samples in a window of given ms at given rate."""
    return max(1, int(window_ms / 1000.0 * rate_hz))


# ── Session loading ──────────────────────────────────────────

def load_sensor_csv(path: str) -> np.ndarray:
    """Load sensor.csv → (N, 7) array [timestamp_ns, ax, ay, az, gx, gy, gz]."""
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append([
                int(row["timestamp_ns"]),
                float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"]),
                float(row["gyro_x"]),  float(row["gyro_y"]),  float(row["gyro_z"]),
            ])
    return np.asarray(rows, dtype=np.float64)


def load_events_csv(path: str, press_only: bool = True) -> np.ndarray:
    """Load events.csv → (M,) array of press-event timestamps in nanoseconds."""
    timestamps = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            if press_only and row["event_type"] != "press":
                continue
            timestamps.append(int(row["timestamp_ns"]))
    return np.asarray(timestamps, dtype=np.int64)


PART_RE = re.compile(r"_part(\d+)_")


def _parse_part(session_prefix: str) -> int:
    """Extract part number from session prefix, or -1 if absent."""
    m = PART_RE.search(os.path.basename(session_prefix))
    return int(m.group(1)) if m else -1


def _session_is_complete(session_prefix: str) -> bool:
    """A session is complete if it has sensor, events, and prompts files."""
    required = [session_prefix + "_sensor.csv", session_prefix + "_events.csv"]
    return all(os.path.exists(p) for p in required)


def _select_latest_complete_sessions(sessions: list[str]) -> list[str]:
    """
    De-duplicate sessions: when multiple sessions exist for the same
    part number, keep only the latest complete one (matching the password
    adaptation script's behaviour).
    """
    by_part: dict[int, list[str]] = {}
    no_part: list[str] = []
    for sess in sessions:
        part = _parse_part(sess)
        if part < 0:
            no_part.append(sess)
        else:
            by_part.setdefault(part, []).append(sess)

    selected = list(no_part)  # sessions without part numbers pass through
    for part in sorted(by_part):
        candidates = sorted(by_part[part])  # alphabetical = chronological
        complete = [s for s in candidates if _session_is_complete(s)]
        if complete:
            selected.append(complete[-1])  # latest complete
        elif candidates:
            selected.append(candidates[-1])  # fallback: latest incomplete
    return selected


def discover_sessions(
    dirs: list[str],
    mode_filter: str = "",
    dedup: bool = True,
) -> list[str]:
    """
    Find session prefixes (paths without _sensor.csv suffix).

    Filters:
      - Skips macOS ._* resource fork files
      - Skips files that don't match mode_filter
      - When dedup=True, keeps only the latest complete session per part
    """
    sessions = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  ⚠ Directory not found: {d}")
            continue
        for f in sorted(os.listdir(d)):
            # Skip macOS resource forks and hidden files
            if f.startswith(".") or f.startswith("._"):
                continue
            if not f.endswith("_sensor.csv"):
                continue
            if mode_filter and f"_{mode_filter}_" not in f:
                continue
            prefix = os.path.join(d, f.replace("_sensor.csv", ""))
            sessions.append(prefix)

    if dedup and sessions:
        before = len(sessions)
        sessions = _select_latest_complete_sessions(sessions)
        if len(sessions) < before:
            print(f"    (dedup: {before} → {len(sessions)} sessions)")

    return sessions


# ── Core: extract sliding windows from one session ────────────

def resample_window(values: np.ndarray, target_len: int) -> np.ndarray:
    """FFT-based resampling to fixed length, matching preprocessor.py."""
    if len(values) < 2:
        return np.zeros((target_len, values.shape[1]), dtype=np.float32)
    out = scipy_resample(values, target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)


def extract_sliding_windows(
    sensor: np.ndarray,
    event_times_ns: np.ndarray,
    window_ms: int = DEFAULT_WINDOW_MS,
    stride_ms: int = DEFAULT_STRIDE_MS,
    label_radius_ms: int = DEFAULT_LABEL_RADIUS_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
    session_id: str = "",
    source_tag: str = "",
) -> dict:
    """
    Slide a window across the sensor stream and label each position.

    Returns dict with arrays: windows, labels, times_s, sessions, sources
    """
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]  # (N, 6)

    if len(ts_ns) < 10:
        return _empty_result()

    t_start_ns = int(ts_ns[0])
    t_end_ns = int(ts_ns[-1])

    win_ns = int(window_ms * 1_000_000)
    stride_ns = int(stride_ms * 1_000_000)
    radius_ns = int(label_radius_ms * 1_000_000)

    target_len = window_samples(window_ms, target_rate_hz)

    # Sort event times for efficient lookup
    events_sorted = np.sort(event_times_ns) if len(event_times_ns) > 0 else np.array([], dtype=np.int64)

    windows = []
    labels = []
    times_s = []

    half_win_ns = win_ns // 2
    centre = t_start_ns + half_win_ns

    while centre + half_win_ns <= t_end_ns:
        w_start = centre - half_win_ns
        w_end = centre + half_win_ns

        idx_start = np.searchsorted(ts_ns, w_start, side="left")
        idx_end = np.searchsorted(ts_ns, w_end, side="right")

        n_raw = idx_end - idx_start
        if n_raw < 4:
            centre += stride_ns
            continue

        # Resample to fixed length
        chunk = vals[idx_start:idx_end]
        win = resample_window(chunk, target_len)
        windows.append(win)

        # Label: is there a keystroke event within ±radius of window centre?
        if len(events_sorted) > 0:
            # Binary search for nearest event
            insert_pos = np.searchsorted(events_sorted, centre)
            near_dists = []
            for offset in [insert_pos - 1, insert_pos]:
                if 0 <= offset < len(events_sorted):
                    near_dists.append(abs(int(events_sorted[offset]) - int(centre)))
            min_dist = min(near_dists) if near_dists else radius_ns + 1
            label = 1 if min_dist <= radius_ns else 0
        else:
            label = 0

        labels.append(label)
        times_s.append(centre / 1e9)

        centre += stride_ns

    if not windows:
        return _empty_result()

    return {
        "windows": np.stack(windows),             # (N, target_len, 6)
        "labels": np.array(labels, dtype=np.int32),
        "times_s": np.array(times_s, dtype=np.float64),
        "sessions": np.array([session_id] * len(windows)),
        "sources": np.array([source_tag] * len(windows)),
    }


def _empty_result():
    return {
        "windows": np.zeros((0, 1, N_CHANNELS), dtype=np.float32),
        "labels": np.array([], dtype=np.int32),
        "times_s": np.array([], dtype=np.float64),
        "sessions": np.array([]),
        "sources": np.array([]),
    }


# ── Process an entire directory of sessions ───────────────────

def process_keyboard_sessions(
    dirs: list[str],
    session_type: str,
    **kwargs,
) -> dict:
    """Process sessions that have both sensor.csv and events.csv."""
    mode_filter = "single_key" if session_type == "single_key" else "free_type"
    sessions = discover_sessions(dirs, mode_filter=mode_filter)
    print(f"  Found {len(sessions)} {session_type} sessions in {dirs}")

    all_results = []
    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        events_path = sess + "_events.csv"
        if not os.path.exists(events_path):
            continue

        sensor = load_sensor_csv(sensor_path)
        events = load_events_csv(events_path, press_only=True)
        sess_id = os.path.basename(sess)

        result = extract_sliding_windows(
            sensor, events,
            session_id=sess_id,
            source_tag=session_type,
            **kwargs,
        )
        if len(result["labels"]) > 0:
            all_results.append(result)
            n_pos = int(result["labels"].sum())
            n_neg = len(result["labels"]) - n_pos
            print(f"    {sess_id}: {n_pos} pos + {n_neg} neg = {len(result['labels'])} windows")

    return _merge_results(all_results)


def process_negative_sessions(
    dirs: list[str],
    **kwargs,
) -> dict:
    """Process pure negative sessions (no events.csv needed)."""
    sessions = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        # Walk subdirectories (e.g. onset_negative/idle/, onset_negative/trackpad_click/)
        for root, _subdirs, files in os.walk(d):
            for f in sorted(files):
                if f.startswith(".") or f.startswith("._"):
                    continue
                if f.endswith("_sensor.csv"):
                    sessions.append(os.path.join(root, f.replace("_sensor.csv", "")))

    print(f"  Found {len(sessions)} negative sessions in {dirs}")

    all_results = []
    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        sensor = load_sensor_csv(sensor_path)
        events = np.array([], dtype=np.int64)  # no events → all negative
        sess_id = os.path.basename(sess)

        # Infer activity type from directory name
        parent_dir = os.path.basename(os.path.dirname(sess))
        source_tag = f"negative_{parent_dir}"

        result = extract_sliding_windows(
            sensor, events,
            session_id=sess_id,
            source_tag=source_tag,
            **kwargs,
        )
        if len(result["labels"]) > 0:
            all_results.append(result)
            print(f"    {sess_id}: {len(result['labels'])} neg windows ({source_tag})")

    return _merge_results(all_results)


def _merge_results(results: list[dict]) -> dict:
    if not results:
        return _empty_result()
    return {
        key: np.concatenate([r[key] for r in results], axis=0)
        for key in ["windows", "labels", "times_s", "sessions", "sources"]
    }


# ── Main CLI ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build onset detection training dataset")

    parser.add_argument(
        "--project-root", default="",
        help="Project root directory. All relative paths resolve from here. "
             "Defaults to current working directory.",
    )
    parser.add_argument(
        "--keyboard-dirs", nargs="+", default=["data/raw/single_key", "data/raw/boost"],
        help="Directories containing single_key sessions (sensor + events)",
    )
    parser.add_argument(
        "--password-dirs", nargs="+", default=["data/raw/password/len_8"],
        help="Directories containing password sessions (sensor + events)",
    )
    parser.add_argument(
        "--negative-dirs", nargs="*", default=[],
        help="Directories containing negative-only sessions (sensor only)",
    )
    parser.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--stride-ms", type=int, default=DEFAULT_STRIDE_MS)
    parser.add_argument("--label-radius-ms", type=int, default=DEFAULT_LABEL_RADIUS_MS)
    parser.add_argument("--target-rate", type=int, default=DEFAULT_TARGET_RATE_HZ)
    parser.add_argument(
        "--output", default="data/processed/onset_dataset.npz",
        help="Output path for the onset dataset",
    )

    args = parser.parse_args()

    # Resolve all relative paths from project root
    if args.project_root:
        root = os.path.abspath(args.project_root)
        args.keyboard_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                              for d in args.keyboard_dirs]
        args.password_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                              for d in args.password_dirs]
        if args.negative_dirs:
            args.negative_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                                  for d in args.negative_dirs]
        if not os.path.isabs(args.output):
            args.output = os.path.join(root, args.output)

    common = dict(
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        label_radius_ms=args.label_radius_ms,
        target_rate_hz=args.target_rate,
    )

    print(f"\n{'='*60}")
    print(f"  ONSET PREPROCESSOR")
    print(f"  window={args.window_ms}ms  stride={args.stride_ms}ms  "
          f"label_radius=±{args.label_radius_ms}ms  rate={args.target_rate}Hz")
    print(f"{'='*60}\n")

    all_parts = []

    # 1. Single-key sessions
    if args.keyboard_dirs:
        print("[1/3] Processing single_key sessions...")
        sk = process_keyboard_sessions(args.keyboard_dirs, "single_key", **common)
        if len(sk["labels"]) > 0:
            all_parts.append(sk)

    # 2. Password sessions
    if args.password_dirs:
        print("\n[2/3] Processing password sessions...")
        pw = process_keyboard_sessions(args.password_dirs, "free_type", **common)
        if len(pw["labels"]) > 0:
            all_parts.append(pw)

    # 3. Negative sessions
    if args.negative_dirs:
        print("\n[3/3] Processing negative sessions...")
        neg = process_negative_sessions(args.negative_dirs, **common)
        if len(neg["labels"]) > 0:
            all_parts.append(neg)
    else:
        print("\n[3/3] No negative-only directories specified (skipping)")

    if not all_parts:
        print("  ❌ No data collected!")
        sys.exit(1)

    merged = _merge_results(all_parts)

    # Summary
    n_total = len(merged["labels"])
    n_pos = int(merged["labels"].sum())
    n_neg = n_total - n_pos
    win_len = window_samples(args.window_ms, args.target_rate)

    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows:   {n_total:,}")
    print(f"  Positive (onset): {n_pos:,}  ({100*n_pos/max(n_total,1):.1f}%)")
    print(f"  Negative:        {n_neg:,}  ({100*n_neg/max(n_total,1):.1f}%)")
    print(f"  Window shape:    ({win_len}, {N_CHANNELS})")
    print(f"  Pos/neg ratio:   1:{n_neg/max(n_pos,1):.1f}")

    # Source breakdown
    source_counts = defaultdict(lambda: {"pos": 0, "neg": 0})
    for src, lbl in zip(merged["sources"], merged["labels"]):
        if lbl == 1:
            source_counts[src]["pos"] += 1
        else:
            source_counts[src]["neg"] += 1
    print(f"\n  By source:")
    for src in sorted(source_counts):
        c = source_counts[src]
        print(f"    {src}: {c['pos']} pos + {c['neg']} neg = {c['pos']+c['neg']}")

    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.output,
        windows=merged["windows"],
        labels=merged["labels"],
        times_s=merged["times_s"],
        sessions=merged["sessions"],
        sources=merged["sources"],
        # Metadata
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        label_radius_ms=args.label_radius_ms,
        target_rate_hz=args.target_rate,
        n_channels=N_CHANNELS,
    )
    print(f"\n  ✓ Saved → {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
