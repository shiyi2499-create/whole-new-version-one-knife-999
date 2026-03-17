"""
Onset Preprocessor
==================
Build sliding-window datasets for:
  1. onset detection    (binary: keystroke onset vs not)
  2. activity segment   (binary: keyboard-active frame vs inactive frame)

Data sources:
  1. Existing single_key / boost / password sessions (auto-label from events.csv)
  2. New negative-only sessions from onset_collector.py
  3. Mixed2 sessions with activity_log.csv for ground-truth episode boundaries

Data source roles for activity segmentation task:
  - **mixed2 sessions are the PRIMARY source** for activity boundary
    supervision.  These provide real start/end segment boundaries from
    the activity_log.csv, so the model learns *where* keyboard activity
    begins and ends relative to other activities.
  - single_key / password sessions are used as **intra-episode positive
    supplements**: the entire session is labelled keyboard-active.  They
    add more positive-class IMU diversity but do NOT contribute real
    start/end boundary supervision (the boundary is synthetic — the
    session boundaries are not ecologically valid activity transitions).
  - negative sessions contribute inactive-class samples with no boundary
    information.

Output: onset_dataset.npz or activity_dataset.npz
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
DEFAULT_WINDOW_MS = 150         # onset detection window
DEFAULT_STRIDE_MS = 25          # onset sliding stride
DEFAULT_LABEL_RADIUS_MS = 30    # ±ms from event to mark as positive
N_CHANNELS = 6                  # accel_xyz + gyro_xyz

# Activity segmenter uses wider windows
ACTIVITY_WINDOW_MS = 400        # activity detection window
ACTIVITY_STRIDE_MS = 50         # activity sliding stride


def window_samples(window_ms: int, rate_hz: int) -> int:
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


def load_activity_log(path: str) -> list[dict]:
    """Load activity_log.csv → list of segment dicts."""
    segments = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg = {
                "start_time_ns": int(row["start_time_ns"]),
                "end_time_ns": int(row["end_time_ns"]),
                "activity": row.get("activity", ""),
                "label": row.get("label", row.get("activity", "")),
                "typing_style": row.get("typing_style", ""),
            }
            prompts_raw = row.get("prompts", row.get("prompt", ""))
            if prompts_raw and prompts_raw.startswith("["):
                try:
                    seg["prompts"] = json.loads(prompts_raw)
                except json.JSONDecodeError:
                    seg["prompts"] = []
            else:
                seg["prompts"] = [prompts_raw] if prompts_raw else []
            segments.append(seg)
    return segments


PART_RE = re.compile(r"_part(\d+)_")


def _parse_part(session_prefix: str) -> int:
    m = PART_RE.search(os.path.basename(session_prefix))
    return int(m.group(1)) if m else -1


def _session_is_complete(session_prefix: str) -> bool:
    required = [session_prefix + "_sensor.csv", session_prefix + "_events.csv"]
    return all(os.path.exists(p) for p in required)


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
        complete = [s for s in candidates if _session_is_complete(s)]
        if complete:
            selected.append(complete[-1])
        elif candidates:
            selected.append(candidates[-1])
    return selected


def discover_sessions(
    dirs: list[str],
    mode_filter: str = "",
    dedup: bool = True,
) -> list[str]:
    sessions = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  ⚠ Directory not found: {d}")
            continue
        for f in sorted(os.listdir(d)):
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


# ── Core: sliding windows for ONSET detection ────────────────

def resample_window(values: np.ndarray, target_len: int) -> np.ndarray:
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
    Slide a window across the sensor stream and label each position
    for onset detection (positive if near a keystroke event).
    """
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]

    if len(ts_ns) < 10:
        return _empty_result()

    t_start_ns = int(ts_ns[0])
    t_end_ns = int(ts_ns[-1])

    win_ns = int(window_ms * 1_000_000)
    stride_ns = int(stride_ms * 1_000_000)
    radius_ns = int(label_radius_ms * 1_000_000)

    target_len = window_samples(window_ms, target_rate_hz)

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

        chunk = vals[idx_start:idx_end]
        win = resample_window(chunk, target_len)
        windows.append(win)

        # Label: is there a keystroke event within ±radius of window centre?
        if len(events_sorted) > 0:
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
        "windows": np.stack(windows),
        "labels": np.array(labels, dtype=np.int32),
        "times_s": np.array(times_s, dtype=np.float64),
        "sessions": np.array([session_id] * len(windows)),
        "sources": np.array([source_tag] * len(windows)),
    }


# ── Core: sliding windows for ACTIVITY segmentation ─────────

def extract_activity_windows(
    sensor: np.ndarray,
    activity_segments: list[dict],
    window_ms: int = ACTIVITY_WINDOW_MS,
    stride_ms: int = ACTIVITY_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
    session_id: str = "",
    source_tag: str = "",
) -> dict:
    """
    Slide a window across the sensor stream and label each position
    for activity segmentation (positive if the window centre falls within
    any keyboard-activity segment).

    activity_segments: list of dicts with 'start_time_ns', 'end_time_ns', 'activity'
    A window is labelled positive (1) if its centre falls within a segment
    whose 'activity' == 'keyboard'.
    """
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]

    if len(ts_ns) < 10:
        return _empty_activity_result()

    t_start_ns = int(ts_ns[0])
    t_end_ns = int(ts_ns[-1])

    win_ns = int(window_ms * 1_000_000)
    stride_ns = int(stride_ms * 1_000_000)
    target_len = window_samples(window_ms, target_rate_hz)

    # Build sorted array of keyboard intervals for fast lookup
    kb_intervals = []
    for seg in activity_segments:
        if seg["activity"] == "keyboard":
            kb_intervals.append((int(seg["start_time_ns"]), int(seg["end_time_ns"])))
    kb_intervals.sort()

    windows = []
    labels = []
    times_s = []
    activity_labels = []  # detailed label per window

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

        chunk = vals[idx_start:idx_end]
        win = resample_window(chunk, target_len)
        windows.append(win)

        # Label: is window centre inside any keyboard segment?
        is_keyboard = 0
        act_label = "non_keyboard"
        for seg in activity_segments:
            seg_start = int(seg["start_time_ns"])
            seg_end = int(seg["end_time_ns"])
            if seg_start <= centre <= seg_end:
                if seg["activity"] == "keyboard":
                    is_keyboard = 1
                    act_label = seg.get("label", "keyboard")
                else:
                    act_label = seg.get("label", seg["activity"])
                break

        labels.append(is_keyboard)
        activity_labels.append(act_label)
        times_s.append(centre / 1e9)

        centre += stride_ns

    if not windows:
        return _empty_activity_result()

    return {
        "windows": np.stack(windows),
        "labels": np.array(labels, dtype=np.int32),
        "activity_labels": np.array(activity_labels),
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


def _empty_activity_result():
    return {
        "windows": np.zeros((0, 1, N_CHANNELS), dtype=np.float32),
        "labels": np.array([], dtype=np.int32),
        "activity_labels": np.array([]),
        "times_s": np.array([], dtype=np.float64),
        "sessions": np.array([]),
        "sources": np.array([]),
    }


# ── Process directories ──────────────────────────────────────

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
        events = np.array([], dtype=np.int64)
        sess_id = os.path.basename(sess)

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


def process_mixed2_sessions(
    dirs: list[str],
    window_ms: int = ACTIVITY_WINDOW_MS,
    stride_ms: int = ACTIVITY_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> dict:
    """
    Process mixed2 sessions with activity_log.csv for activity segmentation.
    Returns activity-labelled windows.
    """
    sessions = discover_sessions(dirs, mode_filter="mixed2", dedup=False)
    if not sessions:
        # Try without mode filter for mixed2 dirs
        sessions = discover_sessions(dirs, mode_filter="", dedup=False)
    print(f"  Found {len(sessions)} mixed2 sessions in {dirs}")

    all_results = []
    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        activity_log_path = sess + "_activity_log.csv"

        if not os.path.exists(activity_log_path):
            print(f"    ⚠ No activity_log for {os.path.basename(sess)}, skipping")
            continue

        sensor = load_sensor_csv(sensor_path)
        activity_segments = load_activity_log(activity_log_path)
        sess_id = os.path.basename(sess)

        result = extract_activity_windows(
            sensor, activity_segments,
            window_ms=window_ms,
            stride_ms=stride_ms,
            target_rate_hz=target_rate_hz,
            session_id=sess_id,
            source_tag="mixed2",
        )
        if len(result["labels"]) > 0:
            all_results.append(result)
            n_pos = int(result["labels"].sum())
            n_neg = len(result["labels"]) - n_pos
            print(f"    {sess_id}: {n_pos} active + {n_neg} inactive = {len(result['labels'])} windows")

    return _merge_activity_results(all_results)


def _merge_results(results: list[dict]) -> dict:
    if not results:
        return _empty_result()
    return {
        key: np.concatenate([r[key] for r in results], axis=0)
        for key in ["windows", "labels", "times_s", "sessions", "sources"]
    }


def _merge_activity_results(results: list[dict]) -> dict:
    if not results:
        return _empty_activity_result()
    return {
        key: np.concatenate([r[key] for r in results], axis=0)
        for key in ["windows", "labels", "activity_labels", "times_s", "sessions", "sources"]
    }


# ── Main CLI ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build onset/activity detection dataset")

    parser.add_argument("--project-root", default="")
    parser.add_argument("--task", choices=["onset", "activity"], default="onset",
                        help="onset: build onset detection dataset. "
                             "activity: build activity segmentation dataset.")
    parser.add_argument("--keyboard-dirs", nargs="+",
                        default=["data/raw/single_key", "data/raw/boost"])
    parser.add_argument("--password-dirs", nargs="+",
                        default=["data/raw/password/len_8"])
    parser.add_argument("--negative-dirs", nargs="*", default=[])
    parser.add_argument("--mixed2-dirs", nargs="*", default=[],
                        help="Directories with mixed2 sessions for activity segmentation")
    parser.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--stride-ms", type=int, default=DEFAULT_STRIDE_MS)
    parser.add_argument("--label-radius-ms", type=int, default=DEFAULT_LABEL_RADIUS_MS)
    parser.add_argument("--target-rate", type=int, default=DEFAULT_TARGET_RATE_HZ)
    parser.add_argument("--output", default="data/processed/onset_dataset.npz")

    args = parser.parse_args()

    # Resolve paths
    if args.project_root:
        root = os.path.abspath(args.project_root)
        args.keyboard_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                              for d in args.keyboard_dirs]
        args.password_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                              for d in args.password_dirs]
        if args.negative_dirs:
            args.negative_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                                  for d in args.negative_dirs]
        if args.mixed2_dirs:
            args.mixed2_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                                for d in args.mixed2_dirs]
        if not os.path.isabs(args.output):
            args.output = os.path.join(root, args.output)

    if args.task == "activity":
        _build_activity_dataset(args)
    else:
        _build_onset_dataset(args)


def _build_onset_dataset(args):
    """Original onset detection dataset builder."""
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

    if args.keyboard_dirs:
        print("[1/3] Processing single_key sessions...")
        sk = process_keyboard_sessions(args.keyboard_dirs, "single_key", **common)
        if len(sk["labels"]) > 0:
            all_parts.append(sk)

    if args.password_dirs:
        print("\n[2/3] Processing password sessions...")
        pw = process_keyboard_sessions(args.password_dirs, "free_type", **common)
        if len(pw["labels"]) > 0:
            all_parts.append(pw)

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
    _save_onset_dataset(merged, args)


def _build_activity_dataset(args):
    """Build activity segmentation dataset from mixed2 sessions."""
    # Override window/stride for activity task
    win_ms = args.window_ms if args.window_ms != DEFAULT_WINDOW_MS else ACTIVITY_WINDOW_MS
    stride_ms = args.stride_ms if args.stride_ms != DEFAULT_STRIDE_MS else ACTIVITY_STRIDE_MS
    output = args.output
    if output == "data/processed/onset_dataset.npz":
        output = "data/processed/activity_dataset.npz"
        if args.project_root:
            output = os.path.join(os.path.abspath(args.project_root), output)

    print(f"\n{'='*60}")
    print(f"  ACTIVITY SEGMENTATION PREPROCESSOR")
    print(f"  window={win_ms}ms  stride={stride_ms}ms  rate={args.target_rate}Hz")
    print(f"{'='*60}\n")

    all_parts = []

    # 1. Mixed2 sessions (primary source)
    if args.mixed2_dirs:
        print("[1/3] Processing mixed2 sessions...")
        m2 = process_mixed2_sessions(
            args.mixed2_dirs,
            window_ms=win_ms,
            stride_ms=stride_ms,
            target_rate_hz=args.target_rate,
        )
        if len(m2["labels"]) > 0:
            all_parts.append(m2)

    # 2. Keyboard sessions as SUPPLEMENTARY positive examples
    #    These entire sessions are labelled keyboard-active.  They add
    #    intra-episode positive IMU diversity but do NOT provide real
    #    start/end boundary supervision (boundaries are synthetic).
    if args.keyboard_dirs or args.password_dirs:
        print("\n[2/3] Processing keyboard sessions as supplementary positives "
              "(no boundary supervision)...")
        for dirs, stype in [(args.keyboard_dirs, "single_key"),
                            (args.password_dirs, "free_type")]:
            if not dirs:
                continue
            mode_filter = "single_key" if stype == "single_key" else "free_type"
            sessions = discover_sessions(dirs, mode_filter=mode_filter)
            for sess in sessions:
                sensor_path = sess + "_sensor.csv"
                if not os.path.exists(sensor_path):
                    continue
                sensor = load_sensor_csv(sensor_path)
                ts_ns = sensor[:, 0]
                # Entire session is keyboard-active (synthetic boundary —
                # provides positive-class IMU samples, not real transition edges)
                fake_seg = [{
                    "start_time_ns": int(ts_ns[0]),
                    "end_time_ns": int(ts_ns[-1]),
                    "activity": "keyboard",
                    "label": "keyboard",
                }]
                result = extract_activity_windows(
                    sensor, fake_seg,
                    window_ms=win_ms,
                    stride_ms=stride_ms,
                    target_rate_hz=args.target_rate,
                    session_id=os.path.basename(sess),
                    source_tag=stype,
                )
                if len(result["labels"]) > 0:
                    all_parts.append(result)

    # 3. Negative sessions as negative examples
    if args.negative_dirs:
        print("\n[3/3] Processing negative sessions as inactive examples...")
        for d in args.negative_dirs:
            if not os.path.isdir(d):
                continue
            for root, _subdirs, files in os.walk(d):
                for f in sorted(files):
                    if f.startswith(".") or not f.endswith("_sensor.csv"):
                        continue
                    sensor_path = os.path.join(root, f)
                    sensor = load_sensor_csv(sensor_path)
                    ts_ns = sensor[:, 0]
                    # Empty segments → all negative
                    result = extract_activity_windows(
                        sensor, [],
                        window_ms=win_ms,
                        stride_ms=stride_ms,
                        target_rate_hz=args.target_rate,
                        session_id=f.replace("_sensor.csv", ""),
                        source_tag=f"negative_{os.path.basename(root)}",
                    )
                    if len(result["labels"]) > 0:
                        all_parts.append(result)

    if not all_parts:
        print("  ❌ No data collected!")
        sys.exit(1)

    merged = _merge_activity_results(all_parts)

    n_total = len(merged["labels"])
    n_pos = int(merged["labels"].sum())
    n_neg = n_total - n_pos
    win_len = window_samples(win_ms, args.target_rate)

    print(f"\n{'='*60}")
    print(f"  ACTIVITY DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows:     {n_total:,}")
    print(f"  Active (keyboard): {n_pos:,}  ({100*n_pos/max(n_total,1):.1f}%)")
    print(f"  Inactive:          {n_neg:,}  ({100*n_neg/max(n_total,1):.1f}%)")
    print(f"  Window shape:      ({win_len}, {N_CHANNELS})")

    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        output,
        windows=merged["windows"],
        labels=merged["labels"],
        activity_labels=merged["activity_labels"],
        times_s=merged["times_s"],
        sessions=merged["sessions"],
        sources=merged["sources"],
        window_ms=win_ms,
        stride_ms=stride_ms,
        label_radius_ms=0,  # not applicable for activity task
        target_rate_hz=args.target_rate,
        n_channels=N_CHANNELS,
        task="activity",
    )
    print(f"\n  ✓ Saved → {output}")
    print(f"{'='*60}\n")


def _save_onset_dataset(merged, args):
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
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        label_radius_ms=args.label_radius_ms,
        target_rate_hz=args.target_rate,
        n_channels=N_CHANNELS,
        task="onset",
    )
    print(f"\n  ✓ Saved → {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
