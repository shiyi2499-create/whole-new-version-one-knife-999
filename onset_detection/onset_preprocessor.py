"""
Onset / Password-Boundary Preprocessor
======================================

This file keeps the original onset dataset builder and adds a complete
`task=password_boundary` chain focused on extracting password episodes
from mixed2 streams.

Priority of the new task:
  1. mixed2 provides real password boundary supervision
  2. free typing / single_key / boost become hard non-password background
  3. dedicated password sessions become supplementary password_active data

Output tasks:
  - onset              -> binary keystroke onset windows
  - activity           -> legacy binary keyboard-active windows
  - password_boundary  -> 4-class windows:
                          non_password / password_start /
                          password_active / password_end
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import numpy as np

try:
    from scipy.signal import resample as scipy_resample
except ImportError as exc:
    raise ImportError("scipy is required: pip install scipy") from exc

from onset_utils import (
    PASSWORD_BOUNDARY_LABELS,
    boundary_label_id,
)


# ── Configuration ─────────────────────────────────────────────

DEFAULT_TARGET_RATE_HZ = 190
DEFAULT_WINDOW_MS = 150
DEFAULT_STRIDE_MS = 25
DEFAULT_LABEL_RADIUS_MS = 30
N_CHANNELS = 6

ACTIVITY_WINDOW_MS = 400
ACTIVITY_STRIDE_MS = 50

PASSWORD_BOUNDARY_WINDOW_MS = 500
PASSWORD_BOUNDARY_STRIDE_MS = 40
PASSWORD_BOUNDARY_RADIUS_MS = 120
PASSWORD_BOUNDARY_PRE_KEY_MS = 120
PASSWORD_BOUNDARY_POST_KEY_MS = 220
PASSWORD_BOUNDARY_TRANSITION_EXCLUSION_MS = 240


def window_samples(window_ms: int, rate_hz: int) -> int:
    return max(1, int(window_ms / 1000.0 * rate_hz))


# ── Session loading ──────────────────────────────────────────

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



def load_events_csv(path: str, press_only: bool = True) -> np.ndarray:
    timestamps = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            if press_only and row["event_type"] != "press":
                continue
            timestamps.append(int(row["timestamp_ns"]))
    return np.asarray(timestamps, dtype=np.int64)



def load_activity_log(path: str) -> list[dict]:
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
        for root, _subdirs, files in os.walk(d):
            for f in sorted(files):
                if f.startswith(".") or f.startswith("._"):
                    continue
                if not f.endswith("_sensor.csv"):
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


# ── Common sliding-window helpers ────────────────────────────

def resample_window(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) < 2:
        return np.zeros((target_len, values.shape[1]), dtype=np.float32)
    out = scipy_resample(values, target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)



def _empty_result() -> dict:
    return {
        "windows": np.zeros((0, 1, N_CHANNELS), dtype=np.float32),
        "labels": np.array([], dtype=np.int32),
        "times_s": np.array([], dtype=np.float64),
        "sessions": np.array([], dtype=str),
        "sources": np.array([], dtype=str),
    }



def _empty_activity_result() -> dict:
    out = _empty_result()
    out["activity_labels"] = np.array([], dtype=str)
    return out



def _empty_password_boundary_result() -> dict:
    out = _empty_result()
    out["boundary_labels"] = np.array([], dtype=str)
    return out



def _iterate_window_chunks(
    sensor: np.ndarray,
    window_ms: int,
    stride_ms: int,
    target_rate_hz: int,
):
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]
    if len(ts_ns) < 10:
        return

    t_start_ns = int(ts_ns[0])
    t_end_ns = int(ts_ns[-1])
    win_ns = int(window_ms * 1_000_000)
    stride_ns = int(stride_ms * 1_000_000)
    target_len = window_samples(window_ms, target_rate_hz)
    half_win_ns = win_ns // 2
    centre = t_start_ns + half_win_ns

    while centre + half_win_ns <= t_end_ns:
        w_start = centre - half_win_ns
        w_end = centre + half_win_ns
        idx_start = np.searchsorted(ts_ns, w_start, side="left")
        idx_end = np.searchsorted(ts_ns, w_end, side="right")
        if idx_end - idx_start >= 4:
            chunk = vals[idx_start:idx_end]
            yield centre, resample_window(chunk, target_len)
        centre += stride_ns


# ── Onset windows (original) ─────────────────────────────────

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
    if len(sensor) < 10:
        return _empty_result()

    radius_ns = int(label_radius_ms * 1_000_000)
    events_sorted = np.sort(event_times_ns) if len(event_times_ns) else np.array([], dtype=np.int64)

    windows, labels, times_s = [], [], []
    for centre, win in _iterate_window_chunks(sensor, window_ms, stride_ms, target_rate_hz):
        windows.append(win)
        if len(events_sorted):
            insert_pos = np.searchsorted(events_sorted, centre)
            near = []
            for offset in (insert_pos - 1, insert_pos):
                if 0 <= offset < len(events_sorted):
                    near.append(abs(int(events_sorted[offset]) - int(centre)))
            min_dist = min(near) if near else radius_ns + 1
            label = 1 if min_dist <= radius_ns else 0
        else:
            label = 0
        labels.append(label)
        times_s.append(centre / 1e9)

    if not windows:
        return _empty_result()
    return {
        "windows": np.stack(windows),
        "labels": np.asarray(labels, dtype=np.int32),
        "times_s": np.asarray(times_s, dtype=np.float64),
        "sessions": np.asarray([session_id] * len(windows)),
        "sources": np.asarray([source_tag] * len(windows)),
    }


# ── Activity windows (legacy) ────────────────────────────────

def extract_activity_windows(
    sensor: np.ndarray,
    activity_segments: list[dict],
    window_ms: int = ACTIVITY_WINDOW_MS,
    stride_ms: int = ACTIVITY_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
    session_id: str = "",
    source_tag: str = "",
) -> dict:
    if len(sensor) < 10:
        return _empty_activity_result()

    windows, labels, times_s, activity_labels = [], [], [], []
    for centre, win in _iterate_window_chunks(sensor, window_ms, stride_ms, target_rate_hz):
        windows.append(win)
        label = 0
        act_label = "non_keyboard"
        for seg in activity_segments:
            if int(seg["start_time_ns"]) <= centre <= int(seg["end_time_ns"]):
                if seg.get("activity") == "keyboard":
                    label = 1
                    act_label = seg.get("label", "keyboard")
                else:
                    act_label = seg.get("label", seg.get("activity", "non_keyboard"))
                break
        labels.append(label)
        activity_labels.append(act_label)
        times_s.append(centre / 1e9)

    if not windows:
        return _empty_activity_result()
    return {
        "windows": np.stack(windows),
        "labels": np.asarray(labels, dtype=np.int32),
        "activity_labels": np.asarray(activity_labels),
        "times_s": np.asarray(times_s, dtype=np.float64),
        "sessions": np.asarray([session_id] * len(windows)),
        "sources": np.asarray([source_tag] * len(windows)),
    }


# ── Password-boundary windows (new main task) ────────────────

def get_password_segments_from_activity_log(activity_segments: list[dict]) -> list[dict]:
    return [
        seg for seg in activity_segments
        if seg.get("label", "") == "typing_2" or seg.get("typing_style", "") == "password"
    ]



def refine_password_segments_with_events(
    activity_segments: list[dict],
    event_times_ns: np.ndarray,
    pre_key_ms: int = PASSWORD_BOUNDARY_PRE_KEY_MS,
    post_key_ms: int = PASSWORD_BOUNDARY_POST_KEY_MS,
) -> list[dict]:
    """
    Convert coarse protocol password blocks into stricter password activity episodes.

    Rule:
      - start is anchored near the first password keystroke, not the protocol block start
      - end is anchored near the last password keystroke, not the protocol block end
      - the refined episode keeps a small pre/post motion margin around those keystrokes

    Windows between the protocol block boundary and the refined boundary are *not*
    treated as positive by default.
    """
    password_segments = get_password_segments_from_activity_log(activity_segments)
    if len(event_times_ns) == 0:
        return []

    event_times_ns = np.asarray(event_times_ns, dtype=np.int64)
    pre_ns = int(pre_key_ms * 1_000_000)
    post_ns = int(post_key_ms * 1_000_000)
    refined = []
    for seg in password_segments:
        protocol_start_ns = int(seg["start_time_ns"])
        protocol_end_ns = int(seg["end_time_ns"])
        in_seg = event_times_ns[(event_times_ns >= protocol_start_ns) & (event_times_ns <= protocol_end_ns)]
        if len(in_seg) == 0:
            continue
        first_key_ns = int(in_seg[0])
        last_key_ns = int(in_seg[-1])
        refined_start_ns = max(protocol_start_ns, first_key_ns - pre_ns)
        refined_end_ns = min(protocol_end_ns, last_key_ns + post_ns)
        if refined_end_ns < refined_start_ns:
            refined_start_ns = first_key_ns
            refined_end_ns = last_key_ns
        out = dict(seg)
        out.update({
            "protocol_start_time_ns": protocol_start_ns,
            "protocol_end_time_ns": protocol_end_ns,
            "start_time_ns": int(refined_start_ns),
            "end_time_ns": int(refined_end_ns),
            "first_key_time_ns": first_key_ns,
            "last_key_time_ns": last_key_ns,
            "n_keys": int(len(in_seg)),
        })
        refined.append(out)
    return refined



def extract_password_boundary_windows(
    sensor: np.ndarray,
    activity_segments: list[dict],
    event_times_ns: Optional[np.ndarray] = None,
    window_ms: int = PASSWORD_BOUNDARY_WINDOW_MS,
    stride_ms: int = PASSWORD_BOUNDARY_STRIDE_MS,
    boundary_radius_ms: int = PASSWORD_BOUNDARY_RADIUS_MS,
    transition_exclusion_ms: int = PASSWORD_BOUNDARY_TRANSITION_EXCLUSION_MS,
    pre_key_ms: int = PASSWORD_BOUNDARY_PRE_KEY_MS,
    post_key_ms: int = PASSWORD_BOUNDARY_POST_KEY_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
    session_id: str = "",
    source_tag: str = "",
    synthetic_password_session: bool = False,
) -> dict:
    """
    Build 4-class password-boundary windows.

    For real mixed2 password segments, labels are anchored to a *refined* password
    activity episode instead of the coarse protocol typing_2 block:
      - `password_start` is centred near the first password keystroke (+ small pre-roll)
      - `password_end`   is centred near the last password keystroke (+ small post-roll)
      - `password_active` spans the interior of that refined episode
      - windows in the transition shell around the refined start/end can be dropped
        to avoid forcing ambiguous protocol-boundary labels

    For `synthetic_password_session=True`, the whole session is treated as
    supplementary password_active only, with NO synthetic start/end labels.
    """
    if len(sensor) < 10:
        return _empty_password_boundary_result()

    radius_ns = int(boundary_radius_ms * 1_000_000)
    exclusion_ns = max(radius_ns, int(transition_exclusion_ms * 1_000_000))
    event_times_ns = np.asarray(event_times_ns if event_times_ns is not None else [], dtype=np.int64)
    refined_password_segments = refine_password_segments_with_events(
        activity_segments,
        event_times_ns,
        pre_key_ms=pre_key_ms,
        post_key_ms=post_key_ms,
    ) if not synthetic_password_session else []

    windows, labels, times_s, boundary_labels = [], [], [], []

    for centre, win in _iterate_window_chunks(sensor, window_ms, stride_ms, target_rate_hz):
        if synthetic_password_session:
            label_name = "password_active"
            ambiguous = False
        else:
            label_name = "non_password"
            ambiguous = False
            for seg in refined_password_segments:
                start_ns = int(seg["start_time_ns"])
                end_ns = int(seg["end_time_ns"])
                if abs(centre - start_ns) <= radius_ns:
                    label_name = "password_start"
                    break
                if abs(centre - end_ns) <= radius_ns:
                    label_name = "password_end"
                    break
                if (start_ns + exclusion_ns) < centre < (end_ns - exclusion_ns):
                    label_name = "password_active"
                    break
                if (start_ns - exclusion_ns) < centre < (start_ns + exclusion_ns):
                    ambiguous = True
                    break
                if (end_ns - exclusion_ns) < centre < (end_ns + exclusion_ns):
                    ambiguous = True
                    break

        if ambiguous:
            continue
        windows.append(win)
        labels.append(boundary_label_id(label_name))
        boundary_labels.append(label_name)
        times_s.append(centre / 1e9)

    if not windows:
        return _empty_password_boundary_result()
    return {
        "windows": np.stack(windows),
        "labels": np.asarray(labels, dtype=np.int32),
        "boundary_labels": np.asarray(boundary_labels),
        "times_s": np.asarray(times_s, dtype=np.float64),
        "sessions": np.asarray([session_id] * len(windows)),
        "sources": np.asarray([source_tag] * len(windows)),
    }



def extract_constant_label_windows(
    sensor: np.ndarray,
    label_name: str,
    window_ms: int,
    stride_ms: int,
    target_rate_hz: int,
    session_id: str,
    source_tag: str,
) -> dict:
    if len(sensor) < 10:
        return _empty_password_boundary_result()
    windows, labels, times_s = [], [], []
    for centre, win in _iterate_window_chunks(sensor, window_ms, stride_ms, target_rate_hz):
        windows.append(win)
        labels.append(boundary_label_id(label_name))
        times_s.append(centre / 1e9)
    if not windows:
        return _empty_password_boundary_result()
    return {
        "windows": np.stack(windows),
        "labels": np.asarray(labels, dtype=np.int32),
        "boundary_labels": np.asarray([label_name] * len(windows)),
        "times_s": np.asarray(times_s, dtype=np.float64),
        "sessions": np.asarray([session_id] * len(windows)),
        "sources": np.asarray([source_tag] * len(windows)),
    }


# ── Directory processors ─────────────────────────────────────

def _merge_results(results: list[dict]) -> dict:
    if not results:
        return _empty_result()
    return {k: np.concatenate([r[k] for r in results], axis=0) for k in ["windows", "labels", "times_s", "sessions", "sources"]}



def _merge_activity_results(results: list[dict]) -> dict:
    if not results:
        return _empty_activity_result()
    return {
        k: np.concatenate([r[k] for r in results], axis=0)
        for k in ["windows", "labels", "activity_labels", "times_s", "sessions", "sources"]
    }



def _merge_password_boundary_results(results: list[dict]) -> dict:
    if not results:
        return _empty_password_boundary_result()
    return {
        k: np.concatenate([r[k] for r in results], axis=0)
        for k in ["windows", "labels", "boundary_labels", "times_s", "sessions", "sources"]
    }



def process_keyboard_sessions(dirs: list[str], session_type: str, **kwargs) -> dict:
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
        result = extract_sliding_windows(sensor, events, session_id=sess_id, source_tag=session_type, **kwargs)
        if len(result["labels"]):
            all_results.append(result)
            n_pos = int(result["labels"].sum())
            n_neg = len(result["labels"]) - n_pos
            print(f"    {sess_id}: {n_pos} pos + {n_neg} neg = {len(result['labels'])} windows")
    return _merge_results(all_results)



def process_negative_sessions(dirs: list[str], **kwargs) -> dict:
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
        sensor = load_sensor_csv(sess + "_sensor.csv")
        sess_id = os.path.basename(sess)
        source_tag = f"negative_{os.path.basename(os.path.dirname(sess))}"
        result = extract_sliding_windows(sensor, np.array([], dtype=np.int64), session_id=sess_id, source_tag=source_tag, **kwargs)
        if len(result["labels"]):
            all_results.append(result)
            print(f"    {sess_id}: {len(result['labels'])} neg windows ({source_tag})")
    return _merge_results(all_results)



def process_mixed2_sessions(
    dirs: list[str],
    window_ms: int = ACTIVITY_WINDOW_MS,
    stride_ms: int = ACTIVITY_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> dict:
    sessions = discover_sessions(dirs, mode_filter="mixed2", dedup=False) or discover_sessions(dirs, mode_filter="", dedup=False)
    print(f"  Found {len(sessions)} mixed2 sessions in {dirs}")
    all_results = []
    for sess in sessions:
        activity_log_path = sess + "_activity_log.csv"
        if not os.path.exists(activity_log_path):
            print(f"    ⚠ No activity_log for {os.path.basename(sess)}, skipping")
            continue
        sensor = load_sensor_csv(sess + "_sensor.csv")
        activity_segments = load_activity_log(activity_log_path)
        sess_id = os.path.basename(sess)
        result = extract_activity_windows(sensor, activity_segments, window_ms=window_ms, stride_ms=stride_ms, target_rate_hz=target_rate_hz, session_id=sess_id, source_tag="mixed2")
        if len(result["labels"]):
            all_results.append(result)
            n_pos = int(result["labels"].sum())
            n_neg = len(result["labels"]) - n_pos
            print(f"    {sess_id}: {n_pos} active + {n_neg} inactive = {len(result['labels'])} windows")
    return _merge_activity_results(all_results)



def process_mixed2_password_boundary_sessions(
    dirs: list[str],
    window_ms: int = PASSWORD_BOUNDARY_WINDOW_MS,
    stride_ms: int = PASSWORD_BOUNDARY_STRIDE_MS,
    boundary_radius_ms: int = PASSWORD_BOUNDARY_RADIUS_MS,
    transition_exclusion_ms: int = PASSWORD_BOUNDARY_TRANSITION_EXCLUSION_MS,
    pre_key_ms: int = PASSWORD_BOUNDARY_PRE_KEY_MS,
    post_key_ms: int = PASSWORD_BOUNDARY_POST_KEY_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> dict:
    sessions = discover_sessions(dirs, mode_filter="mixed2", dedup=False) or discover_sessions(dirs, mode_filter="", dedup=False)
    print(f"  Found {len(sessions)} mixed2 sessions in {dirs}")
    all_results = []
    for sess in sessions:
        activity_log_path = sess + "_activity_log.csv"
        events_path = sess + "_events.csv"
        if not os.path.exists(activity_log_path):
            print(f"    ⚠ No activity_log for {os.path.basename(sess)}, skipping")
            continue
        if not os.path.exists(events_path):
            print(f"    ⚠ No events.csv for {os.path.basename(sess)}, skipping")
            continue
        sensor = load_sensor_csv(sess + "_sensor.csv")
        activity_segments = load_activity_log(activity_log_path)
        event_times_ns = load_events_csv(events_path, press_only=True)
        sess_id = os.path.basename(sess)
        result = extract_password_boundary_windows(
            sensor,
            activity_segments,
            event_times_ns=event_times_ns,
            window_ms=window_ms,
            stride_ms=stride_ms,
            boundary_radius_ms=boundary_radius_ms,
            transition_exclusion_ms=transition_exclusion_ms,
            pre_key_ms=pre_key_ms,
            post_key_ms=post_key_ms,
            target_rate_hz=target_rate_hz,
            session_id=sess_id,
            source_tag="mixed2_password_boundary",
        )
        if len(result["labels"]):
            all_results.append(result)
            counts = np.bincount(result["labels"], minlength=len(PASSWORD_BOUNDARY_LABELS))
            print(
                f"    {sess_id}: non={counts[0]} start={counts[1]} "
                f"active={counts[2]} end={counts[3]}"
            )
    return _merge_password_boundary_results(all_results)



def process_constant_label_sessions(
    dirs: list[str],
    label_name: str,
    mode_filter: str,
    source_tag: str,
    dedup: bool = True,
    window_ms: int = PASSWORD_BOUNDARY_WINDOW_MS,
    stride_ms: int = PASSWORD_BOUNDARY_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> dict:
    sessions = discover_sessions(dirs, mode_filter=mode_filter, dedup=dedup)
    print(f"  Found {len(sessions)} {source_tag} sessions in {dirs}")
    all_results = []
    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        if not os.path.exists(sensor_path):
            continue
        sensor = load_sensor_csv(sensor_path)
        sess_id = os.path.basename(sess)
        result = extract_constant_label_windows(
            sensor,
            label_name=label_name,
            window_ms=window_ms,
            stride_ms=stride_ms,
            target_rate_hz=target_rate_hz,
            session_id=sess_id,
            source_tag=source_tag,
        )
        if len(result["labels"]):
            all_results.append(result)
            print(f"    {sess_id}: {len(result['labels'])} windows -> {label_name}")
    return _merge_password_boundary_results(all_results)



def process_password_active_sessions(
    dirs: list[str],
    window_ms: int = PASSWORD_BOUNDARY_WINDOW_MS,
    stride_ms: int = PASSWORD_BOUNDARY_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> dict:
    sessions = discover_sessions(dirs, mode_filter="free_type")
    print(f"  Found {len(sessions)} password sessions in {dirs}")
    all_results = []
    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        if not os.path.exists(sensor_path):
            continue
        sensor = load_sensor_csv(sensor_path)
        sess_id = os.path.basename(sess)
        result = extract_password_boundary_windows(
            sensor,
            activity_segments=[],
            window_ms=window_ms,
            stride_ms=stride_ms,
            boundary_radius_ms=0,
            target_rate_hz=target_rate_hz,
            session_id=sess_id,
            source_tag="password_active_supplement",
            synthetic_password_session=True,
        )
        if len(result["labels"]):
            all_results.append(result)
            print(f"    {sess_id}: {len(result['labels'])} active-only password windows")
    return _merge_password_boundary_results(all_results)


# ── Builders ─────────────────────────────────────────────────

def _save_npz(output: str, **arrays):
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(output, **arrays)



def _build_onset_dataset(args):
    common = dict(
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        label_radius_ms=args.label_radius_ms,
        target_rate_hz=args.target_rate,
    )
    print(f"\n{'='*60}")
    print("  ONSET PREPROCESSOR")
    print(f"  window={args.window_ms}ms  stride={args.stride_ms}ms  label_radius=±{args.label_radius_ms}ms  rate={args.target_rate}Hz")
    print(f"{'='*60}\n")

    all_parts = []
    if args.keyboard_dirs:
        print("[1/3] Processing single_key sessions...")
        sk = process_keyboard_sessions(args.keyboard_dirs, "single_key", **common)
        if len(sk["labels"]):
            all_parts.append(sk)
    if args.password_dirs:
        print("\n[2/3] Processing password sessions...")
        pw = process_keyboard_sessions(args.password_dirs, "free_type", **common)
        if len(pw["labels"]):
            all_parts.append(pw)
    if args.negative_dirs:
        print("\n[3/3] Processing negative sessions...")
        neg = process_negative_sessions(args.negative_dirs, **common)
        if len(neg["labels"]):
            all_parts.append(neg)
    else:
        print("\n[3/3] No negative dirs specified (skipping)")

    if not all_parts:
        print("  ❌ No onset data collected")
        sys.exit(1)

    merged = _merge_results(all_parts)
    n_total = len(merged["labels"])
    n_pos = int(merged["labels"].sum())
    n_neg = n_total - n_pos
    print(f"\n{'='*60}")
    print("  ONSET DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows: {n_total:,}")
    print(f"  Positive:      {n_pos:,}")
    print(f"  Negative:      {n_neg:,}")

    _save_npz(
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
        label_names=np.array(["non_onset", "onset"]),
    )
    print(f"\n  ✓ Saved → {args.output}\n")



def _build_activity_dataset(args):
    win_ms = args.window_ms if args.window_ms != DEFAULT_WINDOW_MS else ACTIVITY_WINDOW_MS
    stride_ms = args.stride_ms if args.stride_ms != DEFAULT_STRIDE_MS else ACTIVITY_STRIDE_MS
    output = args.output if args.output != "data/processed/onset_dataset.npz" else "data/processed/activity_dataset.npz"
    if args.project_root and not os.path.isabs(output):
        output = os.path.join(os.path.abspath(args.project_root), output)

    print(f"\n{'='*60}")
    print("  ACTIVITY PREPROCESSOR (legacy)")
    print(f"  window={win_ms}ms  stride={stride_ms}ms  rate={args.target_rate}Hz")
    print(f"{'='*60}\n")

    all_parts = []
    if args.mixed2_dirs:
        print("[1/3] Processing mixed2 sessions...")
        m2 = process_mixed2_sessions(args.mixed2_dirs, window_ms=win_ms, stride_ms=stride_ms, target_rate_hz=args.target_rate)
        if len(m2["labels"]):
            all_parts.append(m2)
    if args.keyboard_dirs or args.password_dirs:
        print("\n[2/3] Processing keyboard sessions as positive supplements...")
        for dirs, stype in ((args.keyboard_dirs, "single_key"), (args.password_dirs, "free_type")):
            if not dirs:
                continue
            sessions = discover_sessions(dirs, mode_filter=("single_key" if stype == "single_key" else "free_type"))
            for sess in sessions:
                sensor = load_sensor_csv(sess + "_sensor.csv")
                ts_ns = sensor[:, 0]
                fake_seg = [{
                    "start_time_ns": int(ts_ns[0]),
                    "end_time_ns": int(ts_ns[-1]),
                    "activity": "keyboard",
                    "label": "keyboard",
                }]
                result = extract_activity_windows(sensor, fake_seg, window_ms=win_ms, stride_ms=stride_ms, target_rate_hz=args.target_rate, session_id=os.path.basename(sess), source_tag=stype)
                if len(result["labels"]):
                    all_parts.append(result)
    if args.negative_dirs:
        print("\n[3/3] Processing negative sessions...")
        for d in args.negative_dirs:
            if not os.path.isdir(d):
                continue
            for root, _subdirs, files in os.walk(d):
                for f in sorted(files):
                    if f.startswith(".") or not f.endswith("_sensor.csv"):
                        continue
                    sensor = load_sensor_csv(os.path.join(root, f))
                    result = extract_activity_windows(sensor, [], window_ms=win_ms, stride_ms=stride_ms, target_rate_hz=args.target_rate, session_id=f.replace("_sensor.csv", ""), source_tag=f"negative_{os.path.basename(root)}")
                    if len(result["labels"]):
                        all_parts.append(result)

    if not all_parts:
        print("  ❌ No activity data collected")
        sys.exit(1)

    merged = _merge_activity_results(all_parts)
    _save_npz(
        output,
        windows=merged["windows"],
        labels=merged["labels"],
        activity_labels=merged["activity_labels"],
        times_s=merged["times_s"],
        sessions=merged["sessions"],
        sources=merged["sources"],
        window_ms=win_ms,
        stride_ms=stride_ms,
        label_radius_ms=0,
        target_rate_hz=args.target_rate,
        n_channels=N_CHANNELS,
        task="activity",
        label_names=np.array(["non_keyboard", "keyboard"]),
    )
    print(f"\n  ✓ Saved → {output}\n")



def _build_password_boundary_dataset(args):
    win_ms = args.window_ms if args.window_ms != DEFAULT_WINDOW_MS else PASSWORD_BOUNDARY_WINDOW_MS
    stride_ms = args.stride_ms if args.stride_ms != DEFAULT_STRIDE_MS else PASSWORD_BOUNDARY_STRIDE_MS
    boundary_radius_ms = args.label_radius_ms if args.label_radius_ms != DEFAULT_LABEL_RADIUS_MS else PASSWORD_BOUNDARY_RADIUS_MS
    output = args.output if args.output != "data/processed/onset_dataset.npz" else "data/processed/password_boundary_dataset.npz"
    if args.project_root and not os.path.isabs(output):
        output = os.path.join(os.path.abspath(args.project_root), output)

    print(f"\n{'='*60}")
    print("  PASSWORD BOUNDARY PREPROCESSOR")
    print(f"  window={win_ms}ms  stride={stride_ms}ms  boundary_radius=±{boundary_radius_ms}ms  rate={args.target_rate}Hz")
    print(f"{'='*60}\n")

    all_parts = []

    if args.mixed2_dirs:
        print("[1/4] Processing mixed2 sessions for real password boundary supervision...")
        m2 = process_mixed2_password_boundary_sessions(
            args.mixed2_dirs,
            window_ms=win_ms,
            stride_ms=stride_ms,
            boundary_radius_ms=boundary_radius_ms,
            transition_exclusion_ms=PASSWORD_BOUNDARY_TRANSITION_EXCLUSION_MS,
            pre_key_ms=PASSWORD_BOUNDARY_PRE_KEY_MS,
            post_key_ms=PASSWORD_BOUNDARY_POST_KEY_MS,
            target_rate_hz=args.target_rate,
        )
        if len(m2["labels"]):
            all_parts.append(m2)

    if args.password_dirs:
        print("\n[2/4] Processing password sessions as supplementary password_active windows...")
        pw = process_password_active_sessions(
            args.password_dirs,
            window_ms=win_ms,
            stride_ms=stride_ms,
            target_rate_hz=args.target_rate,
        )
        if len(pw["labels"]):
            all_parts.append(pw)

    if args.keyboard_dirs:
        print("\n[3/4] Processing keyboard sessions as hard non-password background...")
        bg1 = process_constant_label_sessions(
            args.keyboard_dirs,
            label_name="non_password",
            mode_filter="single_key",
            source_tag="single_key_non_password",
            window_ms=win_ms,
            stride_ms=stride_ms,
            target_rate_hz=args.target_rate,
        )
        if len(bg1["labels"]):
            all_parts.append(bg1)
        bg2 = process_constant_label_sessions(
            args.keyboard_dirs,
            label_name="non_password",
            mode_filter="boost",
            source_tag="boost_non_password",
            window_ms=win_ms,
            stride_ms=stride_ms,
            target_rate_hz=args.target_rate,
        )
        if len(bg2["labels"]):
            all_parts.append(bg2)

    if args.negative_dirs:
        print("\n[4/4] Processing negative sessions as non-password background...")
        bg3 = process_constant_label_sessions(
            args.negative_dirs,
            label_name="non_password",
            mode_filter="",
            source_tag="negative_non_password",
            dedup=False,
            window_ms=win_ms,
            stride_ms=stride_ms,
            target_rate_hz=args.target_rate,
        )
        if len(bg3["labels"]):
            all_parts.append(bg3)
    else:
        print("\n[4/4] No negative dirs specified (skipping)")

    if not all_parts:
        print("  ❌ No password_boundary data collected")
        sys.exit(1)

    merged = _merge_password_boundary_results(all_parts)
    counts = np.bincount(merged["labels"], minlength=len(PASSWORD_BOUNDARY_LABELS))
    print(f"\n{'='*60}")
    print("  PASSWORD BOUNDARY DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows:    {len(merged['labels']):,}")
    for idx, name in enumerate(PASSWORD_BOUNDARY_LABELS):
        frac = 100.0 * counts[idx] / max(len(merged["labels"]), 1)
        print(f"  {name:16s} {counts[idx]:8d}  ({frac:5.1f}%)")

    _save_npz(
        output,
        windows=merged["windows"],
        labels=merged["labels"],
        boundary_labels=merged["boundary_labels"],
        times_s=merged["times_s"],
        sessions=merged["sessions"],
        sources=merged["sources"],
        window_ms=win_ms,
        stride_ms=stride_ms,
        label_radius_ms=boundary_radius_ms,
        target_rate_hz=args.target_rate,
        n_channels=N_CHANNELS,
        task="password_boundary",
        label_names=np.asarray(PASSWORD_BOUNDARY_LABELS),
        pre_key_ms=PASSWORD_BOUNDARY_PRE_KEY_MS,
        post_key_ms=PASSWORD_BOUNDARY_POST_KEY_MS,
        transition_exclusion_ms=PASSWORD_BOUNDARY_TRANSITION_EXCLUSION_MS,
    )
    print(f"\n  ✓ Saved → {output}\n")


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build onset / password-boundary datasets")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--task", choices=["onset", "activity", "password_boundary"], default="onset")
    parser.add_argument("--keyboard-dirs", nargs="+", default=["data/raw/single_key", "data/raw/boost"])
    parser.add_argument("--password-dirs", nargs="+", default=["data/raw/password/len_8"])
    parser.add_argument("--negative-dirs", nargs="*", default=[])
    parser.add_argument("--mixed2-dirs", nargs="*", default=[])
    parser.add_argument("--window-ms", type=int, default=DEFAULT_WINDOW_MS)
    parser.add_argument("--stride-ms", type=int, default=DEFAULT_STRIDE_MS)
    parser.add_argument("--label-radius-ms", type=int, default=DEFAULT_LABEL_RADIUS_MS,
                        help="On onset task: onset label radius. On password_boundary: boundary radius.")
    parser.add_argument("--target-rate", type=int, default=DEFAULT_TARGET_RATE_HZ)
    parser.add_argument("--output", default="data/processed/onset_dataset.npz")
    args = parser.parse_args()

    if args.project_root:
        root = os.path.abspath(args.project_root)
        args.keyboard_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in args.keyboard_dirs]
        args.password_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in args.password_dirs]
        args.negative_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in args.negative_dirs]
        args.mixed2_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in args.mixed2_dirs]
        if not os.path.isabs(args.output):
            args.output = os.path.join(root, args.output)

    if args.task == "activity":
        _build_activity_dataset(args)
    elif args.task == "password_boundary":
        _build_password_boundary_dataset(args)
    else:
        _build_onset_dataset(args)


if __name__ == "__main__":
    main()
