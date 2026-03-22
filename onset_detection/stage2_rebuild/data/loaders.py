"""
Data loaders adapted to the current main workspace layout.

This module supports two on-disk formats:
- prefix-based raw files used in the main repo, for example:
  ``.../p01_free_type_password_part1_20260315_200637_sensor.csv``
- directory-style sessions with ``sensor.csv`` / ``events.csv`` / ...

It also provides a mixed2 helper that derives GT password-group structure from
the existing ``activity_log.csv`` + ``events.csv`` files, without requiring a
separate ``attempts.csv``.
"""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SENSOR_SUFFIX = "_sensor.csv"
EVENTS_SUFFIX = "_events.csv"
ATTEMPTS_SUFFIX = "_attempts.csv"
PROMPTS_SUFFIX = "_prompts.csv"
META_SUFFIX = "_meta.txt"
ACTIVITY_LOG_SUFFIX = "_activity_log.csv"
PROTOCOL_SUFFIX = "_protocol.json"

SUPPORTED_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789")
IGNORED_KEYS = {
    "shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
    "left", "right", "up", "down", "delete", "space", "backspace",
}


def supported_key(key: str) -> bool:
    key = (key or "").lower().strip()
    return len(key) == 1 and key in SUPPORTED_CHARS


def _parse_prompts_field(value) -> List[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            continue
    return []


def _resample_imu_segment(
    timestamps_ns: np.ndarray,
    data: np.ndarray,
    target_rate_hz: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if target_rate_hz is None or len(timestamps_ns) < 2:
        return timestamps_ns.astype(np.int64), data.astype(np.float32)

    ts = np.asarray(timestamps_ns, dtype=np.int64)
    vals = np.asarray(data, dtype=np.float32)

    uniq_ts, uniq_idx = np.unique(ts, return_index=True)
    ts = uniq_ts
    vals = vals[uniq_idx]
    if len(ts) < 2 or ts[-1] <= ts[0]:
        return ts.astype(np.int64), vals.astype(np.float32)

    step_ns = max(int(round(1e9 / float(target_rate_hz))), 1)
    new_ts = np.arange(int(ts[0]), int(ts[-1]) + 1, step_ns, dtype=np.int64)
    if len(new_ts) < 2:
        return ts.astype(np.int64), vals.astype(np.float32)

    base_t = ts.astype(np.float64)
    out = np.zeros((len(new_ts), vals.shape[1]), dtype=np.float32)
    for ch in range(vals.shape[1]):
        out[:, ch] = np.interp(new_ts.astype(np.float64), base_t, vals[:, ch]).astype(np.float32)
    return new_ts, out


@dataclass
class SessionPaths:
    sensor: Path
    events: Optional[Path] = None
    attempts: Optional[Path] = None
    prompts: Optional[Path] = None
    meta: Optional[Path] = None
    activity_log: Optional[Path] = None
    protocol: Optional[Path] = None
    session_ref: str = ""


def resolve_session_paths(session_ref: str | Path) -> SessionPaths:
    path = Path(session_ref)

    if path.is_dir() and (path / "sensor.csv").exists():
        return SessionPaths(
            sensor=path / "sensor.csv",
            events=(path / "events.csv") if (path / "events.csv").exists() else None,
            attempts=(path / "attempts.csv") if (path / "attempts.csv").exists() else None,
            prompts=(path / "prompts.csv") if (path / "prompts.csv").exists() else None,
            meta=(path / "meta.txt") if (path / "meta.txt").exists() else None,
            activity_log=(path / "activity_log.csv") if (path / "activity_log.csv").exists() else None,
            protocol=(path / "protocol.json") if (path / "protocol.json").exists() else None,
            session_ref=str(path),
        )

    if path.is_file() and path.name.endswith(SENSOR_SUFFIX):
        prefix = path.with_name(path.name[: -len(SENSOR_SUFFIX)])
    else:
        prefix = path

    return SessionPaths(
        sensor=Path(str(prefix) + SENSOR_SUFFIX),
        events=Path(str(prefix) + EVENTS_SUFFIX),
        attempts=Path(str(prefix) + ATTEMPTS_SUFFIX),
        prompts=Path(str(prefix) + PROMPTS_SUFFIX),
        meta=Path(str(prefix) + META_SUFFIX),
        activity_log=Path(str(prefix) + ACTIVITY_LOG_SUFFIX),
        protocol=Path(str(prefix) + PROTOCOL_SUFFIX),
        session_ref=str(prefix),
    )


class PasswordSessionLoader:
    """
    Load one password-like session from either prefix-style or directory-style
    storage.
    """

    SENSOR_TIME_COLS = ["timestamp", "timestamp_ns", "time", "ts"]
    SENSOR_ACCEL_COLS = [
        ("accel_x", "accel_y", "accel_z"),
        ("acc_x", "acc_y", "acc_z"),
        ("ax", "ay", "az"),
    ]
    SENSOR_GYRO_COLS = [
        ("gyro_x", "gyro_y", "gyro_z"),
        ("gyr_x", "gyr_y", "gyr_z"),
        ("gx", "gy", "gz"),
    ]

    def __init__(self, session_ref: str):
        self.paths = resolve_session_paths(session_ref)
        self._sensor_rows = None
        self._sensor_columns = None
        self._events_rows = None
        self._attempts_rows = None

    @staticmethod
    def _find_col(columns, candidates):
        for cand in candidates:
            if isinstance(cand, tuple):
                if all(x in columns for x in cand):
                    return cand
            elif cand in columns:
                return cand
        return None

    @property
    def sensor_rows(self):
        if self._sensor_rows is None:
            if not self.paths.sensor.exists():
                raise FileNotFoundError(f"Missing sensor file: {self.paths.sensor}")
            with open(self.paths.sensor, newline="") as f:
                reader = csv.DictReader(f)
                self._sensor_columns = list(reader.fieldnames or [])
                self._sensor_rows = list(reader)
        return self._sensor_rows

    @property
    def sensor_columns(self):
        if self._sensor_columns is None:
            _ = self.sensor_rows
        return self._sensor_columns or []

    @property
    def events_rows(self):
        if self._events_rows is None:
            if self.paths.events and self.paths.events.exists():
                with open(self.paths.events, newline="") as f:
                    self._events_rows = list(csv.DictReader(f))
            else:
                self._events_rows = []
        return self._events_rows

    @property
    def attempts_rows(self):
        if self._attempts_rows is None:
            if self.paths.attempts and self.paths.attempts.exists():
                with open(self.paths.attempts, newline="") as f:
                    self._attempts_rows = list(csv.DictReader(f))
            else:
                self._attempts_rows = []
        return self._attempts_rows

    def get_imu_data(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = self.sensor_rows
        cols = self.sensor_columns
        if not rows:
            raise ValueError(f"Empty sensor file: {self.paths.sensor}")

        time_col = self._find_col(cols, self.SENSOR_TIME_COLS) or cols[0]
        accel_cols = self._find_col(cols, self.SENSOR_ACCEL_COLS) or tuple(cols[1:4])
        gyro_cols = self._find_col(cols, self.SENSOR_GYRO_COLS) or tuple(cols[4:7])

        timestamps = np.asarray([int(float(row[time_col])) for row in rows], dtype=np.int64)
        data = np.asarray(
            [
                [float(row[c]) for c in list(accel_cols) + list(gyro_cols)]
                for row in rows
            ],
            dtype=np.float32,
        )
        return timestamps, data

    def get_key_events(self) -> List[Dict]:
        rows = self.events_rows
        if not rows:
            return []

        events = []
        for row in rows:
            event = {}
            for col in ("timestamp_ns", "timestamp", "time", "ts"):
                if col in row and row[col] != "":
                    event["timestamp_ns"] = int(row[col])
                    break
            for col in ("key", "key_char", "char"):
                if col in row:
                    event["key"] = str(row[col]).lower()
                    break
            for col in ("type", "event_type", "action"):
                if col in row:
                    event["type"] = str(row[col]).lower()
                    break
            if "timestamp_ns" in event:
                events.append(event)
        return events

    def get_attempts(self) -> List[Dict]:
        rows = self.attempts_rows
        if not rows:
            return []

        attempts = []
        for row in rows:
            attempt = {}
            for col in ("attempt_start_ns", "start_ns", "start"):
                if col in row and row[col] != "":
                    attempt["start_ns"] = int(row[col])
                    break
            for col in ("submit_ns", "end_ns", "end"):
                if col in row and row[col] != "":
                    attempt["end_ns"] = int(row[col])
                    break
            for col in ("prompt_text", "prompt", "target"):
                if col in row:
                    attempt["prompt"] = str(row[col]).lower()
                    break
            for col in ("typed_text", "typed", "input"):
                if col in row:
                    attempt["typed"] = str(row[col]).lower()
                    break
            for col in ("match",):
                if col in row:
                    attempt["match"] = str(row[col]).upper()
                    break
            if "start_ns" in attempt:
                attempts.append(attempt)
        return attempts

    def extract_attempt_segments(
        self,
        target_rate_hz: Optional[int] = 190,
        expected_len: Optional[int] = 8,
        require_match: bool = True,
        pre_margin_ms: int = 120,
        post_margin_ms: int = 180,
    ) -> List[Dict]:
        """
        Extract refined password attempts around the actual typing interval,
        rather than the much longer prompt-to-submit span.
        """
        timestamps, imu_data = self.get_imu_data()
        attempts = self.get_attempts()
        key_events = self.get_key_events()

        press_events = [
            e for e in key_events
            if e.get("type", "") in ("press", "keydown", "down", "p")
        ]

        segments = []
        for att in attempts:
            prompt = (att.get("prompt") or "").strip().lower()
            typed = (att.get("typed") or "").strip().lower()
            if require_match and att.get("match") not in (None, "YES"):
                continue
            if expected_len is not None and len(prompt) != expected_len:
                continue
            if require_match and prompt and typed and prompt != typed:
                continue

            start_ns = int(att["start_ns"])
            end_ns = int(att.get("end_ns", start_ns + int(5e9)))

            supported_presses = [
                e for e in press_events
                if start_ns <= e["timestamp_ns"] <= end_ns and supported_key(e.get("key", ""))
            ]
            enter_presses = [
                e for e in press_events
                if start_ns <= e["timestamp_ns"] <= end_ns and e.get("key") in {"enter", "return"}
            ]
            if expected_len is not None and len(supported_presses) != expected_len:
                continue
            if not supported_presses:
                continue

            seg_start_ns = supported_presses[0]["timestamp_ns"] - int(pre_margin_ms * 1e6)
            if enter_presses:
                seg_end_ns = enter_presses[0]["timestamp_ns"] + int(post_margin_ms * 1e6)
            else:
                seg_end_ns = supported_presses[-1]["timestamp_ns"] + int(post_margin_ms * 1e6)

            mask = (timestamps >= seg_start_ns) & (timestamps <= seg_end_ns)
            indices = np.where(mask)[0]
            if len(indices) < 10:
                continue

            seg_ts_raw = timestamps[indices]
            seg_imu_raw = imu_data[indices]
            seg_ts, seg_imu = _resample_imu_segment(seg_ts_raw, seg_imu_raw, target_rate_hz)

            seg_onsets = []
            seg_chars = []
            for pe in supported_presses:
                sample_idx = int(np.searchsorted(seg_ts, pe["timestamp_ns"], side="left"))
                sample_idx = min(sample_idx, len(seg_ts) - 1)
                seg_onsets.append(sample_idx)
                seg_chars.append(pe.get("key", "?"))

            duration_s = (seg_ts[-1] - seg_ts[0]) / 1e9 if len(seg_ts) > 1 else 0.0
            segments.append({
                "imu": seg_imu.astype(np.float32),
                "timestamps": seg_ts.astype(np.int64),
                "key_onsets": seg_onsets,
                "key_chars": seg_chars,
                "prompt": prompt,
                "duration_s": duration_s,
                "source_session": self.paths.session_ref,
            })

        return segments


class NegativeDataLoader:
    """
    Load all negative clips under ``data/raw/onset_negative``.
    """

    def __init__(self, negative_dir: str, target_rate_hz: Optional[int] = 190):
        self.negative_dir = Path(negative_dir)
        self.target_rate_hz = target_rate_hz
        self._clips = None
        self._clips_by_category = None

    def _load_clips(self) -> Tuple[List[np.ndarray], Dict[str, List[np.ndarray]]]:
        clips = []
        clips_by_category: Dict[str, List[np.ndarray]] = {}
        if not self.negative_dir.exists():
            return clips, clips_by_category

        for sensor_path in sorted(self.negative_dir.rglob(f"*{SENSOR_SUFFIX}")):
            try:
                loader = PasswordSessionLoader(str(sensor_path))
                ts, data = loader.get_imu_data()
                ts, data = _resample_imu_segment(ts, data, self.target_rate_hz)
                if len(data) > 10:
                    clip = data.astype(np.float32)
                    clips.append(clip)
                    category = sensor_path.parent.name
                    clips_by_category.setdefault(category, []).append(clip)
            except Exception:
                continue
        return clips, clips_by_category

    @property
    def clips(self) -> List[np.ndarray]:
        if self._clips is None:
            self._clips, self._clips_by_category = self._load_clips()
        return self._clips

    @property
    def clips_by_category(self) -> Dict[str, List[np.ndarray]]:
        if self._clips_by_category is None:
            self._clips, self._clips_by_category = self._load_clips()
        return self._clips_by_category

    def sample_clip(
        self,
        duration_samples: int,
        rng: Optional[np.random.RandomState] = None,
        category: Optional[str] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.RandomState()
        clip_pool = None
        if category:
            clip_pool = self.clips_by_category.get(category)
        if not clip_pool:
            clip_pool = self.clips
        if not clip_pool:
            return rng.randn(duration_samples, 6).astype(np.float32) * 0.01

        clip = clip_pool[rng.randint(len(clip_pool))]
        if len(clip) <= duration_samples:
            repeats = (duration_samples // max(len(clip), 1)) + 1
            clip = np.tile(clip, (repeats, 1))
        start = rng.randint(0, len(clip) - duration_samples + 1)
        return clip[start:start + duration_samples].copy()


def discover_sessions(data_dir: str, keyword: Optional[str] = None) -> List[str]:
    """
    Discover both directory-style and prefix-style sessions.
    Returns session references that can be fed back into ``PasswordSessionLoader``.
    """
    root = Path(data_dir)
    if not root.exists():
        return []

    sessions = set()
    for sensor_path in root.rglob("sensor.csv"):
        sessions.add(str(sensor_path.parent))
    for sensor_path in root.rglob(f"*{SENSOR_SUFFIX}"):
        sessions.add(str(sensor_path)[: -len(SENSOR_SUFFIX)])

    out = sorted(sessions)
    if keyword:
        out = [x for x in out if keyword in Path(x).name]
    return out


def load_all_password_segments(
    password_dir: str,
    target_rate_hz: Optional[int] = 190,
    expected_len: int = 8,
) -> List[Dict]:
    sessions = discover_sessions(password_dir)
    all_segments = []
    for sess_ref in sessions:
        try:
            loader = PasswordSessionLoader(sess_ref)
            all_segments.extend(
                loader.extract_attempt_segments(
                    target_rate_hz=target_rate_hz,
                    expected_len=expected_len,
                )
            )
        except Exception as exc:
            print(f"Warning: Failed to load {sess_ref}: {exc}")
    print(f"Loaded {len(all_segments)} password segments from {len(sessions)} sessions")
    return all_segments


def load_all_password_blocks(
    password_dir: str,
    target_rate_hz: int = 190,
    block_size: int = 5,
    expected_len: int = 8,
    block_pre_ms: int = 500,
    block_post_ms: int = 500,
    group_pre_ms: int = 120,
    group_post_ms: int = 180,
) -> List[Dict]:
    """
    Build realistic 5-password block templates directly from raw password
    sessions, preserving the true inter-attempt rhythm within a session.
    """
    sessions = discover_sessions(password_dir)
    blocks: List[Dict] = []

    for sess_ref in sessions:
        try:
            loader = PasswordSessionLoader(sess_ref)
            timestamps, imu_data = loader.get_imu_data()
            attempts = loader.get_attempts()
            events = loader.get_key_events()
        except Exception as exc:
            print(f"Warning: Failed to load block templates from {sess_ref}: {exc}")
            continue

        press_events = [
            e for e in events
            if e.get("type", "") in ("press", "keydown", "down", "p")
        ]

        valid_attempts = []
        for att in attempts:
            prompt = (att.get("prompt") or "").strip().lower()
            typed = (att.get("typed") or "").strip().lower()
            if att.get("match") not in (None, "YES"):
                continue
            if len(prompt) != expected_len or (typed and typed != prompt):
                continue

            start_ns = int(att["start_ns"])
            end_ns = int(att.get("end_ns", start_ns + int(5e9)))
            chars = [
                e for e in press_events
                if start_ns <= e["timestamp_ns"] <= end_ns and supported_key(e.get("key", ""))
            ]
            enters = [
                e for e in press_events
                if start_ns <= e["timestamp_ns"] <= end_ns and e.get("key") in {"enter", "return"}
            ]
            if len(chars) != expected_len:
                continue

            valid_attempts.append({
                "prompt": prompt,
                "char_events": chars,
                "enter_event": enters[0] if enters else None,
            })

        if len(valid_attempts) < block_size:
            continue

        for start_idx in range(0, len(valid_attempts) - block_size + 1):
            chunk = valid_attempts[start_idx:start_idx + block_size]
            block_start_ns = int(chunk[0]["char_events"][0]["timestamp_ns"]) - int(block_pre_ms * 1e6)
            tail_end_ns = (
                int(chunk[-1]["enter_event"]["timestamp_ns"])
                if chunk[-1]["enter_event"] is not None
                else int(chunk[-1]["char_events"][-1]["timestamp_ns"])
            )
            block_end_ns = tail_end_ns + int(block_post_ms * 1e6)

            mask = (timestamps >= block_start_ns) & (timestamps <= block_end_ns)
            idx = np.where(mask)[0]
            if len(idx) < 10:
                continue

            raw_ts = timestamps[idx]
            raw_imu = imu_data[idx]
            block_ts, block_imu = _resample_imu_segment(raw_ts, raw_imu, target_rate_hz)
            if len(block_ts) < 10:
                continue

            group_boundaries = []
            onset_positions = []
            onset_chars = []
            group_labels = np.zeros(len(block_ts), dtype=np.float32)

            for att in chunk:
                group_start_ns = int(att["char_events"][0]["timestamp_ns"]) - int(group_pre_ms * 1e6)
                group_end_anchor = (
                    int(att["enter_event"]["timestamp_ns"])
                    if att["enter_event"] is not None
                    else int(att["char_events"][-1]["timestamp_ns"])
                )
                group_end_ns = group_end_anchor + int(group_post_ms * 1e6)

                g_start = int(np.searchsorted(block_ts, group_start_ns, side="left"))
                g_end = int(np.searchsorted(block_ts, group_end_ns, side="right"))
                g_end = min(g_end, len(block_ts))
                if g_end <= g_start:
                    continue

                group_boundaries.append((g_start, g_end))
                group_labels[g_start:g_end] = 1.0

                local_onsets = []
                chars = []
                for ev in att["char_events"]:
                    pos = int(np.searchsorted(block_ts, int(ev["timestamp_ns"]), side="left"))
                    pos = min(pos, len(block_ts) - 1)
                    local_onsets.append(pos)
                    chars.append(ev.get("key", "?"))
                onset_positions.append(local_onsets)
                onset_chars.append(chars)

            if len(group_boundaries) != block_size:
                continue

            blocks.append({
                "imu": block_imu.astype(np.float32),
                "group_labels": group_labels,
                "group_boundaries": group_boundaries,
                "onset_positions": onset_positions,
                "onset_chars": onset_chars,
                "num_groups": block_size,
                "keys_per_group": expected_len,
                "source_session": sess_ref,
            })

    print(f"Loaded {len(blocks)} password block templates from {len(sessions)} sessions")
    return blocks


def load_mixed2_session(
    session_ref: str,
    target_rate_hz: int = 190,
    pad_ms: int = 500,
    group_pre_ms: int = 120,
    group_post_ms: int = 180,
) -> Optional[Dict]:
    """
    Load one mixed2 session and derive GT group boundaries / onsets from
    ``activity_log`` and key events.
    """
    loader = PasswordSessionLoader(session_ref)
    timestamps, full_imu = loader.get_imu_data()
    events = loader.get_key_events()
    paths = loader.paths

    if not paths.activity_log or not paths.activity_log.exists():
        return None

    with open(paths.activity_log, newline="") as f:
        activity_rows = list(csv.DictReader(f))
    password_rows = [
        row for row in activity_rows
        if row.get("activity") == "keyboard"
        and (row.get("typing_style") == "password" or row.get("label") == "typing_2")
    ]
    if not password_rows:
        return None

    block_start_ns = min(int(row["start_time_ns"]) for row in password_rows)
    block_end_ns = max(int(row["end_time_ns"]) for row in password_rows)
    prompts = []
    for row in password_rows:
        prompts.extend(_parse_prompts_field(row.get("prompts")))

    region_start_ns = block_start_ns - int(pad_ms * 1e6)
    region_end_ns = block_end_ns + int(pad_ms * 1e6)
    mask = (timestamps >= region_start_ns) & (timestamps <= region_end_ns)
    idx = np.where(mask)[0]
    if len(idx) < 10:
        return None

    region_ts_raw = timestamps[idx]
    region_imu_raw = full_imu[idx]
    region_ts, region_imu = _resample_imu_segment(region_ts_raw, region_imu_raw, target_rate_hz)

    press_events = [
        e for e in events
        if e.get("type", "") in ("press", "keydown", "down", "p")
        and block_start_ns <= e["timestamp_ns"] <= block_end_ns
        and e.get("key") not in IGNORED_KEYS
    ]

    gt_groups_ns: List[Tuple[int, int]] = []
    gt_onsets_ns: List[List[int]] = []
    gt_chars: List[List[str]] = []

    cur_char_ns: List[int] = []
    cur_chars: List[str] = []
    cur_start_ns: Optional[int] = None
    cur_end_ns: Optional[int] = None

    def flush_group():
        if not cur_char_ns:
            return
        gt_groups_ns.append((
            cur_start_ns - int(group_pre_ms * 1e6),
            cur_end_ns + int(group_post_ms * 1e6),
        ))
        gt_onsets_ns.append(cur_char_ns.copy())
        gt_chars.append(cur_chars.copy())

    for ev in press_events:
        key = ev.get("key", "")
        ts_ns = int(ev["timestamp_ns"])
        if key in {"enter", "return"}:
            flush_group()
            cur_char_ns = []
            cur_chars = []
            cur_start_ns = None
            cur_end_ns = None
            continue
        if not supported_key(key):
            continue
        if cur_start_ns is None:
            cur_start_ns = ts_ns
        cur_end_ns = ts_ns
        cur_char_ns.append(ts_ns)
        cur_chars.append(key)

    flush_group()

    if prompts and len(gt_groups_ns) > len(prompts):
        gt_groups_ns = gt_groups_ns[: len(prompts)]
        gt_onsets_ns = gt_onsets_ns[: len(prompts)]
        gt_chars = gt_chars[: len(prompts)]

    gt_group_boundaries = []
    gt_onset_positions = []
    for (start_ns, end_ns), group_ns in zip(gt_groups_ns, gt_onsets_ns):
        start_idx = int(np.searchsorted(region_ts, start_ns, side="left"))
        end_idx = int(np.searchsorted(region_ts, end_ns, side="right"))
        gt_group_boundaries.append((start_idx, min(end_idx, len(region_ts))))

        group_idx = []
        for onset_ns in group_ns:
            idx_pos = int(np.searchsorted(region_ts, onset_ns, side="left"))
            idx_pos = min(idx_pos, len(region_ts) - 1)
            group_idx.append(idx_pos)
        gt_onset_positions.append(group_idx)

    return {
        "region_imu": region_imu.astype(np.float32),
        "region_timestamps": region_ts.astype(np.int64),
        "gt_group_boundaries": gt_group_boundaries,
        "gt_onset_positions": gt_onset_positions,
        "gt_chars": gt_chars,
        "num_groups": len(gt_group_boundaries),
        "prompts": prompts,
        "session_ref": loader.paths.session_ref,
    }
