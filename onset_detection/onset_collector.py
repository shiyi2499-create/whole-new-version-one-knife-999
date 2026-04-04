"""
Onset Data Collector
====================
Collect negative-sample and mixed-stream sessions for onset detection.

Four modes:
  1. negative  – Record a single activity type (idle, trackpad_move, freetyping, etc.)
  2. mixed     – Record a scripted sequence of random interleaved activities
  3. mixed2    – Record the structured ~3-minute protocol:
                 idle → trackpad_move → typing_1 (free) → trackpad_click →
                 idle → typing_2 (password) → shake
                 with explicit segment boundaries and labels
  4. mixed_training – Same structured skeleton as mixed2, but saved separately
                 for Stage 2 training and with light per-segment duration jitter
  5. mixed_single_training – Structured single-password stream:
                 idle → trackpad_move → typing_1 (free) → idle → typing_2 (1 password) → shake
  6. mixed_retry_training – Structured retry-style stream:
                 idle → trackpad_move → typing_1 (free) → idle →
                 typing_2 (1 password) → interference(20s) → typing_3 (1 password) → shake

Reuses the existing sensor_reader / spu_backend / keyboard_listener stack.

Run:
  python3 onset_collector.py --mode negative --activity idle --duration 60
  python3 onset_collector.py --mode negative --activity freetyping --duration 60
  python3 onset_collector.py --mode mixed --n-segments 15 --segment-sec 30
  python3 onset_collector.py --mode mixed2 --n-trials 5
  python3 onset_collector.py --mode mixed_training --n-trials 20
  python3 onset_collector.py --mode mixed_single_training --n-trials 3
  python3 onset_collector.py --mode mixed_retry_training --n-trials 3

Data is saved under:
  data/raw/onset_negative/<activity>/    (for negative mode)
  data/raw/onset_mixed/                  (for mixed mode)
  data/raw/onset_mixed2/                 (for mixed2 mode)
  data/raw/mixed_training/               (for mixed_training mode)
  data/raw/mixed_single_training/        (for mixed_single_training mode)
  data/raw/mixed_retry_training/         (for mixed_retry_training mode)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import threading
import random
import secrets
from datetime import datetime
from typing import Optional

_PROJECT_ROOT = None


def _setup_imports(project_root: str = ""):
    global _PROJECT_ROOT
    if project_root:
        root = os.path.abspath(project_root)
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _PROJECT_ROOT = root
    if root not in sys.path:
        sys.path.insert(0, root)


_setup_imports()

from sensor_reader import SensorReader, SensorSample
from keyboard_listener import KeyboardListener, KeyEvent


ACTIVITIES = [
    "idle",
    "trackpad_move",
    "trackpad_use",
    "trackpad_click",
    "shake",
    "desk_bump",
    "freetyping",
    "keyboard",
]

NEGATIVE_ACTIVITIES = [a for a in ACTIVITIES if a != "keyboard"]
PASSWORD_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


# ── Helpers ──────────────────────────────────────────────────

def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_sensor_csv(path: str):
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["timestamp_ns", "accel_x", "accel_y", "accel_z",
                      "gyro_x", "gyro_y", "gyro_z"])
    return f, writer


def open_events_csv(path: str):
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["timestamp_ns", "key", "event_type", "participant_id", "session_id"])
    return f, writer


def write_sensor_rows(writer, samples: list[SensorSample]):
    for s in samples:
        writer.writerow([
            s.timestamp_ns,
            f"{s.accel_x:.8f}", f"{s.accel_y:.8f}", f"{s.accel_z:.8f}",
            f"{s.gyro_x:.6f}",  f"{s.gyro_y:.6f}",  f"{s.gyro_z:.6f}",
        ])


def write_event_rows(writer, events: list[KeyEvent], participant: str, session: str):
    for e in events:
        writer.writerow([e.timestamp_ns, e.key, e.event_type, participant, session])


def _normalize_password_prompt(text: str) -> Optional[str]:
    s = (text or "").strip().lower()
    if len(s) != 8:
        return None
    if any(ch not in PASSWORD_CHARS for ch in s):
        return None
    return s


def _load_existing_password_prompts(project_root: Optional[str] = None) -> set[str]:
    root = project_root or _PROJECT_ROOT or os.getcwd()
    used: set[str] = set()

    search_roots = [
        os.path.join(root, "data/raw/password"),
        os.path.join(root, "data/raw/onset_mixed2"),
        os.path.join(root, "data/raw/mixed_training"),
    ]

    for base in search_roots:
        if not os.path.exists(base):
            continue

        for dirpath, _, filenames in os.walk(base):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    if name.endswith("_prompts.csv") or name.endswith("_attempts.csv"):
                        with open(path, "r", newline="") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                for key in ("prompt_text", "typed_text"):
                                    pw = _normalize_password_prompt(row.get(key, ""))
                                    if pw:
                                        used.add(pw)
                    elif name.endswith("_protocol.json"):
                        with open(path, "r") as f:
                            obj = json.load(f)
                        for seg in obj.get("protocol", []):
                            for item in seg.get("prompts", []) or []:
                                pw = _normalize_password_prompt(item)
                                if pw:
                                    used.add(pw)
                except Exception:
                    continue

    return used


def _generate_fresh_passwords(
    n_passwords: int,
    rng: random.Random,
    used_prompts: Optional[set[str]] = None,
    password_length: int = 8,
) -> list[str]:
    used = set(used_prompts or set())
    out: list[str] = []
    max_tries = max(10_000, n_passwords * 500)

    for _ in range(max_tries):
        if len(out) >= n_passwords:
            break
        candidate = "".join(rng.choices(PASSWORD_CHARS, k=int(password_length)))
        if candidate in used:
            continue
        out.append(candidate)
        used.add(candidate)

    if len(out) != n_passwords:
        raise RuntimeError(
            f"Failed to generate {n_passwords} fresh passwords without repeats."
        )
    return out


# ── Negative mode ────────────────────────────────────────────

def run_negative_mode(
    activity: str,
    duration_sec: float,
    output_dir: str,
    participant: str = "p01",
    gate_rate_hz: float = 150.0,
    precheck_sec: float = 3.0,
    spu_report_interval_us: int = 5000,
    spu_device_rate_control: bool = False,
):
    """Record a single-activity negative session."""
    assert activity in NEGATIVE_ACTIVITIES, f"Unknown activity: {activity}"

    out_dir = os.path.join(output_dir, activity)
    os.makedirs(out_dir, exist_ok=True)

    tag = timestamp_tag()
    session_id = f"{participant}_onset_neg_{activity}_{tag}"
    sensor_path = os.path.join(out_dir, f"{session_id}_sensor.csv")
    meta_path = os.path.join(out_dir, f"{session_id}_meta.json")
    events_path = os.path.join(out_dir, f"{session_id}_events.csv")

    sensor = SensorReader(
        spu_report_interval_us=spu_report_interval_us,
        spu_device_rate_control=spu_device_rate_control,
    )
    sensor.start()
    keyboard = None
    ef = None
    ew = None
    event_count = 0
    if activity == "freetyping":
        keyboard = KeyboardListener()
        keyboard.start()

    print(f"\n{'='*60}")
    print(f"  ONSET NEGATIVE COLLECTION")
    print(f"  Activity:  {activity}")
    print(f"  Duration:  {duration_sec:.0f}s")
    print(f"  Output:    {sensor_path}")
    if activity == "freetyping":
        print(f"  Events:    {events_path}")
    print(f"{'='*60}\n")

    print(f"  Warming up sensor ({precheck_sec:.0f}s)...")
    time.sleep(precheck_sec)
    sensor.drain()
    if keyboard is not None:
        keyboard.drain()
        ef, ew = open_events_csv(events_path)

    sf, sw = open_sensor_csv(sensor_path)
    sample_count = 0

    if activity == "freetyping":
        print(f"  🔴 现在开始 [自由敲击]，持续 {duration_sec:.0f} 秒")
        print(f"     随意输入单词、短句、乱打字都可以，不要刻意模仿 password")
        print(f"     可以自然地按 Enter / Backspace / Space")
        print(f"     （按 Ctrl+C 可提前停止）\n")
    else:
        print(f"  🔴 NOW: perform [{activity}] for {duration_sec:.0f}s")
        print(f"     (press Ctrl+C to stop early)\n")

    t_start = time.time()
    try:
        while time.time() - t_start < duration_sec:
            samples = sensor.drain()
            if samples:
                write_sensor_rows(sw, samples)
                sample_count += len(samples)
            if keyboard is not None:
                events = keyboard.drain()
                if events:
                    write_event_rows(ew, events, participant, session_id)
                    event_count += len(events)
            elapsed = time.time() - t_start
            remaining = max(0, duration_sec - elapsed)
            if keyboard is not None:
                print(f"\r  [{elapsed:.0f}s / {duration_sec:.0f}s]  "
                      f"samples={sample_count:,}  events={event_count:,}  remaining={remaining:.0f}s  ",
                      end="", flush=True)
            else:
                print(f"\r  [{elapsed:.0f}s / {duration_sec:.0f}s]  "
                      f"samples={sample_count:,}  remaining={remaining:.0f}s  ",
                      end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n  Stopped early by user.")

    samples = sensor.drain()
    if samples:
        write_sensor_rows(sw, samples)
        sample_count += len(samples)
    if keyboard is not None:
        events = keyboard.drain()
        if events:
            write_event_rows(ew, events, participant, session_id)
            event_count += len(events)

    sf.flush()
    sf.close()
    sensor.stop()
    if ef is not None:
        ef.flush()
        ef.close()
    if keyboard is not None:
        keyboard.stop()

    elapsed = time.time() - t_start
    rate = sample_count / max(elapsed, 0.1)

    meta = {
        "session_id": session_id,
        "activity": activity,
        "type": "negative",
        "duration_s": elapsed,
        "sample_count": sample_count,
        "avg_rate_hz": rate,
        "participant": participant,
        "spu_report_interval_us": int(spu_report_interval_us),
        "spu_device_rate_control": bool(spu_device_rate_control),
        "sensor_backend": sensor.backend_name,
    }
    if activity == "freetyping":
        meta["event_count"] = event_count
        meta["events_path"] = events_path
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if activity == "freetyping":
        print(f"\n\n  ✓ Done: {sample_count:,} samples + {event_count:,} key events in {elapsed:.1f}s ({rate:.1f} Hz)")
        print(f"  Saved → {sensor_path}")
        print(f"  Events → {events_path}")
    else:
        print(f"\n\n  ✓ Done: {sample_count:,} samples in {elapsed:.1f}s ({rate:.1f} Hz)")
        print(f"  Saved → {sensor_path}")
    print(f"  Meta  → {meta_path}\n")


# ── Mixed mode (original random) ────────────────────────────

def generate_mixed_script(
    n_segments: int = 15,
    segment_sec: float = 30.0,
    seed: int = 42,
) -> list[dict]:
    rng = random.Random(seed)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"

    def random_password():
        return "".join(rng.choices(chars, k=8))

    scripts = []
    for i in range(n_segments):
        n_blocks = rng.randint(4, 6)
        blocks = []
        total = 0.0

        for j in range(n_blocks):
            if j == 0:
                activity = "idle"
            else:
                activity = rng.choice(ACTIVITIES)

            if activity == "keyboard":
                dur = rng.uniform(3.0, 5.0)
                prompt = random_password()
            elif activity == "idle":
                dur = rng.uniform(2.0, 4.0)
                prompt = None
            else:
                dur = rng.uniform(2.0, 4.0)
                prompt = None

            blocks.append({
                "activity": activity,
                "duration_s": round(dur, 1),
                "prompt": prompt,
            })
            total += dur

        scripts.append({
            "segment_idx": i,
            "blocks": blocks,
            "total_duration_s": round(total, 1),
        })

    return scripts


def run_mixed_mode(
    n_segments: int = 15,
    segment_sec: float = 30.0,
    output_dir: str = "data/raw/onset_mixed",
    participant: str = "p01",
    seed: int = 42,
):
    """Record mixed-stream segments for onset detection evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    scripts = generate_mixed_script(n_segments, segment_sec, seed)
    tag = timestamp_tag()

    sensor = SensorReader()
    keyboard = KeyboardListener()
    sensor.start()
    keyboard.start()

    print(f"\n{'='*60}")
    print(f"  ONSET MIXED-STREAM COLLECTION")
    print(f"  Segments:  {n_segments}")
    print(f"  Output:    {output_dir}")
    print(f"{'='*60}\n")

    print("  Warming up sensor (3s)...")
    time.sleep(3)
    sensor.drain()
    keyboard.drain()

    for seg in scripts:
        idx = seg["segment_idx"]
        session_id = f"{participant}_onset_mixed_seg{idx:03d}_{tag}"
        sensor_path = os.path.join(output_dir, f"{session_id}_sensor.csv")
        events_path = os.path.join(output_dir, f"{session_id}_events.csv")
        script_path = os.path.join(output_dir, f"{session_id}_script.json")
        activity_log_path = os.path.join(output_dir, f"{session_id}_activity_log.csv")

        print(f"\n  ── Segment {idx+1}/{n_segments} ──")

        sf, sw = open_sensor_csv(sensor_path)
        ef, ew = open_events_csv(events_path)

        alf = open(activity_log_path, "w", newline="")
        alw = csv.writer(alf)
        alw.writerow(["start_time_ns", "end_time_ns", "activity", "prompt"])

        sample_count = 0
        event_count = 0
        stop = threading.Event()

        def drain_fn():
            nonlocal sample_count, event_count
            while not stop.is_set():
                samples = sensor.drain()
                if samples:
                    write_sensor_rows(sw, samples)
                    sample_count += len(samples)
                time.sleep(0.05)

        drain_t = threading.Thread(target=drain_fn, daemon=True)
        drain_t.start()

        input(f"  Press ENTER to start segment {idx+1} →")
        keyboard.drain()

        for block in seg["blocks"]:
            act = block["activity"]
            dur = block["duration_s"]
            prompt = block.get("prompt")

            if act == "keyboard" and prompt:
                print(f"    🔵 [{act}] Type: {prompt}  ({dur:.1f}s)")
            else:
                print(f"    🟡 [{act}]  ({dur:.1f}s)")

            block_start_ns = time.perf_counter_ns()
            block_start_wall = time.time()

            if act == "keyboard" and prompt:
                t_end = time.time() + dur
                while time.time() < t_end:
                    events = keyboard.drain()
                    if events:
                        write_event_rows(ew, events, participant, session_id)
                        event_count += len(events)
                    time.sleep(0.02)
            else:
                time.sleep(dur)

            block_end_ns = time.perf_counter_ns()
            alw.writerow([block_start_ns, block_end_ns, act, prompt or ""])

        stop.set()
        drain_t.join(timeout=2.0)

        samples = sensor.drain()
        if samples:
            write_sensor_rows(sw, samples)
            sample_count += len(samples)
        events = keyboard.drain()
        if events:
            write_event_rows(ew, events, participant, session_id)
            event_count += len(events)

        sf.flush(); sf.close()
        ef.flush(); ef.close()
        alf.flush(); alf.close()

        with open(script_path, "w") as f:
            json.dump(seg, f, indent=2)

        print(f"    ✓ {sample_count:,} sensor + {event_count} key events")

    sensor.stop()
    keyboard.stop()
    print(f"\n  ✓ All {n_segments} segments recorded.")
    print(f"  Output → {output_dir}\n")


# ══════════════════════════════════════════════════════════════
# Mixed2 mode: structured ~3-minute protocol
# ══════════════════════════════════════════════════════════════

# Default protocol: 7 segments, ~167s total
# Segment labels are used as ground-truth for mixed-stream boundary supervision.
# typing_1 = free typing (random text, faster)
# typing_2 = password-style typing (8-char, slower, controlled)

DEFAULT_MIXED2_PROTOCOL = [
    {"activity": "idle",            "duration_s": 12.0, "label": "idle_1"},
    {"activity": "trackpad_move",   "duration_s": 18.0, "label": "trackpad_move_1"},
    {"activity": "keyboard",        "duration_s": 35.0, "label": "typing_1",
     "typing_style": "free",
     "prompt_instructions": "Type whatever you want – random words, sentences, etc."},
    {"activity": "trackpad_click",  "duration_s": 18.0, "label": "trackpad_click_1"},
    {"activity": "idle",            "duration_s": 12.0, "label": "idle_2"},
    {"activity": "keyboard",        "duration_s": 60.0, "label": "typing_2",
     "typing_style": "password",
     "prompt_instructions": ""},  # filled with actual password prompts
    {"activity": "shake",           "duration_s": 12.0, "label": "shake_1"},
]

DEFAULT_MIXED_TRAINING_JITTER_PCT = 0.15

DISPLAY_LABELS_ZH = {
    "idle_1": "静止阶段 1",
    "idle_2": "密码前静止阶段",
    "trackpad_move_1": "触控板滑动阶段",
    "trackpad_click_1": "触控板点击阶段",
    "shake_1": "整机晃动收尾阶段",
    "typing_1": "自由敲击阶段",
    "typing_2": "密码输入阶段",
    "typing_3": "第二次密码输入阶段",
    "interference_1": "中间自由活动阶段",
}

TYPING_STYLE_ZH = {
    "free": "自由敲击",
    "password": "密码输入",
}


DEFAULT_MIXED_SINGLE_PROTOCOL = [
    {"activity": "idle",          "duration_s": 12.0, "label": "idle_1"},
    {"activity": "trackpad_move", "duration_s": 18.0, "label": "trackpad_move_1"},
    {"activity": "keyboard",      "duration_s": 30.0, "label": "typing_1",
     "typing_style": "free",
     "prompt_instructions": "Type whatever you want – random words, sentences, etc."},
    {"activity": "idle",          "duration_s": 10.0, "label": "idle_2"},
    {"activity": "keyboard",      "duration_s": 18.0, "label": "typing_2",
     "typing_style": "password",
     "prompt_instructions": ""},
    {"activity": "shake",         "duration_s": 10.0, "label": "shake_1"},
]

DEFAULT_MIXED_RETRY_PROTOCOL = [
    {"activity": "idle",          "duration_s": 12.0, "label": "idle_1"},
    {"activity": "trackpad_move", "duration_s": 18.0, "label": "trackpad_move_1"},
    {"activity": "keyboard",      "duration_s": 30.0, "label": "typing_1",
     "typing_style": "free",
     "prompt_instructions": "Type whatever you want – random words, sentences, etc."},
    {"activity": "idle",          "duration_s": 10.0, "label": "idle_2"},
    {"activity": "keyboard",      "duration_s": 18.0, "label": "typing_2",
     "typing_style": "password",
     "prompt_instructions": ""},
    {"activity": "interference",  "duration_s": 20.0, "label": "interference_1",
     "prompt_instructions": "这 20 秒请随意活动：可静止、滑动触控板、敲击触控板、轻微移动设备。"},
    {"activity": "keyboard",      "duration_s": 18.0, "label": "typing_3",
     "typing_style": "password",
     "prompt_instructions": ""},
    {"activity": "shake",         "duration_s": 10.0, "label": "shake_1"},
]


def _jitter_duration(duration_s: float,
                     rng: random.Random,
                     jitter_pct: float,
                     min_duration_s: float = 3.0) -> float:
    if jitter_pct <= 0:
        return float(duration_s)
    scale = 1.0 + rng.uniform(-jitter_pct, jitter_pct)
    return round(max(min_duration_s, duration_s * scale), 1)


def generate_structured_protocol(
    n_passwords: int = 5,
    seed: int = 42,
    duration_jitter_pct: float = 0.0,
    used_prompts: Optional[set[str]] = None,
    password_length: int = 8,
) -> list[dict]:
    """
    Generate the structured ~3-minute mixed-stream protocol.
    The typing_2 segment gets n_passwords random 8-char password prompts.
    """
    rng = random.Random(seed)
    protocol = []
    for seg in DEFAULT_MIXED2_PROTOCOL:
        entry = dict(seg)
        base_duration = float(entry["duration_s"])
        if entry.get("typing_style") == "password":
            # Password stage ends on the final Enter, so keep this as a
            # reference duration in the protocol metadata instead of a hard cap.
            entry["duration_s"] = base_duration
        else:
            entry["duration_s"] = _jitter_duration(
                base_duration,
                rng,
                jitter_pct=duration_jitter_pct,
            )
        if entry.get("typing_style") == "password":
            passwords = _generate_fresh_passwords(
                n_passwords=n_passwords,
                rng=rng,
                used_prompts=used_prompts,
                password_length=password_length,
            )
            entry["prompts"] = passwords
            entry["prompt_instructions"] = (
                f"请慢速、仔细地输入下面每一条密码；每输入完一条后按一次 Enter。\n"
                f"这一阶段不设硬性倒计时，输完第 {n_passwords} 条并按下 Enter 后自动进入下一阶段：\n"
                + "\n".join(f"  {i+1}. {pw}" for i, pw in enumerate(passwords))
            )
        protocol.append(entry)

    return protocol


def generate_single_password_protocol(
    seed: int = 42,
    duration_jitter_pct: float = 0.0,
    used_prompts: Optional[set[str]] = None,
    password_length: int = 8,
) -> list[dict]:
    rng = random.Random(seed)
    protocol = []
    for seg in DEFAULT_MIXED_SINGLE_PROTOCOL:
        entry = dict(seg)
        base_duration = float(entry["duration_s"])
        if entry.get("typing_style") == "password":
            entry["duration_s"] = base_duration
            passwords = _generate_fresh_passwords(
                n_passwords=1,
                rng=rng,
                used_prompts=used_prompts,
                password_length=password_length,
            )
            entry["prompts"] = passwords
            entry["prompt_instructions"] = (
                "请慢速、仔细地输入下面这 1 条密码；输入完后按一次 Enter。\n"
                "这一阶段不设硬性倒计时，按下 Enter 后自动进入下一阶段：\n"
                f"  1. {passwords[0]}"
            )
        else:
            entry["duration_s"] = _jitter_duration(
                base_duration, rng, jitter_pct=duration_jitter_pct
            )
        protocol.append(entry)
    return protocol


def generate_retry_password_protocol(
    seed: int = 42,
    duration_jitter_pct: float = 0.0,
    used_prompts: Optional[set[str]] = None,
    password_length: int = 8,
) -> list[dict]:
    rng = random.Random(seed)
    protocol = []
    pending_passwords = _generate_fresh_passwords(
        n_passwords=2,
        rng=rng,
        used_prompts=used_prompts,
        password_length=password_length,
    )
    pw_idx = 0
    for seg in DEFAULT_MIXED_RETRY_PROTOCOL:
        entry = dict(seg)
        base_duration = float(entry["duration_s"])
        if entry.get("typing_style") == "password":
            entry["duration_s"] = base_duration
            password = pending_passwords[pw_idx]
            pw_idx += 1
            entry["prompts"] = [password]
            phase_name = "第一次" if entry["label"] == "typing_2" else "第二次"
            entry["prompt_instructions"] = (
                f"请慢速、仔细地输入下面这 1 条密码（{phase_name}密码输入）；输入完后按一次 Enter。\n"
                "这一阶段不设硬性倒计时，按下 Enter 后自动进入下一阶段：\n"
                f"  1. {password}"
            )
        else:
            entry["duration_s"] = _jitter_duration(
                base_duration, rng, jitter_pct=duration_jitter_pct
            )
        protocol.append(entry)
    return protocol


def run_structured_mode(
    n_trials: int = 5,
    n_passwords: int = 5,
    output_dir: str = "data/raw/onset_mixed2",
    participant: str = "p01",
    seed: int = 42,
    mode_tag: str = "mixed2",
    duration_jitter_pct: float = 0.0,
    password_length: int = 8,
    spu_report_interval_us: int = 5000,
    spu_device_rate_control: bool = False,
):
    """
    Record the structured ~3-minute mixed-stream protocol.

    Each trial produces:
      - <session_id>_sensor.csv
      - <session_id>_events.csv
      - <session_id>_activity_log.csv   ← ground-truth segment boundaries
      - <session_id>_protocol.json
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = timestamp_tag()

    sensor = SensorReader(
        spu_report_interval_us=int(spu_report_interval_us),
        spu_device_rate_control=bool(spu_device_rate_control),
    )
    keyboard = KeyboardListener()
    sensor.start()
    keyboard.start()

    title = "structured mixed2 hold-out collection"
    if mode_tag == "mixed_training":
        title = "structured mixed-training collection"

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  轮数：      {n_trials}")
    print(f"  每轮密码数：{n_passwords}")
    print(f"  时长抖动：  ±{duration_jitter_pct * 100:.0f}%")
    print(f"  输出目录：  {output_dir}")
    print(f"  SPU间隔：   {int(spu_report_interval_us)} us")
    print(f"  Device ctl: {bool(spu_device_rate_control)}")
    print(f"{'='*60}\n")

    print("  传感器预热中（3 秒）...")
    time.sleep(3)
    sensor.drain()
    keyboard.drain()
    used_prompts = _load_existing_password_prompts()

    for trial_idx in range(n_trials):
        trial_seed = seed + trial_idx
        protocol = generate_structured_protocol(
            n_passwords=n_passwords,
            seed=trial_seed,
            duration_jitter_pct=duration_jitter_pct,
            used_prompts=used_prompts,
            password_length=password_length,
        )
        for seg in protocol:
            if seg.get("typing_style") == "password":
                used_prompts.update(seg.get("prompts", []) or [])

        session_id = f"{participant}_{mode_tag}_trial{trial_idx:03d}_{tag}"
        sensor_path = os.path.join(output_dir, f"{session_id}_sensor.csv")
        events_path = os.path.join(output_dir, f"{session_id}_events.csv")
        activity_log_path = os.path.join(output_dir, f"{session_id}_activity_log.csv")
        protocol_path = os.path.join(output_dir, f"{session_id}_protocol.json")

        total_dur = sum(s["duration_s"] for s in protocol)
        print(f"\n  ══ 第 {trial_idx+1}/{n_trials} 轮（约 {total_dur:.0f} 秒）══")

        sf, sw = open_sensor_csv(sensor_path)
        ef, ew = open_events_csv(events_path)

        # Activity log with explicit segment boundaries
        alf = open(activity_log_path, "w", newline="")
        alw = csv.writer(alf)
        alw.writerow([
            "start_time_ns", "end_time_ns", "activity",
            "label", "typing_style", "prompts",
        ])

        sample_count = 0
        event_count = 0
        stop = threading.Event()

        def drain_fn():
            nonlocal sample_count
            while not stop.is_set():
                samples = sensor.drain()
                if samples:
                    write_sensor_rows(sw, samples)
                    sample_count += len(samples)
                time.sleep(0.05)

        drain_t = threading.Thread(target=drain_fn, daemon=True)
        drain_t.start()

        input(f"  按回车开始第 {trial_idx+1} 轮 →")
        keyboard.drain()

        for seg_idx, seg in enumerate(protocol):
            act = seg["activity"]
            dur = seg["duration_s"]
            label = seg["label"]
            typing_style = seg.get("typing_style", "")
            label_zh = DISPLAY_LABELS_ZH.get(label, label)
            typing_style_zh = TYPING_STYLE_ZH.get(typing_style, typing_style)

            if act == "keyboard":
                prompt_text = seg.get("prompt_instructions", "")
                if typing_style == "password":
                    print(f"    🔵 [{label_zh}]（无硬性时限）{typing_style_zh}")
                    print("       提醒：请保持英文小写输入，Caps Lock 关闭。")
                else:
                    print(f"    🔵 [{label_zh}]（{dur:.0f} 秒）{typing_style_zh}")
                if prompt_text:
                    for line in prompt_text.split("\n"):
                        print(f"       {line}")
            else:
                print(f"    🟡 [{label_zh}]（{dur:.0f} 秒）")
                if label == "idle_2":
                    print("       提醒：下一阶段是 password，请在这段 idle 内切到英文小写。")

            block_start_ns = time.perf_counter_ns()
            block_start_wall = time.time()

            if act == "keyboard" and typing_style == "password":
                target_enters = len(seg.get("prompts", [])) or n_passwords
                enter_count = 0
                while enter_count < target_enters:
                    events = keyboard.drain()
                    if events:
                        write_event_rows(ew, events, participant, session_id)
                        event_count += len(events)
                        for ev in events:
                            if ev.event_type == "press" and ev.key in {"enter", "return"}:
                                enter_count += 1
                    elapsed = time.time() - block_start_wall
                    print(
                        f"\r      [已输入 {enter_count}/{target_enters} 条] "
                        f"已记录按键={event_count}  已耗时={elapsed:.0f} 秒  ",
                        end="",
                        flush=True,
                    )
                    time.sleep(0.02)
                print()
            elif act == "keyboard":
                t_end = time.time() + dur
                while time.time() < t_end:
                    events = keyboard.drain()
                    if events:
                        write_event_rows(ew, events, participant, session_id)
                        event_count += len(events)
                    elapsed = time.time() - (t_end - dur)
                    remaining = max(0, dur - elapsed)
                    print(f"\r      [{elapsed:.0f} 秒 / {dur:.0f} 秒] 已记录按键={event_count}  ",
                          end="", flush=True)
                    time.sleep(0.02)
                print()  # newline after progress
            else:
                t_end = time.time() + dur
                while time.time() < t_end:
                    elapsed = time.time() - (t_end - dur)
                    remaining = max(0, dur - elapsed)
                    print(f"\r      [{elapsed:.0f} 秒 / {dur:.0f} 秒]  ",
                          end="", flush=True)
                    time.sleep(0.2)
                print()

            block_end_ns = time.perf_counter_ns()

            # Write activity log row
            prompts_str = json.dumps(seg.get("prompts", []))
            alw.writerow([
                block_start_ns, block_end_ns, act,
                label, typing_style, prompts_str,
            ])

        stop.set()
        drain_t.join(timeout=2.0)

        # Final drain
        samples = sensor.drain()
        if samples:
            write_sensor_rows(sw, samples)
            sample_count += len(samples)
        events = keyboard.drain()
        if events:
            write_event_rows(ew, events, participant, session_id)
            event_count += len(events)

        sf.flush(); sf.close()
        ef.flush(); ef.close()
        alf.flush(); alf.close()

        with open(protocol_path, "w") as f:
            json.dump({
                "session_id": session_id,
                "trial_idx": trial_idx,
                "participant": participant,
                "mode": mode_tag,
                "n_trials": int(n_trials),
                "n_passwords": int(n_passwords),
                "duration_jitter_pct": duration_jitter_pct,
                "password_length": int(password_length),
                "output_dir": os.path.abspath(output_dir),
                "spu_report_interval_us": int(spu_report_interval_us),
                "spu_device_rate_control": bool(spu_device_rate_control),
                "protocol": protocol,
                "sample_count": sample_count,
                "event_count": event_count,
            }, f, indent=2)

        print(f"    ✓ 已保存 {sample_count:,} 条传感器样本，{event_count} 条按键事件")

    sensor.stop()
    keyboard.stop()
    print(f"\n  ✓ 共完成 {n_trials} 轮采集。")
    print(f"  输出目录：{output_dir}\n")


def run_custom_structured_mode(
    n_trials: int,
    output_dir: str,
    participant: str,
    seed: int,
    mode_tag: str,
    duration_jitter_pct: float,
    protocol_generator,
    password_length: int = 8,
    spu_report_interval_us: int = 5000,
    spu_device_rate_control: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)
    tag = timestamp_tag()

    sensor = SensorReader(
        spu_report_interval_us=int(spu_report_interval_us),
        spu_device_rate_control=bool(spu_device_rate_control),
    )
    keyboard = KeyboardListener()
    sensor.start()
    keyboard.start()

    print(f"\n{'='*60}")
    print(f"  structured {mode_tag} collection")
    print(f"  轮数：      {n_trials}")
    print(f"  时长抖动：  ±{duration_jitter_pct * 100:.0f}%")
    print(f"  输出目录：  {output_dir}")
    print(f"  SPU间隔：   {int(spu_report_interval_us)} us")
    print(f"  Device ctl: {bool(spu_device_rate_control)}")
    print(f"{'='*60}\n")

    print("  传感器预热中（3 秒）...")
    time.sleep(3)
    sensor.drain()
    keyboard.drain()
    used_prompts = _load_existing_password_prompts()

    for trial_idx in range(n_trials):
        trial_seed = seed + trial_idx
        protocol = protocol_generator(
            seed=trial_seed,
            duration_jitter_pct=duration_jitter_pct,
            used_prompts=used_prompts,
        )
        for seg in protocol:
            if seg.get("typing_style") == "password":
                used_prompts.update(seg.get("prompts", []) or [])

        session_id = f"{participant}_{mode_tag}_trial{trial_idx:03d}_{tag}"
        sensor_path = os.path.join(output_dir, f"{session_id}_sensor.csv")
        events_path = os.path.join(output_dir, f"{session_id}_events.csv")
        activity_log_path = os.path.join(output_dir, f"{session_id}_activity_log.csv")
        protocol_path = os.path.join(output_dir, f"{session_id}_protocol.json")

        total_dur = sum(s["duration_s"] for s in protocol)
        print(f"\n  ══ 第 {trial_idx+1}/{n_trials} 轮（约 {total_dur:.0f} 秒）══")

        sf, sw = open_sensor_csv(sensor_path)
        ef, ew = open_events_csv(events_path)

        alf = open(activity_log_path, "w", newline="")
        alw = csv.writer(alf)
        alw.writerow([
            "start_time_ns", "end_time_ns", "activity",
            "label", "typing_style", "prompts",
        ])

        sample_count = 0
        event_count = 0
        stop = threading.Event()

        def drain_fn():
            nonlocal sample_count
            while not stop.is_set():
                samples = sensor.drain()
                if samples:
                    write_sensor_rows(sw, samples)
                    sample_count += len(samples)
                time.sleep(0.05)

        drain_t = threading.Thread(target=drain_fn, daemon=True)
        drain_t.start()

        input(f"  按回车开始第 {trial_idx+1} 轮 →")
        keyboard.drain()

        for seg in protocol:
            act = seg["activity"]
            dur = seg["duration_s"]
            label = seg["label"]
            typing_style = seg.get("typing_style", "")
            label_zh = DISPLAY_LABELS_ZH.get(label, label)
            typing_style_zh = TYPING_STYLE_ZH.get(typing_style, typing_style)

            if act == "keyboard":
                prompt_text = seg.get("prompt_instructions", "")
                print(f"    🔵 [{label_zh}]（无硬性时限）{typing_style_zh}")
                print("       提醒：请保持英文小写输入，Caps Lock 关闭。")
                if prompt_text:
                    for line in prompt_text.split("\n"):
                        print(f"       {line}")
            else:
                print(f"    🟡 [{label_zh}]（{dur:.0f} 秒）")
                prompt_text = seg.get("prompt_instructions", "")
                if prompt_text:
                    for line in prompt_text.split("\n"):
                        print(f"       {line}")
                elif label == "idle_2":
                    print("       提醒：下一阶段是 password，请在这段 idle 内切到英文小写。")

            block_start_ns = time.perf_counter_ns()
            block_start_wall = time.time()

            if act == "keyboard" and typing_style == "password":
                # Start each keyboard stage with a clean buffer so the current
                # password phase is not polluted by stray keys from earlier
                # free-typing or prompt interactions.
                keyboard.drain()
                target_enters = len(seg.get("prompts", [])) or 1
                enter_count = 0
                while enter_count < target_enters:
                    events = keyboard.drain()
                    if events:
                        write_event_rows(ew, events, participant, session_id)
                        event_count += len(events)
                        for ev in events:
                            if ev.event_type == "press" and ev.key in {"enter", "return"}:
                                enter_count += 1
                    elapsed = time.time() - block_start_wall
                    print(
                        f"\r      [已完成 {enter_count}/{target_enters} 条] "
                        f"已记录按键={event_count}  已耗时={elapsed:.0f} 秒  ",
                        end="",
                        flush=True,
                    )
                    time.sleep(0.02)
                print()
            elif act == "keyboard":
                # Mirror the legacy structured collector: non-password typing
                # phases must still continuously drain and log keyboard events,
                # otherwise they accumulate and get misattributed to the next
                # password phase.
                keyboard.drain()
                t_end = time.time() + dur
                while time.time() < t_end:
                    events = keyboard.drain()
                    if events:
                        write_event_rows(ew, events, participant, session_id)
                        event_count += len(events)
                    elapsed = time.time() - (t_end - dur)
                    print(f"\r      [{elapsed:.0f} 秒 / {dur:.0f} 秒] 已记录按键={event_count}  ",
                          end="", flush=True)
                    time.sleep(0.02)
                print()
            else:
                t_end = time.time() + dur
                while time.time() < t_end:
                    elapsed = time.time() - (t_end - dur)
                    print(f"\r      [{elapsed:.0f} 秒 / {dur:.0f} 秒]  ", end="", flush=True)
                    time.sleep(0.2)
                print()

            block_end_ns = time.perf_counter_ns()
            prompts_str = json.dumps(seg.get("prompts", []))
            alw.writerow([
                block_start_ns, block_end_ns, act,
                label, typing_style, prompts_str,
            ])

        stop.set()
        drain_t.join(timeout=2.0)

        samples = sensor.drain()
        if samples:
            write_sensor_rows(sw, samples)
            sample_count += len(samples)
        events = keyboard.drain()
        if events:
            write_event_rows(ew, events, participant, session_id)
            event_count += len(events)

        sf.flush(); sf.close()
        ef.flush(); ef.close()
        alf.flush(); alf.close()

        with open(protocol_path, "w") as f:
            json.dump({
                "session_id": session_id,
                "trial_idx": trial_idx,
                "participant": participant,
                "mode": mode_tag,
                "n_trials": int(n_trials),
                "duration_jitter_pct": duration_jitter_pct,
                "password_length": int(password_length),
                "output_dir": os.path.abspath(output_dir),
                "spu_report_interval_us": int(spu_report_interval_us),
                "spu_device_rate_control": bool(spu_device_rate_control),
                "protocol": protocol,
                "sample_count": sample_count,
                "event_count": event_count,
            }, f, indent=2)

        print(f"    ✓ 已保存 {sample_count:,} 条传感器样本，{event_count} 条按键事件")

    sensor.stop()
    keyboard.stop()
    print(f"\n  ✓ 共完成 {n_trials} 轮采集。")
    print(f"  输出目录：{output_dir}\n")


def run_mixed2_mode(
    n_trials: int = 5,
    n_passwords: int = 5,
    output_dir: str = "data/raw/onset_mixed2",
    participant: str = "p01",
    seed: int = 42,
    password_length: int = 8,
    spu_report_interval_us: int = 5000,
    spu_device_rate_control: bool = False,
):
    return run_structured_mode(
        n_trials=n_trials,
        n_passwords=n_passwords,
        output_dir=output_dir,
        participant=participant,
        seed=seed,
        mode_tag="mixed2",
        duration_jitter_pct=0.0,
        password_length=password_length,
        spu_report_interval_us=spu_report_interval_us,
        spu_device_rate_control=spu_device_rate_control,
    )


def run_mixed_training_mode(
    n_trials: int = 20,
    n_passwords: int = 5,
    output_dir: str = "data/raw/mixed_training",
    participant: str = "p01",
    seed: int = 42,
    duration_jitter_pct: float = DEFAULT_MIXED_TRAINING_JITTER_PCT,
    password_length: int = 8,
    spu_report_interval_us: int = 5000,
    spu_device_rate_control: bool = False,
):
    return run_structured_mode(
        n_trials=n_trials,
        n_passwords=n_passwords,
        output_dir=output_dir,
        participant=participant,
        seed=seed,
        mode_tag="mixed_training",
        duration_jitter_pct=duration_jitter_pct,
        password_length=password_length,
        spu_report_interval_us=spu_report_interval_us,
        spu_device_rate_control=spu_device_rate_control,
    )


def run_mixed_single_training_mode(
    n_trials: int = 3,
    output_dir: str = "data/raw/mixed_single_training",
    participant: str = "p01",
    seed: int = 42,
    duration_jitter_pct: float = DEFAULT_MIXED_TRAINING_JITTER_PCT,
    password_length: int = 8,
    spu_report_interval_us: int = 5000,
    spu_device_rate_control: bool = False,
):
    return run_custom_structured_mode(
        n_trials=n_trials,
        output_dir=output_dir,
        participant=participant,
        seed=seed,
        mode_tag="mixed_single_training",
        duration_jitter_pct=duration_jitter_pct,
        protocol_generator=lambda **kwargs: generate_single_password_protocol(
            password_length=password_length, **kwargs
        ),
        password_length=password_length,
        spu_report_interval_us=spu_report_interval_us,
        spu_device_rate_control=spu_device_rate_control,
    )


def run_mixed_retry_training_mode(
    n_trials: int = 3,
    output_dir: str = "data/raw/mixed_retry_training",
    participant: str = "p01",
    seed: int = 42,
    duration_jitter_pct: float = DEFAULT_MIXED_TRAINING_JITTER_PCT,
    password_length: int = 8,
    spu_report_interval_us: int = 5000,
    spu_device_rate_control: bool = False,
):
    return run_custom_structured_mode(
        n_trials=n_trials,
        output_dir=output_dir,
        participant=participant,
        seed=seed,
        mode_tag="mixed_retry_training",
        duration_jitter_pct=duration_jitter_pct,
        protocol_generator=lambda **kwargs: generate_retry_password_protocol(
            password_length=password_length, **kwargs
        ),
        password_length=password_length,
        spu_report_interval_us=spu_report_interval_us,
        spu_device_rate_control=spu_device_rate_control,
    )


# ── Activity log loading helper ──────────────────────────────

def load_activity_log(path: str) -> list[dict]:
    """
    Load an activity_log.csv and return list of dicts with fields:
      start_time_ns, end_time_ns, activity, label, typing_style, prompts

    Works for both mixed and mixed2 formats.
    """
    segments = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seg = {
                "start_time_ns": int(row["start_time_ns"]),
                "end_time_ns": int(row["end_time_ns"]),
                "activity": row.get("activity", ""),
            }
            # mixed2 format has extra columns
            seg["label"] = row.get("label", row.get("activity", ""))
            seg["typing_style"] = row.get("typing_style", "")

            prompts_raw = row.get("prompts", row.get("prompt", ""))
            if prompts_raw and prompts_raw.startswith("["):
                try:
                    seg["prompts"] = json.loads(prompts_raw)
                except json.JSONDecodeError:
                    seg["prompts"] = [prompts_raw] if prompts_raw else []
            else:
                seg["prompts"] = [prompts_raw] if prompts_raw else []

            segments.append(seg)
    return segments


def get_keyboard_episodes_from_activity_log(
    segments: list[dict],
) -> list[dict]:
    """
    Extract keyboard episodes from activity log segments.
    Returns list of dicts with: start_s, end_s, label, typing_style, prompts
    """
    episodes = []
    for seg in segments:
        if seg["activity"] == "keyboard":
            episodes.append({
                "start_s": seg["start_time_ns"] / 1e9,
                "end_s": seg["end_time_ns"] / 1e9,
                "label": seg["label"],
                "typing_style": seg.get("typing_style", ""),
                "prompts": seg.get("prompts", []),
            })
    return episodes


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Onset detection data collector")
    parser.add_argument(
        "--project-root", default="",
        help="Project root directory for resolving imports and default output paths.",
    )
    parser.add_argument(
        "--mode", choices=["negative", "mixed", "mixed2", "mixed_training", "mixed_single_training", "mixed_retry_training"], required=True,
        help="negative: single-activity. mixed: random interleaved. "
             "mixed2: structured hold-out protocol with typing_1/typing_2. "
             "mixed_training: same skeleton, separate training split with light duration jitter. "
             "mixed_single_training: realistic single-password stream. "
             "mixed_retry_training: single-password + 20s free activity + second password."
    )
    parser.add_argument("--activity", choices=NEGATIVE_ACTIVITIES, default="idle",
                        help="Activity type for negative mode")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration in seconds for negative mode (default: 60)")
    parser.add_argument("--n-segments", type=int, default=15,
                        help="Number of mixed segments (default: 15)")
    parser.add_argument("--segment-sec", type=float, default=30.0,
                        help="Approx duration per mixed segment (default: 30)")
    parser.add_argument("--n-trials", type=int, default=1,
                        help="Number of structured trials to record this run (default: 1)")
    parser.add_argument("--n-passwords", type=int, default=5,
                        help="Passwords per mixed2 trial (default: 5)")
    parser.add_argument("--password-length", type=int, default=8,
                        help="Password length for structured password stages (default: 8)")
    parser.add_argument("--duration-jitter-pct", type=float, default=None,
                        help="Optional per-segment duration jitter for structured modes, "
                             "e.g. 0.15 means +/-15%%")
    parser.add_argument("--output-dir", default="",
                        help="Override output directory")
    parser.add_argument("--participant", default="p01")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed. If omitted, a fresh seed is generated per run.")
    parser.add_argument("--spu-report-interval-us", type=int, default=5000,
                        help="Direct SPU ReportInterval in microseconds (use 1250 for ~800Hz).")
    parser.add_argument("--spu-device-rate-control", action="store_true",
                        help="Also set ReportInterval on opened IOHIDDeviceRef. Required for non-root ~800Hz.")
    args = parser.parse_args()

    if args.seed is None:
        args.seed = secrets.randbelow(1_000_000_000)

    if args.mode == "negative":
        out_dir = args.output_dir or "data/raw/onset_negative"
        if args.project_root:
            _setup_imports(args.project_root)
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.path.abspath(args.project_root), out_dir)
        run_negative_mode(
            activity=args.activity,
            duration_sec=args.duration,
            output_dir=out_dir,
            participant=args.participant,
            spu_report_interval_us=args.spu_report_interval_us,
            spu_device_rate_control=args.spu_device_rate_control,
        )
    elif args.mode == "mixed":
        out_dir = args.output_dir or "data/raw/onset_mixed"
        if args.project_root:
            _setup_imports(args.project_root)
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.path.abspath(args.project_root), out_dir)
        run_mixed_mode(
            n_segments=args.n_segments,
            segment_sec=args.segment_sec,
            output_dir=out_dir,
            participant=args.participant,
            seed=args.seed,
        )
    elif args.mode == "mixed2":
        out_dir = args.output_dir or "data/raw/onset_mixed2"
        if args.project_root:
            _setup_imports(args.project_root)
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.path.abspath(args.project_root), out_dir)
        run_mixed2_mode(
            n_trials=args.n_trials,
            n_passwords=args.n_passwords,
            output_dir=out_dir,
            participant=args.participant,
            seed=args.seed,
            password_length=args.password_length,
            spu_report_interval_us=args.spu_report_interval_us,
            spu_device_rate_control=args.spu_device_rate_control,
        )
    elif args.mode == "mixed_training":
        out_dir = args.output_dir or "data/raw/mixed_training"
        if args.project_root:
            _setup_imports(args.project_root)
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.path.abspath(args.project_root), out_dir)
        jitter = args.duration_jitter_pct
        if jitter is None:
            jitter = DEFAULT_MIXED_TRAINING_JITTER_PCT
        run_mixed_training_mode(
            n_trials=args.n_trials,
            n_passwords=args.n_passwords,
            output_dir=out_dir,
            participant=args.participant,
            seed=args.seed,
            duration_jitter_pct=jitter,
            password_length=args.password_length,
            spu_report_interval_us=args.spu_report_interval_us,
            spu_device_rate_control=args.spu_device_rate_control,
        )
    elif args.mode == "mixed_single_training":
        if args.output_dir:
            out_dir = args.output_dir
        elif int(args.password_length) == 8:
            out_dir = "data/raw/mixed_single_training"
        else:
            out_dir = f"data/raw/mixed_single_len{int(args.password_length)}"
        if args.project_root:
            _setup_imports(args.project_root)
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.path.abspath(args.project_root), out_dir)
        jitter = args.duration_jitter_pct
        if jitter is None:
            jitter = DEFAULT_MIXED_TRAINING_JITTER_PCT
        run_mixed_single_training_mode(
            n_trials=args.n_trials,
            output_dir=out_dir,
            participant=args.participant,
            seed=args.seed,
            duration_jitter_pct=jitter,
            password_length=args.password_length,
            spu_report_interval_us=args.spu_report_interval_us,
            spu_device_rate_control=args.spu_device_rate_control,
        )
    elif args.mode == "mixed_retry_training":
        if args.output_dir:
            out_dir = args.output_dir
        elif int(args.password_length) == 8:
            out_dir = "data/raw/mixed_retry_training"
        else:
            out_dir = f"data/raw/mixed_retry_len{int(args.password_length)}"
        if args.project_root:
            _setup_imports(args.project_root)
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.path.abspath(args.project_root), out_dir)
        jitter = args.duration_jitter_pct
        if jitter is None:
            jitter = DEFAULT_MIXED_TRAINING_JITTER_PCT
        run_mixed_retry_training_mode(
            n_trials=args.n_trials,
            output_dir=out_dir,
            participant=args.participant,
            seed=args.seed,
            duration_jitter_pct=jitter,
            password_length=args.password_length,
            spu_report_interval_us=args.spu_report_interval_us,
            spu_device_rate_control=args.spu_device_rate_control,
        )


if __name__ == "__main__":
    main()
