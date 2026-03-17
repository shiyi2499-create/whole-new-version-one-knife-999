"""
Onset Data Collector
====================
Collect negative-sample and mixed-stream sessions for onset detection.

Three modes:
  1. negative  – Record a single activity type (idle, trackpad_move, etc.)
  2. mixed     – Record a scripted sequence of random interleaved activities
  3. mixed2    – Record the structured ~3-minute protocol:
                 idle → trackpad_move → typing_1 (free) → trackpad_click →
                 idle → typing_2 (password) → shake
                 with explicit segment boundaries and labels

Reuses the existing sensor_reader / spu_backend / keyboard_listener stack.

Run:
  python3 onset_collector.py --mode negative --activity idle --duration 60
  python3 onset_collector.py --mode mixed --n-segments 15 --segment-sec 30
  python3 onset_collector.py --mode mixed2 --n-trials 5

Data is saved under:
  data/raw/onset_negative/<activity>/    (for negative mode)
  data/raw/onset_mixed/                  (for mixed mode)
  data/raw/onset_mixed2/                 (for mixed2 mode)
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
    "trackpad_click",
    "shake",
    "desk_bump",
    "keyboard",
]

NEGATIVE_ACTIVITIES = [a for a in ACTIVITIES if a != "keyboard"]


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


# ── Negative mode ────────────────────────────────────────────

def run_negative_mode(
    activity: str,
    duration_sec: float,
    output_dir: str,
    participant: str = "p01",
    gate_rate_hz: float = 150.0,
    precheck_sec: float = 3.0,
):
    """Record a single-activity negative session."""
    assert activity in NEGATIVE_ACTIVITIES, f"Unknown activity: {activity}"

    out_dir = os.path.join(output_dir, activity)
    os.makedirs(out_dir, exist_ok=True)

    tag = timestamp_tag()
    session_id = f"{participant}_onset_neg_{activity}_{tag}"
    sensor_path = os.path.join(out_dir, f"{session_id}_sensor.csv")
    meta_path = os.path.join(out_dir, f"{session_id}_meta.json")

    sensor = SensorReader()
    sensor.start()

    print(f"\n{'='*60}")
    print(f"  ONSET NEGATIVE COLLECTION")
    print(f"  Activity:  {activity}")
    print(f"  Duration:  {duration_sec:.0f}s")
    print(f"  Output:    {sensor_path}")
    print(f"{'='*60}\n")

    print(f"  Warming up sensor ({precheck_sec:.0f}s)...")
    time.sleep(precheck_sec)
    sensor.drain()

    sf, sw = open_sensor_csv(sensor_path)
    sample_count = 0

    print(f"  🔴 NOW: perform [{activity}] for {duration_sec:.0f}s")
    print(f"     (press Ctrl+C to stop early)\n")

    t_start = time.time()
    try:
        while time.time() - t_start < duration_sec:
            samples = sensor.drain()
            if samples:
                write_sensor_rows(sw, samples)
                sample_count += len(samples)
            elapsed = time.time() - t_start
            remaining = max(0, duration_sec - elapsed)
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

    sf.flush()
    sf.close()
    sensor.stop()

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
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

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

DISPLAY_LABELS_ZH = {
    "idle_1": "静止阶段 1",
    "idle_2": "密码前静止阶段",
    "trackpad_move_1": "触控板滑动阶段",
    "trackpad_click_1": "触控板点击阶段",
    "shake_1": "整机晃动收尾阶段",
    "typing_1": "自由敲击阶段",
    "typing_2": "密码输入阶段",
}

TYPING_STYLE_ZH = {
    "free": "自由敲击",
    "password": "密码输入",
}


def generate_mixed2_protocol(
    n_passwords: int = 5,
    seed: int = 42,
) -> list[dict]:
    """
    Generate the structured ~3-minute mixed-stream protocol.
    The typing_2 segment gets n_passwords random 8-char password prompts.
    """
    rng = random.Random(seed)
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"

    protocol = []
    for seg in DEFAULT_MIXED2_PROTOCOL:
        entry = dict(seg)
        if entry.get("typing_style") == "password":
            passwords = ["".join(rng.choices(chars, k=8)) for _ in range(n_passwords)]
            entry["prompts"] = passwords
            entry["prompt_instructions"] = (
                f"请慢速、仔细地输入下面每一条密码；每输入完一条后按一次 Enter：\n"
                + "\n".join(f"  {i+1}. {pw}" for i, pw in enumerate(passwords))
            )
        protocol.append(entry)

    return protocol


def run_mixed2_mode(
    n_trials: int = 5,
    n_passwords: int = 5,
    output_dir: str = "data/raw/onset_mixed2",
    participant: str = "p01",
    seed: int = 42,
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

    sensor = SensorReader()
    keyboard = KeyboardListener()
    sensor.start()
    keyboard.start()

    print(f"\n{'='*60}")
    print(f"  结构化约 3 分钟混合流采集")
    print(f"  轮数：      {n_trials}")
    print(f"  每轮密码数：{n_passwords}")
    print(f"  输出目录：  {output_dir}")
    print(f"{'='*60}\n")

    print("  传感器预热中（3 秒）...")
    time.sleep(3)
    sensor.drain()
    keyboard.drain()

    for trial_idx in range(n_trials):
        trial_seed = seed + trial_idx
        protocol = generate_mixed2_protocol(n_passwords=n_passwords, seed=trial_seed)

        session_id = f"{participant}_mixed2_trial{trial_idx:03d}_{tag}"
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
                print(f"    🔵 [{label_zh}]（{dur:.0f} 秒）{typing_style_zh}")
                if prompt_text:
                    for line in prompt_text.split("\n"):
                        print(f"       {line}")
            else:
                print(f"    🟡 [{label_zh}]（{dur:.0f} 秒）")

            block_start_ns = time.perf_counter_ns()

            if act == "keyboard":
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
                "protocol": protocol,
                "sample_count": sample_count,
                "event_count": event_count,
            }, f, indent=2)

        print(f"    ✓ 已保存 {sample_count:,} 条传感器样本，{event_count} 条按键事件")

    sensor.stop()
    keyboard.stop()
    print(f"\n  ✓ 共完成 {n_trials} 轮采集。")
    print(f"  输出目录：{output_dir}\n")


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
        "--mode", choices=["negative", "mixed", "mixed2"], required=True,
        help="negative: single-activity. mixed: random interleaved. "
             "mixed2: structured 2-min protocol with typing_1/typing_2."
    )
    parser.add_argument("--activity", choices=NEGATIVE_ACTIVITIES, default="idle",
                        help="Activity type for negative mode")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration in seconds for negative mode (default: 60)")
    parser.add_argument("--n-segments", type=int, default=15,
                        help="Number of mixed segments (default: 15)")
    parser.add_argument("--segment-sec", type=float, default=30.0,
                        help="Approx duration per mixed segment (default: 30)")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of mixed2 trials (default: 5)")
    parser.add_argument("--n-passwords", type=int, default=5,
                        help="Passwords per mixed2 trial (default: 5)")
    parser.add_argument("--output-dir", default="",
                        help="Override output directory")
    parser.add_argument("--participant", default="p01")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
        )


if __name__ == "__main__":
    main()
