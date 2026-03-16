"""
Onset Data Collector
====================
Collect negative-sample and mixed-stream sessions for onset detection.

Two modes:
  1. negative  – Record a single activity type (idle, trackpad_move, etc.)
                 Outputs sensor.csv only (all-negative, no keyboard labels)
  2. mixed     – Record a scripted sequence of interleaved activities
                 including keyboard typing segments with event labels

Reuses the existing sensor_reader / spu_backend / keyboard_listener stack.

Run:
  python3 onset_collector.py --mode negative --activity idle --duration 60
  python3 onset_collector.py --mode negative --activity trackpad_move --duration 60
  python3 onset_collector.py --mode negative --activity trackpad_click --duration 60
  python3 onset_collector.py --mode negative --activity shake --duration 45
  python3 onset_collector.py --mode negative --activity desk_bump --duration 45

  python3 onset_collector.py --mode mixed --n-segments 15 --segment-sec 30

Data is saved under:
  data/raw/onset_negative/<activity>/    (for negative mode)
  data/raw/onset_mixed/                  (for mixed mode)
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

# Import from the main project.  The project root is resolved at runtime
# via --project-root or best-effort detection (parent of onset_detection/).

_PROJECT_ROOT = None


def _setup_imports(project_root: str = ""):
    """Add project root to sys.path for sensor_reader / keyboard_listener."""
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

    # Pre-check
    print(f"  Warming up sensor ({precheck_sec:.0f}s)...")
    time.sleep(precheck_sec)
    sensor.drain()  # discard warm-up samples

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

    # Final drain
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


# ── Mixed mode ───────────────────────────────────────────────

def generate_mixed_script(
    n_segments: int = 15,
    segment_sec: float = 30.0,
    seed: int = 42,
) -> list[dict]:
    """
    Generate a random script of mixed-stream segments.
    Each segment is ~segment_sec long with 4-6 activity blocks.
    """
    rng = random.Random(seed)

    # Typing prompts: random 8-char strings like the password protocol
    chars = "abcdefghijklmnopqrstuvwxyz0123456789"
    def random_password():
        return "".join(rng.choices(chars, k=8))

    scripts = []
    for i in range(n_segments):
        # Build a random sequence of activity blocks for this segment
        n_blocks = rng.randint(4, 6)
        blocks = []
        total = 0.0

        for j in range(n_blocks):
            # Pick activity
            if j == 0:
                activity = "idle"  # always start with idle for baseline
            else:
                activity = rng.choice(ACTIVITIES)

            # Duration
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

        # Activity log
        alf = open(activity_log_path, "w", newline="")
        alw = csv.writer(alf)
        alw.writerow(["start_time_ns", "end_time_ns", "activity", "prompt"])

        sample_count = 0
        event_count = 0
        stop = threading.Event()

        # Drain thread
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
        keyboard.drain()  # clear stale events

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
                # Collect keyboard events during typing
                t_end = time.time() + dur
                while time.time() < t_end:
                    events = keyboard.drain()
                    if events:
                        write_event_rows(ew, events, participant, session_id)
                        event_count += len(events)
                    time.sleep(0.02)
            else:
                # Non-keyboard activity: just wait
                time.sleep(dur)

            block_end_ns = time.perf_counter_ns()
            alw.writerow([block_start_ns, block_end_ns, act, prompt or ""])

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

        with open(script_path, "w") as f:
            json.dump(seg, f, indent=2)

        print(f"    ✓ {sample_count:,} sensor + {event_count} key events")

    sensor.stop()
    keyboard.stop()
    print(f"\n  ✓ All {n_segments} segments recorded.")
    print(f"  Output → {output_dir}\n")


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Onset detection data collector")
    parser.add_argument(
        "--project-root", default="",
        help="Project root directory for resolving imports and default output paths.",
    )
    parser.add_argument(
        "--mode", choices=["negative", "mixed"], required=True,
        help="negative: single-activity negative samples. mixed: interleaved evaluation streams."
    )
    parser.add_argument("--activity", choices=NEGATIVE_ACTIVITIES, default="idle",
                        help="Activity type for negative mode")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration in seconds for negative mode (default: 60)")
    parser.add_argument("--n-segments", type=int, default=15,
                        help="Number of mixed segments (default: 15)")
    parser.add_argument("--segment-sec", type=float, default=30.0,
                        help="Approx duration per mixed segment (default: 30)")
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


if __name__ == "__main__":
    main()
