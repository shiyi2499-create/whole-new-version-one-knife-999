"""
Keystroke Vibration Data Collector - Main Orchestrator
======================================================
Coordinates sensor reading, keyboard listening, and rate monitoring.
Supports two modes:
  1. single_key  - guided per-key repeated pressing
  2. free_type   - natural typing with auto key logging

FIXED v2:
  - Background drain thread runs continuously in ALL modes
    (no more rate drops during pauses between keys)
  - Rate monitor uses simple counter-based calculation
  - Sensor reader handles None samples gracefully

FIXED v3:
  - free_type mode no longer has two keyboard consumers.
    Background thread drains sensor only; prompt loop exclusively drains keyboard.
  - Added a short free_type keyboard self-check ("abc 123") before prompts.

Run with:  sudo python3 collector.py --mode single_key
           sudo python3 collector.py --mode free_type
"""

import os
import csv
import sys
import time
import argparse
import threading
from statistics import median
from datetime import datetime
from typing import Optional

from config import CollectorConfig
from sensor_reader import SensorReader, SensorSample
from keyboard_listener import KeyboardListener, KeyEvent
from rate_monitor import RateMonitor


class DataCollector:
    """
    Main data collection coordinator.
    """

    def __init__(self, config: CollectorConfig, mode: str, group: int = 0,
                 free_type_part: int = 0, free_type_parts_total: int = 16,
                 single_rate_gate_hz: float = 190.0,
                 free_rate_gate_hz: float = 150.0,
                 precheck_sec: float = 5.0):
        self.cfg = config
        self.mode = mode
        self.group = group
        self._free_type_part = free_type_part
        self._free_type_parts_total = max(1, int(free_type_parts_total))
        self._single_rate_gate_hz = float(single_rate_gate_hz)
        self._free_rate_gate_hz = float(free_rate_gate_hz)
        self._precheck_sec = float(precheck_sec)
        # Avoid keyboard drain race in free_type:
        # single_key  -> background thread drains keyboard
        # free_type   -> prompt loop drains keyboard exclusively
        self._drain_keyboard_in_background = (mode == "single_key")

        # Components
        self.sensor = SensorReader()
        self.keyboard = KeyboardListener()
        self.rate_monitor = RateMonitor(
            min_rate_hz=config.MIN_ACCEPTABLE_RATE_HZ,
            check_interval_sec=config.RATE_CHECK_INTERVAL_SEC,
            on_rate_drop=self._on_rate_drop,
        )

        # Session info
        self.session_prefix = config.session_prefix(mode, group, free_type_part)
        self.sensor_csv_path = os.path.join(
            config.RAW_DIR, f"{self.session_prefix}_sensor.csv"
        )
        self.events_csv_path = os.path.join(
            config.RAW_DIR, f"{self.session_prefix}_events.csv"
        )
        self.meta_path = os.path.join(
            config.RAW_DIR, f"{self.session_prefix}_meta.txt"
        )
        self.prompts_log_path = self.events_csv_path.replace("_events.csv", "_prompts.csv")
        self.attempts_log_path = self.events_csv_path.replace("_events.csv", "_attempts.csv")

        # State
        self._stop_event = threading.Event()
        self._rate_drop_detected = False
        self._session_valid = True
        self._discard_session = False
        self._discard_reason = ""

        # Counters
        self._sensor_count = 0
        self._event_count = 0

        # CSV writers
        self._sensor_file = None
        self._sensor_writer = None
        self._events_file = None
        self._events_writer = None
        self._csv_lock = threading.Lock()

        # For single_key mode: track per-key press count from drain thread
        self._current_target_key: Optional[str] = None
        self._target_press_count = 0
        self._target_press_lock = threading.Lock()

    # ── Rate drop callback ───────────────────────────────────

    def _on_rate_drop(self, rate: float):
        self._rate_drop_detected = True
        self._stop_event.set()
        print(
            f"\n{'='*60}\n"
            f"  ⚠️  SAMPLING RATE ALERT!\n"
            f"  Rate dropped to {rate:.1f} Hz (minimum: {self.cfg.MIN_ACCEPTABLE_RATE_HZ} Hz)\n"
            f"  Recording stopped. Data collected so far has been SAVED.\n"
            f"{'='*60}"
        )

    # ── CSV I/O ──────────────────────────────────────────────

    def _open_csv_files(self):
        self._sensor_file = open(self.sensor_csv_path, "w", newline="")
        self._sensor_writer = csv.writer(self._sensor_file)
        self._sensor_writer.writerow([
            "timestamp_ns", "accel_x", "accel_y", "accel_z",
            "gyro_x", "gyro_y", "gyro_z"
        ])

        self._events_file = open(self.events_csv_path, "w", newline="")
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow([
            "timestamp_ns", "key", "event_type", "participant_id", "session_id"
        ])

    def _close_csv_files(self):
        if self._sensor_file:
            self._sensor_file.flush()
            self._sensor_file.close()
        if self._events_file:
            self._events_file.flush()
            self._events_file.close()

    def _delete_session_files(self):
        """Delete session artifacts for aborted runs (e.g., failed frequency gate)."""
        for p in [
            self.sensor_csv_path,
            self.events_csv_path,
            self.meta_path,
            self.prompts_log_path,
            self.attempts_log_path,
        ]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError as e:
                    print(f"  ⚠ Failed to delete {p}: {e}")

    def _recent_rate_stats(self, recent_sec: float = 5.0) -> dict:
        now = time.time()
        rates = [r for t, r in self.rate_monitor.rate_history if (now - t) <= recent_sec and r > 0]
        if not rates and self.rate_monitor.current_rate > 0:
            rates = [self.rate_monitor.current_rate]
        if not rates:
            return {"n": 0, "avg": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "n": len(rates),
            "avg": sum(rates) / len(rates),
            "median": float(median(rates)),
            "min": min(rates),
            "max": max(rates),
        }

    def _rate_precheck(self, gate_hz: float, label: str) -> bool:
        """
        Check recent sampling-rate stability before formal collection starts.
        If check fails, mark session as discard-only and stop early.
        """
        if gate_hz <= 0:
            return True

        print(f"  Running {label} rate precheck for {self._precheck_sec:.1f}s (gate: {gate_hz:.1f} Hz)...")
        t_end = time.time() + self._precheck_sec
        while time.time() < t_end and not self._stop_event.is_set():
            time.sleep(0.2)

        stats = self._recent_rate_stats(recent_sec=max(1.0, self._precheck_sec))
        print(
            f"    Precheck rates: median={stats['median']:.1f}Hz  "
            f"avg={stats['avg']:.1f}Hz  min={stats['min']:.1f}Hz  max={stats['max']:.1f}Hz"
        )

        # Use median to avoid occasional bursty high-frequency outliers.
        if stats["median"] < gate_hz:
            self._discard_session = True
            self._discard_reason = (
                f"{label} rate gate failed: median {stats['median']:.1f}Hz < {gate_hz:.1f}Hz"
            )
            print(f"  ⚠ {self._discard_reason}")
            print("  This session will be discarded and no files will be kept.\n")
            return False
        return True

    def _write_sensor_samples(self, samples: list[SensorSample]):
        for s in samples:
            self._sensor_writer.writerow([
                s.timestamp_ns,
                f"{s.accel_x:.8f}", f"{s.accel_y:.8f}", f"{s.accel_z:.8f}",
                f"{s.gyro_x:.6f}", f"{s.gyro_y:.6f}", f"{s.gyro_z:.6f}",
            ])
        self._sensor_count += len(samples)

    def _write_key_events(self, events: list[KeyEvent]):
        for e in events:
            self._events_writer.writerow([
                e.timestamp_ns, e.key, e.event_type,
                self.cfg.PARTICIPANT_ID, self.session_prefix,
            ])
        self._event_count += len(events)

    # ── Background drain thread (runs in ALL modes) ──────────

    def _drain_thread_fn(self):
        """
        Continuously drains sensor + keyboard buffers to CSV.
        This runs the ENTIRE session, including pauses between keys.
        This ensures the rate monitor always gets ticks.
        """
        flush_counter = 0

        while not self._stop_event.is_set():
            # Drain sensor
            samples = self.sensor.drain()
            if samples:
                with self._csv_lock:
                    self._write_sensor_samples(samples)
                self.rate_monitor.tick(count=len(samples))

            # Drain keyboard events (single_key only; free_type owns keyboard drain)
            if self._drain_keyboard_in_background:
                events = self.keyboard.drain()
                if events:
                    with self._csv_lock:
                        self._write_key_events(events)

                    # Count target key presses for single_key mode
                    if self._current_target_key is not None:
                        with self._target_press_lock:
                            for e in events:
                                if (e.key == self._current_target_key
                                        and e.event_type == "press"):
                                    self._target_press_count += 1

            # Periodic flush
            flush_counter += 1
            if flush_counter >= 20:  # every ~2s
                with self._csv_lock:
                    if self._sensor_file:
                        self._sensor_file.flush()
                    if self._events_file:
                        self._events_file.flush()
                flush_counter = 0

            time.sleep(0.1)

        # Final drain
        samples = self.sensor.drain()
        if samples:
            with self._csv_lock:
                self._write_sensor_samples(samples)
        if self._drain_keyboard_in_background:
            events = self.keyboard.drain()
            if events:
                with self._csv_lock:
                    self._write_key_events(events)

    def _run_free_type_keyboard_self_check(self):
        """
        Quick sanity check before formal prompts.
        Important: this does NOT write keyboard events to CSV, so it won't
        contaminate sentence/event alignment used by downstream preprocessing.
        Also prints current sampling rate; if rate is below threshold, aborts
        this round before prompt collection starts.
        """
        expected = "abc 123"
        print(f"  Keyboard self-check: type exactly '{expected}' then press ENTER.")
        print("  (This check is not recorded to prompts/event files.)")

        # Clear stale events before the check starts
        self.keyboard.drain()

        attempt = 0
        while not self._stop_event.is_set():
            attempt += 1
            typed_buffer = []
            press_keys = []

            while not self._stop_event.is_set():
                events = self.keyboard.drain()
                if events:
                    for e in events:
                        if e.event_type != "press":
                            continue
                        press_keys.append(e.key)
                        if e.key == "enter":
                            typed_text = "".join(typed_buffer).strip()
                            rate_now = self.rate_monitor.current_rate
                            print(f"    Attempt {attempt}: press seq = {press_keys}")
                            print(f"    Attempt {attempt}: typed     = '{typed_text}'")
                            print(f"    Attempt {attempt}: rate      = {rate_now:.1f}Hz")
                            if typed_text == expected:
                                if 0 < rate_now < self.cfg.MIN_ACCEPTABLE_RATE_HZ:
                                    print(
                                        "    ⚠ Keyboard self-check text passed, but sampling rate is too low.\n"
                                        f"      Current: {rate_now:.1f} Hz, minimum: {self.cfg.MIN_ACCEPTABLE_RATE_HZ} Hz\n"
                                        "      This round is terminated early. Please start the next round.\n"
                                    )
                                    self.keyboard.drain()
                                    return False
                                print("    ✓ Keyboard self-check passed.\n")
                                # Drop any trailing release noise before prompts
                                self.keyboard.drain()
                                return True
                            print("    ⚠ Mismatch, please retry the self-check.\n")
                            # Drop trailing events before next attempt
                            self.keyboard.drain()
                            break
                        elif e.key == "backspace":
                            if typed_buffer:
                                typed_buffer.pop()
                        elif e.key == "space":
                            typed_buffer.append(" ")
                        elif len(e.key) == 1:
                            typed_buffer.append(e.key)

                    if press_keys and press_keys[-1] == "enter":
                        # Move to next attempt
                        break

                time.sleep(0.02)

        return False

    # ── Single Key Mode ──────────────────────────────────────

    def _run_single_key_mode(self):
        group_info = ""
        if self.group > 0:
            g = self.cfg.KEY_GROUPS[self.group]
            group_info = f"  Group:     {g['name']}\n"

        print(
            f"\n{'='*60}\n"
            f"  SINGLE KEY MODE\n"
            f"{group_info}"
            f"  Keys:      {' '.join(self.cfg.KEY_LIST)}\n"
            f"  Repeats:   {self.cfg.REPEATS_PER_KEY} per key\n"
            f"  Total:     {len(self.cfg.KEY_LIST)} keys × "
            f"{self.cfg.REPEATS_PER_KEY} = "
            f"{len(self.cfg.KEY_LIST) * self.cfg.REPEATS_PER_KEY} presses\n"
            f"  Ctrl+C to stop early.\n"
            f"{'='*60}\n"
        )

        print("  Warming up sensor (3 seconds)...")
        time.sleep(3)
        if self._stop_event.is_set():
            return

        if not self._rate_precheck(self._single_rate_gate_hz, "single_key"):
            return

        print("  Sensor ready! Let's go.\n")

        for idx, target_key in enumerate(self.cfg.KEY_LIST):
            if self._stop_event.is_set():
                break

            remaining = len(self.cfg.KEY_LIST) - idx
            print(
                f"  [{idx+1}/{len(self.cfg.KEY_LIST)}] "
                f"Press  [ {target_key} ]  × {self.cfg.REPEATS_PER_KEY}  "
                f"(remaining: {remaining})"
            )

            # Reset target key press counter
            with self._target_press_lock:
                self._current_target_key = target_key
                self._target_press_count = 0

            # Wait for user to finish pressing
            last_display = -1
            while not self._stop_event.is_set():
                with self._target_press_lock:
                    count = self._target_press_count

                if count >= self.cfg.REPEATS_PER_KEY:
                    break

                # Update progress bar
                if count != last_display:
                    bar_len = 30
                    filled = int(bar_len * count / self.cfg.REPEATS_PER_KEY)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    rate = self.rate_monitor.current_rate
                    print(
                        f"\r    [{bar}] {count}/{self.cfg.REPEATS_PER_KEY}  "
                        f"Rate: {rate:.0f}Hz  ",
                        end="", flush=True
                    )
                    last_display = count

                time.sleep(0.05)

            # Final bar
            rate = self.rate_monitor.current_rate
            bar = "█" * 30
            with self._target_press_lock:
                final_count = self._target_press_count
            print(
                f"\r    [{bar}] {final_count}/{self.cfg.REPEATS_PER_KEY}  "
                f"Rate: {rate:.0f}Hz  "
            )
            print(f"    ✓ '{target_key}' done!")

            # Clear target so drain thread stops counting for old key
            with self._target_press_lock:
                self._current_target_key = None

            # Pause between keys (drain thread keeps running!)
            if idx < len(self.cfg.KEY_LIST) - 1 and not self._stop_event.is_set():
                print(f"    (pause {self.cfg.PAUSE_BETWEEN_KEYS_SEC}s...)")
                time.sleep(self.cfg.PAUSE_BETWEEN_KEYS_SEC)

        print("\n  All keys completed!")

    # ── Free Type Mode (Guided with validation) ────────────────

    def _run_free_type_mode(self):
        from typing_prompts import PROMPTS

        # Split into configurable parts (default: 16 groups for 208 prompts)
        n_parts = self._free_type_parts_total
        part_size = (len(PROMPTS) + n_parts - 1) // n_parts  # ceiling division
        parts = {}
        for i in range(1, n_parts + 1):
            start = (i - 1) * part_size
            end = min(i * part_size, len(PROMPTS))
            if start < len(PROMPTS):
                parts[i] = PROMPTS[start:end]

        part = self._free_type_part
        if part not in parts:
            prompts = PROMPTS
            part_label = "ALL"
        else:
            prompts = parts[part]
            part_label = f"Part {part}/{n_parts}"

        print(
            f"\n{'='*60}\n"
            f"  FREE TYPE MODE (Guided)\n"
            f"{'='*60}\n"
            f"  {part_label}: {len(prompts)} sentences\n"
            f"\n"
            f"  How it works:\n"
            f"    1. A sentence appears → type it exactly\n"
            f"    2. Press ENTER to submit\n"
            f"    3. If correct → next sentence\n"
            f"       If wrong  → shows the diff, you retype\n"
            f"\n"
            f"  Tips:\n"
            f"    - All lowercase, no need for shift\n"
            f"    - Backspace to fix typos before pressing ENTER\n"
            f"    - Ctrl+C to stop early\n"
            f"{'='*60}\n"
        )

        print("  Warming up sensor (3 seconds)...")
        time.sleep(3)
        if self._stop_event.is_set():
            return

        if not self._rate_precheck(self._free_rate_gate_hz, "free_type"):
            return

        print("  Sensor ready! Start typing.\n")

        if not self._run_free_type_keyboard_self_check():
            return

        # Prompts log
        prompts_file = open(self.prompts_log_path, "w", newline="")
        prompts_writer = csv.writer(prompts_file)
        prompts_writer.writerow(["prompt_index", "timestamp_ns", "prompt_text", "typed_text", "match"])
        attempts_file = open(self.attempts_log_path, "w", newline="")
        attempts_writer = csv.writer(attempts_file)
        attempts_writer.writerow([
            "prompt_index",
            "attempt_index",
            "attempt_start_ns",
            "submit_ns",
            "elapsed_ms",
            "prompt_text",
            "typed_text",
            "match",
            "backspace_count",
            "typed_char_count",
            "keypress_count",
            "rate_now_hz",
            "rate_median_hz_recent",
            "rate_min_hz_recent",
            "rate_max_hz_recent",
        ])

        completed = 0
        for idx, prompt in enumerate(prompts):
            if self._stop_event.is_set():
                break

            matched = False
            attempt = 0

            while not matched and not self._stop_event.is_set():
                attempt += 1

                # Display prompt
                if attempt == 1:
                    print(f"  ┌─ Sentence {idx+1}/{len(prompts)} ──────────────────")
                else:
                    print(f"  ┌─ Sentence {idx+1}/{len(prompts)} (retry #{attempt}) ───")
                print(f"  │  {prompt}")
                print(f"  └─────────────────────────────────────────")

                prompt_ts = time.perf_counter_ns()
                attempt_start_ns = prompt_ts

                # Track typed characters in a buffer
                typed_buffer = []
                backspace_count = 0
                keypress_count = 0

                waiting = True
                while waiting and not self._stop_event.is_set():
                    events = self.keyboard.drain()
                    if events:
                        with self._csv_lock:
                            self._write_key_events(events)

                        for e in events:
                            if e.event_type != "press":
                                continue

                            keypress_count += 1
                            if e.key == "enter":
                                waiting = False
                                break
                            elif e.key == "backspace":
                                if typed_buffer:
                                    typed_buffer.pop()
                                backspace_count += 1
                            elif e.key == "space":
                                typed_buffer.append(" ")
                            elif len(e.key) == 1:
                                # Single character (a-z, 0-9, punctuation)
                                typed_buffer.append(e.key)
                            # Ignore other special keys (shift, ctrl, etc.)

                    # Sensor draining is handled by the background drain thread.
                    # Only sleep here to avoid busy-spinning on keyboard events.
                    time.sleep(0.05)

                # Check what was typed
                typed_text = "".join(typed_buffer).strip()
                expected = prompt.strip()
                submit_ns = time.perf_counter_ns()
                elapsed_ms = (submit_ns - attempt_start_ns) / 1_000_000.0
                rate_now = self.rate_monitor.current_rate
                rate_recent = self._recent_rate_stats(recent_sec=3.0)

                if typed_text == expected:
                    matched = True
                    prompts_writer.writerow([idx, prompt_ts, prompt, typed_text, "YES"])
                    prompts_file.flush()
                    attempts_writer.writerow([
                        idx, attempt, attempt_start_ns, submit_ns, f"{elapsed_ms:.2f}",
                        prompt, typed_text, "YES", backspace_count, len(typed_text),
                        keypress_count, f"{rate_now:.2f}", f"{rate_recent['median']:.2f}",
                        f"{rate_recent['min']:.2f}", f"{rate_recent['max']:.2f}",
                    ])
                    attempts_file.flush()
                    completed += 1
                    print(f"    ✅ Correct! ({completed}/{len(prompts)})  "
                          f"Rate: {rate_now:.0f}Hz\n")
                else:
                    # Show diff
                    prompts_writer.writerow([idx, prompt_ts, prompt, typed_text, "NO"])
                    prompts_file.flush()
                    attempts_writer.writerow([
                        idx, attempt, attempt_start_ns, submit_ns, f"{elapsed_ms:.2f}",
                        prompt, typed_text, "NO", backspace_count, len(typed_text),
                        keypress_count, f"{rate_now:.2f}", f"{rate_recent['median']:.2f}",
                        f"{rate_recent['min']:.2f}", f"{rate_recent['max']:.2f}",
                    ])
                    attempts_file.flush()
                    print(f"    ❌ Mismatch! Please retype.")
                    print(f"    Expected: {expected}")
                    print(f"    You typed: {typed_text}")
                    # Find first difference position
                    for di in range(min(len(expected), len(typed_text))):
                        if di >= len(typed_text) or expected[di] != typed_text[di]:
                            print(f"    Diff at position {di}: "
                                  f"expected '{expected[di] if di < len(expected) else '(end)'}' "
                                  f"got '{typed_text[di] if di < len(typed_text) else '(end)'}'")
                            break
                    else:
                        if len(typed_text) != len(expected):
                            print(f"    Length: expected {len(expected)}, got {len(typed_text)}")
                    print()

        prompts_file.close()
        attempts_file.close()
        print(f"\n  Done! {completed} sentences completed.")
        print(f"  Prompts log: {self.prompts_log_path}")
        print(f"  Attempts log: {self.attempts_log_path}")

    # ── Metadata ─────────────────────────────────────────────

    def _save_metadata(self, start_time: float, end_time: float):
        rate_summary = self.rate_monitor.get_rate_summary()
        with open(self.meta_path, "w") as f:
            f.write(f"Session: {self.session_prefix}\n")
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Round: {self.cfg.ROUND}\n")
            if self.group > 0:
                g = self.cfg.KEY_GROUPS[self.group]
                f.write(f"Group: {self.group} - {g['name']}\n")
                f.write(f"Keys in group: {' '.join(g['keys'])}\n")
            if self._free_type_part > 0:
                f.write(f"Free type part: {self._free_type_part}/{self._free_type_parts_total}\n")
            f.write(f"Participant: {self.cfg.PARTICIPANT_ID}\n")
            f.write(f"Start: {datetime.fromtimestamp(start_time).isoformat()}\n")
            f.write(f"End: {datetime.fromtimestamp(end_time).isoformat()}\n")
            f.write(f"Duration: {end_time - start_time:.1f}s\n")
            f.write(f"Total sensor samples: {self._sensor_count}\n")
            f.write(f"Total key events: {self._event_count}\n")
            f.write(f"Rate drop detected: {self._rate_drop_detected}\n")
            f.write(f"Session valid: {self._session_valid}\n")
            f.write(f"Sampling rate - min: {rate_summary['min']:.1f} Hz\n")
            f.write(f"Sampling rate - max: {rate_summary['max']:.1f} Hz\n")
            f.write(f"Sampling rate - avg: {rate_summary['avg']:.1f} Hz\n")
            f.write(f"Sensor CSV: {self.sensor_csv_path}\n")
            f.write(f"Events CSV: {self.events_csv_path}\n")
            if self.mode == "free_type":
                f.write(f"Prompts CSV: {self.prompts_log_path}\n")
                f.write(f"Attempts CSV: {self.attempts_log_path}\n")

    # ── Public run ───────────────────────────────────────────

    def run(self):
        start_wall = time.time()

        print(f"\n  Session: {self.session_prefix}")
        print(f"  Sensor CSV: {self.sensor_csv_path}")
        print(f"  Events CSV: {self.events_csv_path}")

        self._open_csv_files()

        # Start sensor
        try:
            self.sensor.start()
        except Exception as e:
            print(f"\n  ❌ Failed to start sensor: {e}")
            print("  Make sure you're running with sudo on Apple Silicon Mac.")
            self._close_csv_files()
            return

        # Start keyboard listener & rate monitor
        self.keyboard.start()
        self.rate_monitor.start()

        # Start background drain thread:
        # - single_key: drains sensor + keyboard
        # - free_type:  drains sensor only (prompt loop exclusively drains keyboard)
        drain_thread = threading.Thread(target=self._drain_thread_fn, daemon=True)
        drain_thread.start()

        try:
            if self.mode == "single_key":
                self._run_single_key_mode()
            elif self.mode == "free_type":
                self._run_free_type_mode()
            else:
                print(f"  Unknown mode: {self.mode}")

        except KeyboardInterrupt:
            print("\n\n  ⏹  Stopped by user (Ctrl+C)")

        finally:
            end_wall = time.time()

            # Signal everything to stop
            self._stop_event.set()

            # Wait for drain thread to finish final flush
            drain_thread.join(timeout=3.0)

            # Stop components
            self.rate_monitor.stop()
            self.keyboard.stop()
            self.sensor.stop()
            self._close_csv_files()

            if self._discard_session:
                self._session_valid = False
                self._delete_session_files()
                print(
                    f"\n{'='*60}\n"
                    f"  SESSION DISCARDED\n"
                    f"{'='*60}\n"
                    f"  Reason: {self._discard_reason}\n"
                    f"  All session artifacts removed.\n"
                    f"{'='*60}\n"
                )
                return

            self._session_valid = not self._rate_drop_detected
            self._save_metadata(start_wall, end_wall)

            rate_summary = self.rate_monitor.get_rate_summary()
            duration = end_wall - start_wall

            print(
                f"\n{'='*60}\n"
                f"  SESSION SUMMARY\n"
                f"{'='*60}\n"
                f"  Duration:        {duration:.1f}s\n"
                f"  Sensor samples:  {self._sensor_count:,}\n"
                f"  Key events:      {self._event_count:,}\n"
                f"  Avg rate:        {rate_summary['avg']:.1f} Hz\n"
                f"  Min rate:        {rate_summary['min']:.1f} Hz\n"
                f"  Max rate:        {rate_summary['max']:.1f} Hz\n"
                f"  Rate drop:       {'YES ⚠️' if self._rate_drop_detected else 'No ✓'}\n"
                f"  Session valid:   {'YES ✓' if self._session_valid else 'NO ⚠️'}\n"
                f"{'='*60}\n"
                f"  Files saved:\n"
                f"    {self.sensor_csv_path}\n"
                f"    {self.events_csv_path}\n"
                f"    {self.meta_path}\n"
                f"{'='*60}\n"
            )

            if not self._session_valid:
                print(
                    "  ⚠️  Rate drop detected. Data before the drop is still\n"
                    "  valid and saved. Consider re-running this session.\n"
                )


# ── CLI ──────────────────────────────────────────────────────

def show_group_menu(cfg: CollectorConfig) -> int:
    """Interactive group selection menu. Returns group number 1-7 or 0 for all."""
    print(
        f"\n  ┌──────────────────────────────────────────────┐\n"
        f"  │           SELECT A KEY GROUP                  │\n"
        f"  ├──────────────────────────────────────────────┤"
    )
    for gid, g in cfg.KEY_GROUPS.items():
        keys_str = "  ".join(g["keys"])
        print(f"  │  {gid})  {g['name']:<22s}  [ {keys_str} ]")
    print(
        f"  │  0)  ALL groups at once                      │\n"
        f"  └──────────────────────────────────────────────┘"
    )

    max_group = max(cfg.KEY_GROUPS.keys())
    while True:
        try:
            choice = input(f"\n  Enter group number (0-{max_group}): ").strip()
            choice = int(choice)
            if 0 <= choice <= max_group:
                return choice
            print(f"  Invalid choice, enter 0-{max_group}.")
        except (ValueError, EOFError):
            print(f"  Invalid input, enter a number 0-{max_group}.")


def main():
    parser = argparse.ArgumentParser(
        description="Keystroke Vibration Data Collector"
    )
    parser.add_argument(
        "--mode", choices=["single_key", "free_type"],
        default="single_key",
        help="Collection mode (default: single_key)"
    )
    parser.add_argument(
        "--participant", default="p01",
        help="Participant ID (default: p01)"
    )
    parser.add_argument(
        "--repeats", type=int, default=50,
        help="Presses per key in single_key mode (default: 50)"
    )
    parser.add_argument(
        "--min-rate", type=int, default=96,
        help="Runtime watchdog minimum rate in Hz (default: 96)"
    )
    parser.add_argument(
        "--single-gate-rate", type=float, default=190.0,
        help="Precheck gate for single_key mode (median Hz, default: 190)"
    )
    parser.add_argument(
        "--free-gate-rate", type=float, default=150.0,
        help="Precheck gate for free_type mode (median Hz, default: 150)"
    )
    parser.add_argument(
        "--precheck-sec", type=float, default=5.0,
        help="Sampling-rate precheck duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--group", type=int, default=-1,
        help="Key group 1-8, or 0 for all (default: interactive menu)"
    )
    parser.add_argument(
        "--round", type=int, default=1,
        help="Data collection round number (default: 1). "
             "Data saves to data/raw/round{N}/ unless --raw-subdir is set."
    )
    parser.add_argument(
        "--raw-subdir", default="",
        help="Raw data subdirectory under data/raw (e.g. single_key, boost, free_type). "
             "If set, overrides --round output path."
    )
    parser.add_argument(
        "--part", type=int, default=0,
        help="Free type part index. 0 = all sentences (default: 0). "
             "When used with --mode free_type, prompts are split by --free-groups."
    )
    parser.add_argument(
        "--free-groups", type=int, default=16,
        help="Number of groups to split free_type prompts into (default: 16, min: 1)"
    )
    args = parser.parse_args()

    cfg = CollectorConfig(
        PARTICIPANT_ID=args.participant,
        REPEATS_PER_KEY=args.repeats,
        MIN_ACCEPTABLE_RATE_HZ=args.min_rate,
        ROUND=args.round,
        RAW_SUBDIR=args.raw_subdir,
    )

    print(
        f"\n{'='*60}\n"
        f"  🎹 KEYSTROKE VIBRATION DATA COLLECTOR\n"
        f"{'='*60}\n"
        f"  Mode:        {args.mode}\n"
        f"  Raw dir:     {cfg.RAW_DIR}\n"
        f"  Participant: {args.participant}\n"
        f"  Watchdog:    >= {args.min_rate} Hz\n"
        f"  Gate(single):>= {args.single_gate_rate:.1f} Hz\n"
        f"  Gate(free):  >= {args.free_gate_rate:.1f} Hz\n"
        f"  Precheck:    {args.precheck_sec:.1f}s\n"
    )

    # Group selection (only for single_key mode)
    group = 0
    if args.mode == "single_key":
        if args.group == -1:
            # Interactive menu
            group = show_group_menu(cfg)
        else:
            group = args.group

        if group == 0:
            # All keys
            cfg.KEY_LIST = []
            for g in cfg.KEY_GROUPS.values():
                cfg.KEY_LIST.extend(g["keys"])
            print(f"  Selected:    ALL keys ({len(cfg.KEY_LIST)} keys)")
        else:
            g = cfg.KEY_GROUPS[group]
            cfg.KEY_LIST = g["keys"]
            print(f"  Selected:    {g['name']}")
            print(f"  Keys:        {' '.join(g['keys'])}")

        print(f"  Repeats/key: {args.repeats}")
        total = len(cfg.KEY_LIST) * args.repeats
        print(f"  Total:       {len(cfg.KEY_LIST)} keys × {args.repeats} = {total} presses\n")

    if args.mode == "free_type" and args.part > 0:
        print(f"  Part:        {args.part}/{max(1, args.free_groups)}\n")

    if os.geteuid() != 0:
        print("  ⚠️  Not running as root! SPU sensor requires sudo.")
        print("  Run:  sudo .venv/bin/python3 collector.py --mode single_key\n")
        sys.exit(1)

    collector = DataCollector(
        cfg,
        args.mode,
        group,
        free_type_part=args.part,
        free_type_parts_total=args.free_groups,
        single_rate_gate_hz=args.single_gate_rate,
        free_rate_gate_hz=args.free_gate_rate,
        precheck_sec=args.precheck_sec,
    )
    collector.run()


if __name__ == "__main__":
    main()
