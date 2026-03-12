"""
Preprocessor - Post-processing pipeline
========================================
Takes raw sensor CSV + key events CSV and produces:
  1. Windowed samples aligned to each keystroke
  2. Feature-ready training dataset

Run:  python3 preprocessor.py --rounds single_key            (default: single_key only, 190Hz)
      python3 preprocessor.py --rounds free_type --session-type free_type

NOTE on sample rate:
  This project currently targets the high-rate collection regime (~190-200 Hz).
  Default target-rate is 190, giving 57 samples per 300ms window.
  If you are processing old low-rate data, pass --target-rate explicitly.
  If you have old data processed at a different target rate, delete data/processed/*.npz
  and re-run this script.
"""

import os
import csv
import sys
import json
import argparse
import re
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class WindowConfig:
    """Configuration for keystroke window extraction."""
    pre_trigger_ms: int = 100       # ms before keypress
    post_trigger_ms: int = 200      # ms after keypress
    target_rate_hz: int = 190       # resample ALL windows to this rate (high-rate regime)
    min_window_samples: int = 10    # discard windows smaller than this

    @property
    def target_window_len(self) -> int:
        """Fixed number of samples per window after resampling."""
        total_ms = self.pre_trigger_ms + self.post_trigger_ms
        return int(total_ms / 1000 * self.target_rate_hz)  # 300ms × 190Hz = 57


class Preprocessor:
    """
    Processes raw data into ML-ready training samples.

    Pipeline:
      1. Load sensor CSV + events CSV
      2. For each 'press' event, cut a window of sensor data
      3. Validate window (enough samples, no overlap with neighbors)
      4. Save windowed samples as NPZ or CSV
    """

    def __init__(self, session_prefix: str, output_dir: str = "data/processed",
                 window_cfg: Optional[WindowConfig] = None):
        self.session_prefix = session_prefix
        self.output_dir = output_dir
        self.wcfg = window_cfg or WindowConfig()

        self.sensor_path = f"{session_prefix}_sensor.csv"
        self.events_path = f"{session_prefix}_events.csv"

        os.makedirs(output_dir, exist_ok=True)

        # Data storage
        self.sensor_data: Optional[np.ndarray] = None  # Nx7 (ts, ax,ay,az, gx,gy,gz)
        self.key_events: list[dict] = []
        self.windows: list[dict] = []  # final windowed samples

    def _infer_group_tag(self, session_id: str) -> str:
        m_group = re.search(r"_g(\d+)_", session_id)
        if m_group:
            return f"g{m_group.group(1)}"
        m_part = re.search(r"_part(\d+)_", session_id)
        if m_part:
            return f"part{m_part.group(1)}"
        return ""

    def load(self):
        """Load raw CSV files into memory."""
        print(f"  Loading sensor data from: {self.sensor_path}")
        sensor_rows = []
        with open(self.sensor_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sensor_rows.append([
                    int(row["timestamp_ns"]),
                    float(row["accel_x"]),
                    float(row["accel_y"]),
                    float(row["accel_z"]),
                    float(row["gyro_x"]),
                    float(row["gyro_y"]),
                    float(row["gyro_z"]),
                ])
        self.sensor_data = np.array(sensor_rows)
        print(f"    → {len(self.sensor_data)} sensor samples loaded")

        print(f"  Loading key events from: {self.events_path}")
        self.key_events = []
        with open(self.events_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.key_events.append({
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "key": row["key"],
                    "event_type": row["event_type"],
                })
        print(f"    → {len(self.key_events)} key events loaded")

    def compute_actual_rate(self) -> float:
        """Compute actual sampling rate from sensor timestamps."""
        if len(self.sensor_data) < 2:
            return 0.0
        ts = self.sensor_data[:, 0]
        dt_ns = np.diff(ts)
        median_dt = np.median(dt_ns)
        if median_dt > 0:
            return 1e9 / median_dt
        return 0.0

    @staticmethod
    def resample_window(timestamps_ns: np.ndarray, values: np.ndarray,
                        target_len: int) -> np.ndarray:
        """
        Resample a variable-length window to exactly target_len samples
        using linear interpolation on the original timestamps.

        Regardless of whether a window has 35 or 45 raw samples, the output
        is always (target_len, 6) at a uniform time grid.

        Args:
            timestamps_ns: shape (N,) - original timestamps
            values:        shape (N, 6) - sensor channels
            target_len:    desired output length (e.g. 39)

        Returns:
            shape (target_len, 6) - uniformly resampled data
        """
        n = len(timestamps_ns)
        if n < 2:
            return np.zeros((target_len, values.shape[1]), dtype=np.float64)

        t_start = timestamps_ns[0]
        t_end = timestamps_ns[-1]
        uniform_t = np.linspace(t_start, t_end, target_len)

        n_channels = values.shape[1]
        resampled = np.zeros((target_len, n_channels), dtype=np.float64)
        for ch in range(n_channels):
            resampled[:, ch] = np.interp(uniform_t, timestamps_ns, values[:, ch])

        return resampled

    def extract_windows(self) -> list[dict]:
        """
        For each keypress event, extract a window of sensor data,
        then resample to uniform target_window_len.
        """
        print("\n  Extracting keystroke windows...")

        target_len = self.wcfg.target_window_len
        print(f"    Target: {target_len} samples per window "
              f"({self.wcfg.pre_trigger_ms}+{self.wcfg.post_trigger_ms}ms "
              f"@ {self.wcfg.target_rate_hz}Hz)")

        ts = self.sensor_data[:, 0]
        sensor_vals = self.sensor_data[:, 1:]

        pre_ns = self.wcfg.pre_trigger_ms * 1_000_000
        post_ns = self.wcfg.post_trigger_ms * 1_000_000

        presses = [e for e in self.key_events if e["event_type"] == "press"]
        print(f"    {len(presses)} press events found")

        windows = []
        skipped = 0
        raw_lengths = []

        for evt in presses:
            evt_ts = evt["timestamp_ns"]
            win_start = evt_ts - pre_ns
            win_end = evt_ts + post_ns

            idx_start = np.searchsorted(ts, win_start, side="left")
            idx_end = np.searchsorted(ts, win_end, side="right")

            n = idx_end - idx_start

            if n < self.wcfg.min_window_samples:
                skipped += 1
                continue

            window_ts = ts[idx_start:idx_end]
            window_data = sensor_vals[idx_start:idx_end]
            raw_lengths.append(n)

            # ── Resample to fixed length ──
            resampled = self.resample_window(window_ts, window_data, target_len)

            windows.append({
                "key": evt["key"],
                "timestamp_ns": evt_ts,
                "window": resampled,          # always (target_len, 6)
                "raw_samples": n,             # original count (for diagnostics)
                "n_samples": target_len,      # uniform
            })

        self.windows = windows
        print(f"    ✓ {len(windows)} valid windows extracted & resampled")
        if skipped:
            print(f"    ⚠ {skipped} windows skipped (too few raw samples)")

        if raw_lengths:
            print(f"    Raw window sizes: min={min(raw_lengths)}, "
                  f"max={max(raw_lengths)}, avg={np.mean(raw_lengths):.1f}")
            print(f"    After resampling: all → {target_len} samples")

        # Stats by key
        key_counts = defaultdict(int)
        for w in windows:
            key_counts[w["key"]] += 1
        print("\n  Samples per key:")
        for k in sorted(key_counts.keys()):
            print(f"    '{k}': {key_counts[k]}")

        return windows

    def save_npz(self):
        """
        Save all windows as a single .npz file for ML training.
        All windows are already resampled to target_window_len.
        """
        if not self.windows:
            print("  No windows to save!")
            return

        target_len = self.wcfg.target_window_len

        X = np.array([w["window"] for w in self.windows], dtype=np.float32)
        y = np.array([w["key"] for w in self.windows])
        timestamps = np.array([w["timestamp_ns"] for w in self.windows], dtype=np.int64)

        basename = os.path.basename(self.session_prefix)
        source_dir = os.path.basename(os.path.dirname(self.session_prefix))
        group_tag = self._infer_group_tag(basename)
        session_ids = np.array([basename] * len(self.windows))
        source_dirs = np.array([source_dir] * len(self.windows))
        group_tags = np.array([group_tag] * len(self.windows))
        out_path = os.path.join(self.output_dir, f"{basename}_dataset.npz")

        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            timestamps=timestamps,
            session_ids=session_ids,
            source_dirs=source_dirs,
            group_tags=group_tags,
            target_rate_hz=self.wcfg.target_rate_hz,
            window_len=target_len,
            channels=["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
        )
        print(f"\n  ✓ Dataset saved: {out_path}")
        print(f"    X shape: {X.shape}  (samples × timesteps × channels)")
        print(f"    y shape: {y.shape}  (labels)")
        print(f"    Unique keys: {len(set(y.tolist()))}")

        return out_path

    def save_flat_csv(self):
        """
        Save as flat CSV (one row per window) for tools like sklearn.
        Each row = [key, ts, raw_samples, ax_0, ay_0, ..., gz_38]
        """
        if not self.windows:
            print("  No windows to save!")
            return

        target_len = self.wcfg.target_window_len
        basename = os.path.basename(self.session_prefix)
        out_path = os.path.join(self.output_dir, f"{basename}_flat.csv")

        channels = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
        header = ["key", "timestamp_ns", "raw_samples"]
        for i in range(target_len):
            for ch in channels:
                header.append(f"{ch}_{i}")

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for w in self.windows:
                row = [w["key"], w["timestamp_ns"], w["raw_samples"]]
                row.extend(w["window"].flatten().tolist())
                writer.writerow(row)

        print(f"\n  ✓ Flat CSV saved: {out_path}")
        print(f"    {len(self.windows)} rows × {len(header)} columns")

        return out_path

    def run(self):
        """Full preprocessing pipeline."""
        print(
            f"\n{'='*60}\n"
            f"  PREPROCESSOR\n"
            f"{'='*60}"
        )

        self.load()

        actual_rate = self.compute_actual_rate()
        target_len = self.wcfg.target_window_len

        print(f"\n  Actual sampling rate:  {actual_rate:.1f} Hz")
        print(f"  Target resample rate:  {self.wcfg.target_rate_hz} Hz")
        print(f"  Window: {self.wcfg.pre_trigger_ms}ms + {self.wcfg.post_trigger_ms}ms "
              f"= {self.wcfg.pre_trigger_ms + self.wcfg.post_trigger_ms}ms")
        print(f"  Output window size:    {target_len} samples (fixed)")

        if actual_rate < 50:
            print("  ⚠ WARNING: actual rate very low, data quality may be poor")

        self.extract_windows()

        # Save both formats
        self.save_npz()
        self.save_flat_csv()

        print(
            f"\n{'='*60}\n"
            f"  ✓ Preprocessing complete!\n"
            f"  All windows resampled to {target_len} samples @ "
            f"{self.wcfg.target_rate_hz}Hz\n"
            f"  Sessions with different raw rates are now compatible.\n"
            f"{'='*60}\n"
        )


def resolve_data_dirs(selectors: list[str], data_root: str = "data/raw") -> list[str]:
    """
    Resolve user-provided selectors to concrete raw-data directories.

    Supported selector forms:
      - direct path: data/raw/single_key
      - folder name under data_root: single_key / free_type / boost / round4
      - legacy round number: 4  -> data/raw/round4
    """
    out: list[str] = []
    seen: set[str] = set()

    for sel in selectors:
        cand = []
        if os.path.isdir(sel):
            cand.append(sel)
        cand.append(os.path.join(data_root, sel))
        cand.append(os.path.join(data_root, f"round{sel}"))

        resolved = None
        for c in cand:
            if os.path.isdir(c):
                resolved = c
                break

        if resolved is None:
            print(f"  ⚠ Data dir not found for selector: {sel}")
            continue

        rp = os.path.normpath(resolved)
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    return out


def find_sessions_in_dirs(data_dirs: list[str],
                          session_type: str = "single_key") -> list[str]:
    """
    Auto-discover session prefixes in the given data directories.
    Looks for *_sensor.csv files and extracts the prefix.

    Args:
        data_dirs:     list of raw-data directory paths to scan
        session_type:  'single_key' | 'free_type' | 'all'
                       Filters sessions by the mode embedded in the filename.
                       Default is 'single_key' to exclude free_type data from
                       training (free_type is reserved for transfer/fine-tuning).
    """
    sessions = []
    for rd in data_dirs:
        if not os.path.isdir(rd):
            print(f"  ⚠ Directory not found: {rd}")
            continue
        for f in sorted(os.listdir(rd)):
            if not f.endswith("_sensor.csv"):
                continue
            # Filter by mode embedded in filename
            if session_type != "all":
                if f"_{session_type}_" not in f:
                    continue
            prefix = os.path.join(rd, f.replace("_sensor.csv", ""))
            events_path = prefix + "_events.csv"
            if os.path.exists(events_path):
                sessions.append(prefix)
    return sessions


def find_sessions_in_rounds(round_dirs: list[str],
                            session_type: str = "single_key") -> list[str]:
    """
    Backward-compatible alias.
    """
    return find_sessions_in_dirs(round_dirs, session_type=session_type)


def infer_group_tag(session_id: str) -> str:
    m_group = re.search(r"_g(\d+)_", session_id)
    if m_group:
        return f"g{m_group.group(1)}"
    m_part = re.search(r"_part(\d+)_", session_id)
    if m_part:
        return f"part{m_part.group(1)}"
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw keystroke vibration data"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session", nargs="+",
        help="One or more session prefix paths (explicit)."
    )
    group.add_argument(
        "--rounds", nargs="+",
        help="Data selectors to scan (e.g. --rounds single_key free_type 3 round4). "
             "Each selector can be a folder name under data/raw, a direct path, or a legacy round number."
    )
    parser.add_argument(
        "--pre-ms", type=int, default=100,
        help="Pre-trigger window in ms (default: 100)"
    )
    parser.add_argument(
        "--post-ms", type=int, default=200,
        help="Post-trigger window in ms (default: 200)"
    )
    parser.add_argument(
        "--target-rate", type=int, default=190,
        help="Resample all windows to this rate in Hz (default: 190, for high-rate gated data)"
    )
    parser.add_argument(
        "--session-type",
        choices=["single_key", "free_type", "all"],
        default="single_key",
        help="Which session types to include when using --rounds selectors. "
             "'single_key' (default): only guided single-key sessions → "
             "clean isolated-keystroke data for training. "
             "'free_type': only free-typing sessions (reserved for fine-tuning). "
             "'all': include everything."
    )
    args = parser.parse_args()

    wcfg = WindowConfig(
        pre_trigger_ms=args.pre_ms,
        post_trigger_ms=args.post_ms,
        target_rate_hz=args.target_rate,
    )

    # Resolve session list
    if args.rounds:
        data_dirs = resolve_data_dirs(args.rounds, data_root="data/raw")
        sessions = find_sessions_in_dirs(data_dirs, session_type=args.session_type)
        if not sessions:
            print(f"  ❌ No '{args.session_type}' sessions found in specified selectors!")
            sys.exit(1)
        print(f"\n  Auto-discovered {len(sessions)} '{args.session_type}' sessions "
              f"across {len(data_dirs)} data dir(s):")
        for s in sessions:
            print(f"    {s}")
        print()
    else:
        sessions = args.session

    all_npz_paths: list[tuple[str, str]] = []

    for sess in sessions:
        proc = Preprocessor(session_prefix=sess, window_cfg=wcfg)
        proc.run()
        npz_path = os.path.join(
            proc.output_dir,
            f"{os.path.basename(sess)}_dataset.npz"
        )
        if os.path.exists(npz_path):
            all_npz_paths.append((npz_path, sess))

    # If multiple sessions, also merge into one combined dataset
    if len(all_npz_paths) > 1:
        print(f"\n{'='*60}")
        print(f"  MERGING {len(all_npz_paths)} datasets...")
        print(f"{'='*60}")

        X_all, y_all, ts_all = [], [], []
        session_all, source_all, group_all = [], [], []
        for p, sess in all_npz_paths:
            data = np.load(p, allow_pickle=True)
            n = len(data["X"])
            sess_id = os.path.basename(sess)
            source_dir = os.path.basename(os.path.dirname(sess))
            group_tag = infer_group_tag(sess_id)

            X_all.append(data["X"])
            y_all.append(data["y"])
            ts_all.append(data["timestamps"])
            if "session_ids" in data.files:
                session_all.append(data["session_ids"].astype(str))
            else:
                session_all.append(np.array([sess_id] * n))
            if "source_dirs" in data.files:
                source_all.append(data["source_dirs"].astype(str))
            else:
                source_all.append(np.array([source_dir] * n))
            if "group_tags" in data.files:
                group_all.append(data["group_tags"].astype(str))
            else:
                group_all.append(np.array([group_tag] * n))
            print(f"    + {os.path.basename(p)}: {len(data['X'])} samples")

        X_merged = np.concatenate(X_all, axis=0)
        y_merged = np.concatenate(y_all, axis=0)
        ts_merged = np.concatenate(ts_all, axis=0)
        session_merged = np.concatenate(session_all, axis=0)
        source_merged = np.concatenate(source_all, axis=0)
        group_merged = np.concatenate(group_all, axis=0)

        merged_path = os.path.join("data/processed", "merged_dataset.npz")
        np.savez_compressed(
            merged_path,
            X=X_merged, y=y_merged, timestamps=ts_merged,
            session_ids=session_merged,
            source_dirs=source_merged,
            group_tags=group_merged,
            target_rate_hz=wcfg.target_rate_hz,
            window_len=wcfg.target_window_len,
            channels=["accel_x", "accel_y", "accel_z",
                       "gyro_x", "gyro_y", "gyro_z"],
        )

        unique_keys = sorted(set(y_merged.tolist()))
        key_counts = defaultdict(int)
        for k in y_merged:
            key_counts[k] += 1

        print(f"\n  ✓ Merged dataset: {merged_path}")
        print(f"    X shape: {X_merged.shape}")
        print(f"    Unique keys: {len(unique_keys)} → {' '.join(unique_keys)}")
        print(f"    Samples per key (min/max): "
              f"{min(key_counts.values())}/{max(key_counts.values())}")
        print(f"    All sessions resampled to {wcfg.target_rate_hz}Hz ✓")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
