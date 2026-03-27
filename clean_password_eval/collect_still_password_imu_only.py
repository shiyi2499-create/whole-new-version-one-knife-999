from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensor_reader import SensorReader, SensorSample


@dataclass
class CapturePaths:
    prefix: str
    sensor_csv: str
    protocol_json: str
    meta_txt: str


def _next_trial_index(dataset_root: Path) -> int:
    trials = []
    for p in dataset_root.glob("*_protocol.json"):
        stem = p.name
        marker = "_trial"
        if marker not in stem:
            continue
        try:
            chunk = stem.split(marker, 1)[1][:3]
            trials.append(int(chunk))
        except Exception:
            continue
    return 0 if not trials else max(trials) + 1


def _build_paths(
    dataset_root: Path,
    participant_id: str,
    label: str,
    trial_index: Optional[int],
) -> CapturePaths:
    dataset_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if trial_index is None:
        trial_index = _next_trial_index(dataset_root)
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)[:32] or "unlabeled"
    prefix = f"{participant_id}_still_password_imu_only_{safe_label}_trial{trial_index:03d}_{ts}"
    return CapturePaths(
        prefix=prefix,
        sensor_csv=str(dataset_root / f"{prefix}_sensor.csv"),
        protocol_json=str(dataset_root / f"{prefix}_protocol.json"),
        meta_txt=str(dataset_root / f"{prefix}_meta.txt"),
    )


def _write_sensor_rows(writer: csv.writer, samples: list[SensorSample]) -> int:
    for s in samples:
        writer.writerow(
            [
                int(s.timestamp_ns),
                f"{s.accel_x:.8f}",
                f"{s.accel_y:.8f}",
                f"{s.accel_z:.8f}",
                f"{s.gyro_x:.6f}",
                f"{s.gyro_y:.6f}",
                f"{s.gyro_z:.6f}",
            ]
        )
    return len(samples)


def _drain_once(sensor: SensorReader, sensor_writer: csv.writer) -> int:
    samples = sensor.drain()
    return _write_sensor_rows(sensor_writer, samples)


def run_capture(
    label: str,
    participant_id: str = "p01",
    dataset_root: str = "data/raw/clean_password_eval_local_imu_only",
    pre_idle_sec: float = 5.0,
    typing_sec: float = 12.0,
    post_idle_sec: float = 5.0,
    note: str = "",
    trial_index: Optional[int] = None,
    force_macimu: bool = False,
) -> CapturePaths:
    dataset_dir = (REPO_ROOT / dataset_root).resolve() if not os.path.isabs(dataset_root) else Path(dataset_root)
    paths = _build_paths(dataset_dir, participant_id, label, trial_index)

    sensor = SensorReader(force_macimu=force_macimu)
    total_sensor = 0
    phase_markers: dict[str, int] = {}

    print(f"\nSession: {paths.prefix}")
    print(f"Dataset root: {dataset_dir}")
    print(f"Sensor CSV: {paths.sensor_csv}")
    print(f"Label (manual truth): {label}")
    print(f"Pre-idle: {pre_idle_sec:.1f}s")
    print(f"Typing window: {typing_sec:.1f}s")
    print(f"Post-idle: {post_idle_sec:.1f}s")
    print("This collector records IMU only. No keyboard events are captured.")

    with open(paths.sensor_csv, "w", newline="") as sf:
        sensor_writer = csv.writer(sf)
        sensor_writer.writerow(["timestamp_ns", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])

        sensor.start()
        sensor.drain()
        try:
            print("\n[1/3] Pre-idle started. Keep still.")
            phase_markers["capture_start_ns"] = time.perf_counter_ns()
            phase_markers["pre_idle_start_ns"] = phase_markers["capture_start_ns"]
            t_end = time.time() + pre_idle_sec
            while time.time() < t_end:
                total_sensor += _drain_once(sensor, sensor_writer)
                time.sleep(0.01)

            print("\n[2/3] Typing window started. Input the password now.")
            phase_markers["typing_start_ns"] = time.perf_counter_ns()
            t_end = time.time() + typing_sec
            while time.time() < t_end:
                total_sensor += _drain_once(sensor, sensor_writer)
                time.sleep(0.01)

            print("\n[3/3] Post-idle started. Keep still again.")
            phase_markers["post_idle_start_ns"] = time.perf_counter_ns()
            t_end = time.time() + post_idle_sec
            while time.time() < t_end:
                total_sensor += _drain_once(sensor, sensor_writer)
                time.sleep(0.01)
            phase_markers["capture_end_ns"] = time.perf_counter_ns()
        finally:
            total_sensor += _drain_once(sensor, sensor_writer)
            sensor.stop()

    protocol = {
        "session_prefix": paths.prefix,
        "participant_id": participant_id,
        "dataset_name": "clean_password_eval_local_imu_only",
        "imu_only": True,
        "eval_only": True,
        "include_in_training": False,
        "capture_pattern": "still_then_password_typing_window_then_still",
        "pre_idle_sec": float(pre_idle_sec),
        "typing_sec": float(typing_sec),
        "post_idle_sec": float(post_idle_sec),
        "manual_label": label,
        "backend_name": sensor.backend_name,
        "phase_markers_ns": phase_markers,
        "paths": asdict(paths),
        "note": note,
        "created_at": datetime.now().isoformat(),
        "warning": "IMU-only eval capture. Manual label must be used during analysis.",
    }
    Path(paths.protocol_json).write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(paths.meta_txt, "w", encoding="utf-8") as f:
        f.write(f"Session: {paths.prefix}\n")
        f.write("Mode: clean_password_eval_local_imu_only\n")
        f.write(f"Participant: {participant_id}\n")
        f.write(f"Manual label: {label}\n")
        f.write(f"Sensor backend: {sensor.backend_name}\n")
        f.write(f"Total sensor rows: {total_sensor}\n")
        f.write(f"Pre-idle: {pre_idle_sec:.2f}s\n")
        f.write(f"Typing window: {typing_sec:.2f}s\n")
        f.write(f"Post-idle: {post_idle_sec:.2f}s\n")
        if note:
            f.write(f"Note: {note}\n")
        f.write("Eval only: YES\n")
        f.write("Include in training: NO\n")
        for k, v in phase_markers.items():
            f.write(f"{k}: {v}\n")

    print("\nDone.")
    print(f"  Saved sensor CSV: {paths.sensor_csv}")
    print(f"  Saved protocol:   {paths.protocol_json}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record IMU-only still-password-still samples on the local Mac.")
    parser.add_argument("--label", required=True, help="Manual truth label, e.g. password123")
    parser.add_argument("--participant", default="p01", help="Participant/device id label")
    parser.add_argument("--dataset-root", default="data/raw/clean_password_eval_local_imu_only")
    parser.add_argument("--pre-idle-sec", type=float, default=5.0)
    parser.add_argument("--typing-sec", type=float, default=12.0)
    parser.add_argument("--post-idle-sec", type=float, default=5.0)
    parser.add_argument("--trial-index", type=int, default=None)
    parser.add_argument("--note", default="")
    parser.add_argument("--force-macimu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_capture(
        label=args.label,
        participant_id=args.participant,
        dataset_root=args.dataset_root,
        pre_idle_sec=args.pre_idle_sec,
        typing_sec=args.typing_sec,
        post_idle_sec=args.post_idle_sec,
        note=args.note,
        trial_index=args.trial_index,
        force_macimu=args.force_macimu,
    )


if __name__ == "__main__":
    main()
