from __future__ import annotations

import argparse
import csv
import getpass
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

from keyboard_listener import KeyboardListener
from sensor_reader import SensorReader, SensorSample


@dataclass
class CapturePaths:
    prefix: str
    sensor_csv: str
    events_csv: str
    attempts_csv: str
    protocol_json: str
    meta_txt: str


def _next_trial_index(dataset_root: Path) -> int:
    trials = []
    for p in dataset_root.glob('*_protocol.json'):
        stem = p.name
        marker = '_trial'
        if marker not in stem:
            continue
        try:
            chunk = stem.split(marker, 1)[1][:3]
            trials.append(int(chunk))
        except Exception:
            continue
    return 0 if not trials else max(trials) + 1


def _build_paths(dataset_root: Path, participant_id: str, reference_text: str, trial_index: Optional[int]) -> CapturePaths:
    dataset_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if trial_index is None:
        trial_index = _next_trial_index(dataset_root)
    prefix = f"{participant_id}_clean_password_eval_len{len(reference_text)}_trial{trial_index:03d}_{ts}"
    return CapturePaths(
        prefix=prefix,
        sensor_csv=str(dataset_root / f'{prefix}_sensor.csv'),
        events_csv=str(dataset_root / f'{prefix}_events.csv'),
        attempts_csv=str(dataset_root / f'{prefix}_attempts.csv'),
        protocol_json=str(dataset_root / f'{prefix}_protocol.json'),
        meta_txt=str(dataset_root / f'{prefix}_meta.txt'),
    )


def _write_sensor_rows(writer: csv.writer, samples: list[SensorSample]) -> int:
    for s in samples:
        writer.writerow([
            int(s.timestamp_ns),
            f'{s.accel_x:.8f}', f'{s.accel_y:.8f}', f'{s.accel_z:.8f}',
            f'{s.gyro_x:.6f}', f'{s.gyro_y:.6f}', f'{s.gyro_z:.6f}',
        ])
    return len(samples)


def _normalize_press_char(key: str) -> Optional[str]:
    if key == 'space':
        return ' '
    if len(key) == 1:
        return key
    return None


def _drain_once(sensor: SensorReader, keyboard: KeyboardListener, sensor_writer: csv.writer, event_writer: csv.writer) -> tuple[int, list[dict]]:
    samples = sensor.drain()
    events = keyboard.drain()
    sensor_count = _write_sensor_rows(sensor_writer, samples)
    event_payloads = []
    for e in events:
        event_writer.writerow([int(e.timestamp_ns), e.key, e.event_type])
        event_payloads.append({'timestamp_ns': int(e.timestamp_ns), 'key': e.key, 'event_type': e.event_type})
    return sensor_count, event_payloads


def run_capture(reference_text: str, participant_id: str = 'p01', dataset_root: str = 'data/raw/clean_password_eval', idle_sec: float = 3.0, note: str = '', trial_index: Optional[int] = None, force_macimu: bool = False) -> CapturePaths:
    dataset_dir = (REPO_ROOT / dataset_root).resolve() if not os.path.isabs(dataset_root) else Path(dataset_root)
    paths = _build_paths(dataset_dir, participant_id, reference_text, trial_index)

    sensor = SensorReader(force_macimu=force_macimu)
    keyboard = KeyboardListener()

    total_sensor = 0
    total_events = 0
    phase_markers: dict[str, int] = {}
    typed_buffer: list[str] = []
    backspace_count = 0
    keypress_count = 0
    first_keypress_ns: Optional[int] = None
    submit_ns: Optional[int] = None

    print(f'\nSession: {paths.prefix}')
    print(f'Dataset root: {dataset_dir}')
    print(f'Sensor CSV: {paths.sensor_csv}')
    print(f'Events CSV: {paths.events_csv}')
    print(f'Idle before typing: {idle_sec:.1f}s')
    print(f'Idle after Enter: {idle_sec:.1f}s')
    print('This dataset is eval-only and must NOT be added to training.')

    with open(paths.sensor_csv, 'w', newline='') as sf, open(paths.events_csv, 'w', newline='') as ef, open(paths.attempts_csv, 'w', newline='') as af:
        sensor_writer = csv.writer(sf)
        event_writer = csv.writer(ef)
        attempts_writer = csv.writer(af)
        sensor_writer.writerow(['timestamp_ns', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        event_writer.writerow(['timestamp_ns', 'key', 'event_type'])
        attempts_writer.writerow([
            'session_prefix', 'reference_text', 'typed_text', 'match', 'attempt_start_ns', 'first_keypress_ns',
            'submit_ns', 'elapsed_ms', 'reference_length', 'typed_length', 'backspace_count', 'keypress_count',
            'idle_before_sec', 'idle_after_sec', 'backend_name'
        ])

        start_wall = time.time()
        sensor.start()
        keyboard.start()
        keyboard.drain()
        sensor.drain()

        try:
            print('\nGet ready. Pre-idle capture started...')
            phase_markers['capture_start_ns'] = time.perf_counter_ns()
            phase_markers['pre_idle_start_ns'] = phase_markers['capture_start_ns']
            t_end = time.time() + idle_sec
            while time.time() < t_end:
                s_count, events = _drain_once(sensor, keyboard, sensor_writer, event_writer)
                total_sensor += s_count
                total_events += len(events)
                time.sleep(0.01)

            print('\nStart typing the password now. Press Enter to finish this trial.')
            phase_markers['typing_prompt_ns'] = time.perf_counter_ns()
            attempt_start_ns = phase_markers['typing_prompt_ns']
            waiting = True
            while waiting:
                s_count, events = _drain_once(sensor, keyboard, sensor_writer, event_writer)
                total_sensor += s_count
                total_events += len(events)
                for e in events:
                    if e['event_type'] != 'press':
                        continue
                    if first_keypress_ns is None and e['key'] != 'enter':
                        first_keypress_ns = int(e['timestamp_ns'])
                        phase_markers['typing_first_keypress_ns'] = first_keypress_ns
                    keypress_count += 1
                    if e['key'] == 'enter':
                        submit_ns = int(e['timestamp_ns'])
                        phase_markers['typing_submit_ns'] = submit_ns
                        waiting = False
                        break
                    if e['key'] == 'backspace':
                        if typed_buffer:
                            typed_buffer.pop()
                        backspace_count += 1
                        continue
                    ch = _normalize_press_char(e['key'])
                    if ch is not None:
                        typed_buffer.append(ch)
                time.sleep(0.01)

            print('\nEnter detected. Post-idle capture started...')
            phase_markers['post_idle_start_ns'] = time.perf_counter_ns()
            t_end = time.time() + idle_sec
            while time.time() < t_end:
                s_count, events = _drain_once(sensor, keyboard, sensor_writer, event_writer)
                total_sensor += s_count
                total_events += len(events)
                time.sleep(0.01)
            phase_markers['capture_end_ns'] = time.perf_counter_ns()

        finally:
            s_count, events = _drain_once(sensor, keyboard, sensor_writer, event_writer)
            total_sensor += s_count
            total_events += len(events)
            keyboard.stop()
            sensor.stop()

        typed_text = ''.join(typed_buffer)
        elapsed_ms = None if submit_ns is None else (submit_ns - attempt_start_ns) / 1_000_000.0
        attempts_writer.writerow([
            paths.prefix,
            reference_text,
            typed_text,
            'YES' if typed_text == reference_text else 'NO',
            int(attempt_start_ns),
            '' if first_keypress_ns is None else int(first_keypress_ns),
            '' if submit_ns is None else int(submit_ns),
            '' if elapsed_ms is None else f'{elapsed_ms:.2f}',
            len(reference_text),
            len(typed_text),
            backspace_count,
            keypress_count,
            f'{idle_sec:.2f}',
            f'{idle_sec:.2f}',
            sensor.backend_name,
        ])

    protocol = {
        'session_prefix': paths.prefix,
        'participant_id': participant_id,
        'dataset_name': 'clean_password_eval',
        'eval_only': True,
        'include_in_training': False,
        'capture_pattern': 'still_3s_then_password_until_enter_then_still_3s',
        'idle_before_sec': float(idle_sec),
        'idle_after_sec': float(idle_sec),
        'reference_text': reference_text,
        'reference_length': len(reference_text),
        'typed_text': ''.join(typed_buffer),
        'typed_length': len(''.join(typed_buffer)),
        'typed_matches_reference': ''.join(typed_buffer) == reference_text,
        'backspace_count': backspace_count,
        'keypress_count': keypress_count,
        'backend_name': sensor.backend_name,
        'phase_markers_ns': phase_markers,
        'paths': asdict(paths),
        'note': note,
        'created_at': datetime.now().isoformat(),
        'warning': 'This dataset is for evaluation only. Do not merge into any training split.',
    }
    Path(paths.protocol_json).write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')

    with open(paths.meta_txt, 'w', encoding='utf-8') as f:
        f.write(f'Session: {paths.prefix}\n')
        f.write('Mode: clean_password_eval\n')
        f.write(f'Participant: {participant_id}\n')
        f.write(f'Reference text: {reference_text}\n')
        f.write(f'Reference length: {len(reference_text)}\n')
        f.write(f'Typed text: {"".join(typed_buffer)}\n')
        f.write(f'Typed match: {"YES" if "".join(typed_buffer)==reference_text else "NO"}\n')
        f.write(f'Sensor backend: {sensor.backend_name}\n')
        f.write(f'Total sensor rows: {total_sensor}\n')
        f.write(f'Total keyboard events: {total_events}\n')
        f.write(f'Idle before: {idle_sec:.2f}s\n')
        f.write(f'Idle after: {idle_sec:.2f}s\n')
        if note:
            f.write(f'Note: {note}\n')
        f.write('Eval only: YES\n')
        f.write('Include in training: NO\n')
        for k, v in phase_markers.items():
            f.write(f'{k}: {v}\n')

    print('\nDone.')
    print(f'  Typed text: {"".join(typed_buffer)}')
    print(f'  Match: {"YES" if "".join(typed_buffer)==reference_text else "NO"}')
    print(f'  Sensor rows: {total_sensor}')
    print(f'  Keyboard events: {total_events}')
    print(f'  Protocol: {paths.protocol_json}')
    print('  Reminder: this dataset is eval-only and should stay out of training splits.')
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description='Collect an eval-only clean password trial: still -> one password -> still.')
    parser.add_argument('--participant', default='p01')
    parser.add_argument('--dataset-root', default='data/raw/clean_password_eval')
    parser.add_argument('--idle-sec', type=float, default=3.0)
    parser.add_argument('--label', default='')
    parser.add_argument('--note', default='')
    parser.add_argument('--trial-index', type=int, default=None)
    parser.add_argument('--force-macimu', action='store_true')
    args = parser.parse_args()

    label = args.label.strip()
    if not label:
        label = getpass.getpass('Reference password label (not recorded, used only as ground truth): ').strip()
    if not label:
        raise SystemExit('Reference password label cannot be empty.')

    run_capture(
        reference_text=label,
        participant_id=args.participant,
        dataset_root=args.dataset_root,
        idle_sec=args.idle_sec,
        note=args.note,
        trial_index=args.trial_index,
        force_macimu=args.force_macimu,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
