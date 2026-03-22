"""
Data loaders for password sessions and negative clips.
Supports variable-length passwords.
"""
import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class SessionLoader:
    """Load a single recording session (sensor.csv + events.csv + attempts.csv)."""

    def __init__(self, session_path: str):
        self.path = Path(session_path)
        self.is_dir = self.path.is_dir()
        self.stem = self.path if not self.is_dir else None

    def _read(self, name):
        if self.is_dir:
            p = self.path / name
        else:
            p = self.stem.parent / f"{self.stem.name}_{name}"
        if not p.exists():
            return [], []
        with open(p, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows, list(reader.fieldnames or [])

    def _read_json(self, name):
        if self.is_dir:
            p = self.path / name
        else:
            p = self.stem.parent / f"{self.stem.name}_{name}"
        if not p.exists():
            return {}
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _col(self, columns, candidates):
        for c in candidates:
            if isinstance(c, (list, tuple)):
                if all(x in columns for x in c):
                    return c
            elif c in columns:
                return c
        return None

    def get_imu(self):
        """Returns (timestamps_ns [T], data [T,6])."""
        rows, columns = self._read('sensor.csv')
        if not rows:
            return np.array([]), np.zeros((0, 6))
        tcol = self._col(columns, ['timestamp', 'timestamp_ns', 'time', 'ts'])
        if tcol is None:
            tcol = columns[0]
        acols = self._col(columns, [('accel_x', 'accel_y', 'accel_z'),
                                ('acc_x', 'acc_y', 'acc_z'), ('ax', 'ay', 'az')])
        gcols = self._col(columns, [('gyro_x', 'gyro_y', 'gyro_z'),
                                ('gyr_x', 'gyr_y', 'gyr_z'), ('gx', 'gy', 'gz')])
        if acols is None:
            acols = tuple(columns[1:4])
        if gcols is None:
            gcols = tuple(columns[4:7])
        ts = np.array([int(r[tcol]) for r in rows], dtype=np.int64)
        data = np.array(
            [[float(r[c]) for c in list(acols) + list(gcols)] for r in rows],
            dtype=np.float32,
        )
        return ts, data

    def get_press_events(self):
        """Returns list of {'ts': int, 'key': str}."""
        rows, columns = self._read('events.csv')
        if not rows:
            return []
        events = []
        for r in rows:
            etype = ''
            for c in ['type', 'event_type', 'action']:
                if c in columns:
                    etype = str(r[c]).lower()
                    break
            if etype not in ('press', 'keydown', 'down', 'p'):
                continue
            ts = None
            for c in ['timestamp_ns', 'timestamp', 'time', 'ts']:
                if c in columns:
                    ts = int(r[c])
                    break
            key = '?'
            for c in ['key', 'key_char', 'char']:
                if c in columns:
                    key = str(r[c])
                    break
            if ts is not None:
                events.append({'ts': ts, 'key': key})
        return events

    def get_attempts(self):
        """Returns list of {'start_ns', 'end_ns', 'prompt', 'typed'}."""
        rows, columns = self._read('attempts.csv')
        if not rows:
            return []
        attempts = []
        for r in rows:
            a = {}
            for c in ['attempt_start_ns', 'start_ns', 'start']:
                if c in columns and r.get(c) not in (None, ''):
                    a['start_ns'] = int(r[c]); break
            for c in ['submit_ns', 'end_ns', 'end']:
                if c in columns and r.get(c) not in (None, ''):
                    a['end_ns'] = int(r[c]); break
            for c in ['prompt_text', 'prompt', 'target']:
                if c in columns:
                    a['prompt'] = str(r[c]); break
            for c in ['typed_text', 'typed', 'input']:
                if c in columns:
                    a['typed'] = str(r[c]); break
            if 'start_ns' in a:
                attempts.append(a)
        return attempts

    def get_activity_log(self):
        """Returns activity log rows if available."""
        rows, columns = self._read('activity_log.csv')
        if not rows:
            return []
        out_rows = []
        for r in rows:
            row = {}
            for c in ['start_time_ns', 'start_ns', 'start']:
                if c in columns and r.get(c) not in (None, ''):
                    row['start_ns'] = int(r[c])
                    break
            for c in ['end_time_ns', 'end_ns', 'end']:
                if c in columns and r.get(c) not in (None, ''):
                    row['end_ns'] = int(r[c])
                    break
            for c in ['activity', 'label', 'typing_style', 'prompts']:
                if c in columns:
                    row[c] = r[c]
            out_rows.append(row)
        return out_rows

    def get_protocol(self):
        """Returns parsed protocol.json when present."""
        return self._read_json('protocol.json')

    def get_password_block(self):
        """
        Returns a password typing block dict if present.

        For mixed2 / mixed_training this is the 'typing_2' keyboard segment
        with typing_style=password.
        """
        rows = self.get_activity_log()
        for row in rows:
            if str(row.get('activity', '')) == 'keyboard' and str(row.get('typing_style', '')) == 'password':
                out = {
                    'start_ns': row.get('start_ns'),
                    'end_ns': row.get('end_ns'),
                    'label': row.get('label', 'typing_2'),
                }
                prompts = row.get('prompts')
                if isinstance(prompts, str) and prompts.strip():
                    try:
                        out['prompts'] = json.loads(prompts)
                    except Exception:
                        out['prompts'] = []
                else:
                    out['prompts'] = []
                return out

        protocol = self.get_protocol()
        for seg in protocol.get('protocol', []):
            if seg.get('activity') == 'keyboard' and seg.get('typing_style') == 'password':
                return {
                    'start_ns': None,
                    'end_ns': None,
                    'label': seg.get('label', 'typing_2'),
                    'prompts': seg.get('prompts', []) or [],
                }
        return None

    def split_password_groups_from_enters(self):
        """
        Split a password block into groups using Enter presses as separators.

        Returns list of groups:
        {
            'start_ns', 'end_ns', 'keys': [{'ts','key'}, ...], 'num_keys'
        }
        """
        block = self.get_password_block()
        if block is None or block.get('start_ns') is None or block.get('end_ns') is None:
            return []

        events = [
            e for e in self.get_press_events()
            if block['start_ns'] <= e['ts'] <= block['end_ns']
        ]
        groups = []
        current = []

        for e in events:
            key = str(e['key']).lower()
            if key == 'enter':
                if current:
                    groups.append({
                        'start_ns': current[0]['ts'],
                        'end_ns': current[-1]['ts'],
                        'keys': current[:],
                        'num_keys': len(current),
                    })
                    current = []
                continue
            current.append(e)

        if current:
            groups.append({
                'start_ns': current[0]['ts'],
                'end_ns': current[-1]['ts'],
                'keys': current[:],
                'num_keys': len(current),
            })

        return groups

    def extract_attempt_segments(self):
        """
        Returns list of {
            'imu': [T, 6], 'onsets': [list of sample idx],
            'chars': [list of str], 'prompt': str, 'num_keys': int
        }
        """
        ts, imu = self.get_imu()
        attempts = self.get_attempts()
        presses = self.get_press_events()
        segments = []
        for att in attempts:
            s = att['start_ns']
            e = att.get('end_ns', s + int(5e9))
            mask = (ts >= s) & (ts <= e)
            idx = np.where(mask)[0]
            if len(idx) < 5:
                continue
            seg_imu = imu[idx]
            seg_ts = ts[idx]
            onsets, chars = [], []
            for p in presses:
                if s <= p['ts'] <= e:
                    si = int(np.searchsorted(seg_ts, p['ts']))
                    si = min(si, len(seg_ts) - 1)
                    onsets.append(si)
                    chars.append(p['key'])
            segments.append({
                'imu': seg_imu, 'onsets': onsets, 'chars': chars,
                'prompt': att.get('prompt', ''),
                'num_keys': len(onsets),
            })
        return segments


class NegativeLoader:
    """Load negative clips from idle/trackpad/shake/freetyping dirs."""

    def __init__(self, neg_dir: str):
        self.clips = []
        nd = Path(neg_dir)
        if not nd.exists():
            return
        for sess in discover_sessions(str(nd)):
            try:
                loader = SessionLoader(sess)
                _, d = loader.get_imu()
                if len(d) > 10:
                    self.clips.append(d)
            except Exception:
                pass

    def sample(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        if not self.clips:
            return rng.randn(n_samples, 6).astype(np.float32) * 0.01
        clip = self.clips[rng.randint(len(self.clips))]
        if len(clip) <= n_samples:
            clip = np.tile(clip, (n_samples // len(clip) + 1, 1))
        start = rng.randint(0, len(clip) - n_samples)
        return clip[start:start + n_samples].copy()


def discover_sessions(data_dir: str) -> List[str]:
    dp = Path(data_dir)
    out = []
    if not dp.exists():
        return out

    flat_stems = set()
    for sensor_file in sorted(dp.glob("*_sensor.csv")):
        name = sensor_file.name
        if not name.endswith("_sensor.csv"):
            continue
        flat_stems.add(str(sensor_file.with_name(name[:-11])))

    for item in sorted(dp.iterdir()):
        if item.is_dir():
            if (item / 'sensor.csv').exists():
                out.append(str(item))
            else:
                for sub in sorted(item.iterdir()):
                    if sub.is_dir() and (sub / 'sensor.csv').exists():
                        out.append(str(sub))
                    elif sub.is_file() and sub.name.endswith("_sensor.csv"):
                        flat_stems.add(str(sub.with_name(sub.name[:-11])))

    out.extend(sorted(flat_stems))
    return sorted(set(out))
