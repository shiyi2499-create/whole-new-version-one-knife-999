"""
Data loaders for stage2_ctc.

Re-exports SessionLoader, NegativeLoader, discover_sessions from
stage2_episode. These are identical — we import rather than copy to
avoid drift.

If stage2_episode is not on sys.path, fall back to a self-contained
minimal copy of the essential classes.
"""
import sys
import os

try:
    raise ImportError("Use local fallback loader to avoid package shadowing")
    from data.loaders import SessionLoader, NegativeLoader, discover_sessions
except ImportError:
    # Fallback: self-contained minimal implementation
    import json
    import csv
    import numpy as np
    from pathlib import Path
    from typing import List, Optional

    class SessionLoader:
        """Load a single recording session."""

        def __init__(self, session_path: str):
            self.path = Path(session_path)
            self.is_dir = self.path.is_dir()

        def _read(self, name):
            if self.is_dir:
                p = self.path / name
            else:
                p = self.path.parent / f"{self.path.name}_{name}"
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
                p = self.path.parent / f"{self.path.name}_{name}"
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
            rows, columns = self._read('sensor.csv')
            if not rows:
                return np.array([]), np.zeros((0, 6))
            tcol = self._col(columns, ['timestamp', 'timestamp_ns', 'time', 'ts'])
            if tcol is None:
                tcol = columns[0]
            acols = self._col(columns, [('accel_x', 'accel_y', 'accel_z'),
                                    ('acc_x', 'acc_y', 'acc_z')])
            gcols = self._col(columns, [('gyro_x', 'gyro_y', 'gyro_z'),
                                    ('gyr_x', 'gyr_y', 'gyr_z')])
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

        def get_activity_log(self):
            rows, columns = self._read('activity_log.csv')
            if not rows:
                return []
            out_rows = []
            for r in rows:
                row = {}
                for c in ['start_time_ns', 'start_ns']:
                    if c in columns and r.get(c) not in (None, ''):
                        row['start_ns'] = int(r[c]); break
                for c in ['end_time_ns', 'end_ns']:
                    if c in columns and r.get(c) not in (None, ''):
                        row['end_ns'] = int(r[c]); break
                for c in ['activity', 'label', 'typing_style', 'prompts']:
                    if c in columns:
                        row[c] = r[c]
                out_rows.append(row)
            return out_rows

        def get_protocol(self):
            return self._read_json('protocol.json')

        def get_password_block(self):
            rows = self.get_activity_log()
            for row in rows:
                if (str(row.get('activity', '')) == 'keyboard' and
                        str(row.get('typing_style', '')) == 'password'):
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
                if (seg.get('activity') == 'keyboard' and
                        seg.get('typing_style') == 'password'):
                    return {
                        'start_ns': None, 'end_ns': None,
                        'label': seg.get('label', 'typing_2'),
                        'prompts': seg.get('prompts', []) or [],
                    }
            return None

        def split_password_groups_from_enters(self):
            block = self.get_password_block()
            if block is None or block.get('start_ns') is None:
                return []
            events = [
                e for e in self.get_press_events()
                if block['start_ns'] <= e['ts'] <= block['end_ns']
            ]
            groups, current = [], []
            for e in events:
                key = str(e['key']).lower()
                if key in ('enter', 'return'):
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
            ts, imu = self.get_imu()
            attempts_rows, _ = self._read('attempts.csv')
            presses = self.get_press_events()
            segments = []
            for r in attempts_rows:
                s = int(r.get('attempt_start_ns', r.get('start_ns', 0)))
                e = int(r.get('submit_ns', r.get('end_ns', s + int(5e9))))
                mask = (ts >= s) & (ts <= e)
                idx = np.where(mask)[0]
                if len(idx) < 5:
                    continue
                seg_imu = imu[idx]
                seg_ts = ts[idx]
                onsets, chars = [], []
                for p in presses:
                    if s <= p['ts'] <= e:
                        si = min(int(np.searchsorted(seg_ts, p['ts'])), len(seg_ts) - 1)
                        onsets.append(si)
                        chars.append(p['key'])
                segments.append({
                    'imu': seg_imu, 'onsets': onsets, 'chars': chars,
                    'prompt': r.get('prompt_text', r.get('prompt', '')),
                    'num_keys': len(onsets),
                })
            return segments

    class NegativeLoader:
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
        for sf in sorted(dp.glob("*_sensor.csv")):
            name = sf.name
            if name.endswith("_sensor.csv"):
                flat_stems.add(str(sf.with_name(name[:-11])))
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
