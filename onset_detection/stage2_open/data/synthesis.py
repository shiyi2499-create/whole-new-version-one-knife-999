"""
Variable-length synthetic session generator.

Each session has:
  - N passwords, N ~ Uniform[min_passwords, max_passwords]
  - Each password has L keys, L ~ Uniform[min_password_len, max_password_len]
  - Gaps between passwords: random duration
  - Context before/after: random negative clips

Frame-level labels:
  0 = gap (no keystroke activity, or inter-keystroke gap within a password)
  1 = keystroke (within radius_ms of a key press)
  2 = separator (inter-password pause)

This is the core training data for the open Stage 2.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from data.loaders import NegativeLoader
from utils.signal_processing import splice_smooth, time_stretch
from configs.config import SynthesisConfig


class OpenSynthesizer:

    def __init__(self, password_segments: List[Dict],
                 neg_loader: NegativeLoader,
                 config: SynthesisConfig,
                 sample_rate: int = 100,
                 seed: int = 42):
        """
        password_segments: list of dicts with 'imu', 'onsets', 'chars', 'num_keys'
                           from SessionLoader.extract_attempt_segments()
                           These may have DIFFERENT lengths.
        """
        self.segments = password_segments
        self.neg = neg_loader
        self.cfg = config
        self.sr = sample_rate
        self.rng = np.random.RandomState(seed)

        # Group segments by number of keys for efficient sampling
        self.by_nkeys = {}
        for seg in self.segments:
            nk = seg['num_keys']
            if nk < 1:
                continue
            self.by_nkeys.setdefault(nk, []).append(seg)

        self.all_valid = [s for s in self.segments if s['num_keys'] >= 1]
        print(f"OpenSynthesizer: {len(self.all_valid)} segments, "
              f"key counts: {sorted(self.by_nkeys.keys())}")

    def _sample_segment(self, desired_len: int) -> Dict:
        """
        Sample a password segment. If we have one with exactly desired_len keys, prefer it.
        Otherwise take any segment and truncate/pad the onset list.
        """
        # Try exact match first
        if desired_len in self.by_nkeys and self.by_nkeys[desired_len]:
            seg = self.by_nkeys[desired_len][
                self.rng.randint(len(self.by_nkeys[desired_len]))
            ]
        else:
            # Take any segment
            seg = self.all_valid[self.rng.randint(len(self.all_valid))]

        seg = dict(seg)  # shallow copy
        imu = seg['imu'].copy()
        onsets = list(seg['onsets'])
        chars = list(seg['chars'])

        # Time stretch augmentation
        lo, hi = self.cfg.time_stretch_range
        rate = self.rng.uniform(lo, hi)
        if abs(rate - 1.0) > 0.01:
            old_len = len(imu)
            imu = time_stretch(imu, rate)
            ratio = len(imu) / max(old_len, 1)
            onsets = [int(o * ratio) for o in onsets]

        # Gain augmentation
        lo, hi = self.cfg.gain_range
        imu = imu * self.rng.uniform(lo, hi)

        # Noise
        if self.cfg.noise_std > 0:
            imu = imu + self.rng.randn(*imu.shape).astype(np.float32) * self.cfg.noise_std

        # Truncate to desired_len keys if we have more
        if len(onsets) > desired_len:
            onsets = onsets[:desired_len]
            chars = chars[:desired_len]
            # Trim IMU to just after last onset + small buffer
            if onsets:
                end = min(len(imu), onsets[-1] + int(0.3 * self.sr))
                imu = imu[:end]

        seg['imu'] = imu
        seg['onsets'] = onsets
        seg['chars'] = chars
        seg['num_keys'] = len(onsets)
        return seg

    def generate_one(self) -> Dict:
        """
        Generate one variable-length synthetic session.

        Returns {
            'imu': [T, 6],
            'frame_labels': [T] int array (0=gap, 1=keystroke, 2=separator),
            'groups': list of {'start': int, 'end': int, 'onsets': list, 'chars': list},
            'num_passwords': int,
            'password_lengths': list of int,
        }
        """
        cfg = self.cfg

        # Random number of passwords
        n_pw = self.rng.randint(cfg.min_passwords, cfg.max_passwords + 1)

        # Random length for each password
        pw_lengths = [self.rng.randint(cfg.min_password_len, cfg.max_password_len + 1)
                      for _ in range(n_pw)]

        # Sample password segments
        pw_segs = [self._sample_segment(L) for L in pw_lengths]

        # Sample inter-password separator clips
        n_seps = n_pw - 1
        sep_durations_s = self.rng.uniform(cfg.gap_min_s, cfg.gap_max_s, size=max(n_seps, 0))
        sep_clips = [self.neg.sample(int(d * self.sr), self.rng) for d in sep_durations_s]

        # Context prefix/suffix
        pre_dur = self.rng.uniform(cfg.context_min_s, cfg.context_max_s)
        suf_dur = self.rng.uniform(cfg.context_min_s, cfg.context_max_s)
        prefix = self.neg.sample(int(pre_dur * self.sr), self.rng)
        suffix = self.neg.sample(int(suf_dur * self.sr), self.rng)

        # Assemble: prefix + [pw_0, sep_0, pw_1, sep_1, ..., pw_N-1] + suffix
        parts = []      # (imu_array, label_type)
        # label_type: 'gap' | 'password' | 'separator'

        parts.append((prefix, 'gap'))
        for i in range(n_pw):
            parts.append((pw_segs[i]['imu'], 'password', i))
            if i < n_seps:
                parts.append((sep_clips[i], 'separator'))
        parts.append((suffix, 'gap'))

        # Concatenate and build frame labels
        smooth = cfg.splice_smooth_samples
        all_imu_chunks = []
        all_labels_chunks = []

        keystroke_radius = int(cfg.keystroke_label_radius_ms / 1000.0 * self.sr)

        groups_info = []
        global_offset = 0

        for part_info in parts:
            if len(part_info) == 3:
                imu_chunk, ptype, pw_idx = part_info
            else:
                imu_chunk, ptype = part_info
                pw_idx = None

            T_chunk = len(imu_chunk)
            labels_chunk = np.zeros(T_chunk, dtype=np.int64)

            if ptype == 'separator':
                labels_chunk[:] = 2
            elif ptype == 'password' and pw_idx is not None:
                seg = pw_segs[pw_idx]
                # Default: gap (0) within the password segment (inter-keystroke)
                # Mark keystroke regions as 1
                for onset in seg['onsets']:
                    lo = max(0, onset - keystroke_radius)
                    hi = min(T_chunk, onset + keystroke_radius + 1)
                    labels_chunk[lo:hi] = 1

                groups_info.append({
                    'start': global_offset,
                    'end': global_offset + T_chunk,
                    'onsets': [o + global_offset for o in seg['onsets']],
                    'chars': seg['chars'][:len(seg['onsets'])],
                })
            # else: 'gap' → stays 0

            all_imu_chunks.append(imu_chunk)
            all_labels_chunks.append(labels_chunk)
            global_offset += T_chunk

        # Concatenate (simple — splice_smooth for signal quality is optional here)
        full_imu = np.concatenate(all_imu_chunks, axis=0).astype(np.float32)
        full_labels = np.concatenate(all_labels_chunks, axis=0)

        return {
            'imu': full_imu,
            'frame_labels': full_labels,
            'groups': groups_info,
            'num_passwords': n_pw,
            'password_lengths': [len(g['onsets']) for g in groups_info],
        }

    def generate_dataset(self, num_sessions: int, output_dir: str,
                         splits=(0.7, 0.15, 0.15)):
        """Generate and save full dataset with train/val/test splits."""
        out = Path(output_dir)
        sessions = []
        for i in range(num_sessions):
            if (i + 1) % 50 == 0:
                print(f"  generating {i + 1}/{num_sessions}")
            sessions.append(self.generate_one())

        indices = list(range(num_sessions))
        self.rng.shuffle(indices)
        n_train = int(num_sessions * splits[0])
        n_val = int(num_sessions * splits[1])

        split_map = {
            'train': indices[:n_train],
            'val': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:],
        }

        for split, idxs in split_map.items():
            d = out / split
            d.mkdir(parents=True, exist_ok=True)
            for j, idx in enumerate(idxs):
                s = sessions[idx]
                np.savez_compressed(
                    d / f"session_{j:04d}.npz",
                    imu=s['imu'],
                    frame_labels=s['frame_labels'],
                    groups_json=json.dumps(s['groups']),
                    num_passwords=s['num_passwords'],
                    password_lengths=np.array(s['password_lengths']),
                )

        meta = {
            'num_sessions': num_sessions,
            'splits': {k: len(v) for k, v in split_map.items()},
            'sample_rate': self.sr,
            'config': {
                'min_passwords': self.cfg.min_passwords,
                'max_passwords': self.cfg.max_passwords,
                'min_password_len': self.cfg.min_password_len,
                'max_password_len': self.cfg.max_password_len,
            },
        }
        with open(out / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Saved to {output_dir}: " +
              ", ".join(f"{k}={len(v)}" for k, v in split_map.items()))
        return meta


def load_all_segments(password_dir: str) -> List[Dict]:
    """Load all attempt segments from all sessions under password_dir."""
    from data.loaders import discover_sessions, SessionLoader
    sessions = discover_sessions(password_dir)
    all_segs = []
    for sp in sessions:
        try:
            segs = SessionLoader(sp).extract_attempt_segments()
            all_segs.extend(segs)
        except Exception as e:
            print(f"  warn: {sp}: {e}")
    print(f"Loaded {len(all_segs)} segments from {len(sessions)} sessions")
    return all_segs
