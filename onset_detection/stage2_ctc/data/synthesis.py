"""
Synthetic episode generator for CTC training.

Adapted from stage2_episode/data/synthesis.py.
Key change: output includes per-frame character targets, not just onset Gaussians.
"""
import os
import sys
import importlib.util
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from data.loaders import NegativeLoader, discover_sessions, SessionLoader
try:
    from utils.signal_processing import time_stretch
    from utils.vocab import char_index, BLANK_IDX
except ModuleNotFoundError:
    def _load_local_module(mod_name: str, rel_path: str):
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(PKG_ROOT, rel_path)
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    _sig = _load_local_module('stage2_ctc_signal_processing', 'utils/signal_processing.py')
    _voc = _load_local_module('stage2_ctc_vocab', 'utils/vocab.py')
    time_stretch = _sig.time_stretch
    char_index = _voc.char_index
    BLANK_IDX = _voc.BLANK_IDX

from data.datasets import build_frame_targets
from configs.config import DataConfig


class CTCSynthesizer:

    def __init__(self, password_segments: List[Dict],
                 neg_loader: NegativeLoader,
                 config: DataConfig,
                 sample_rate: int = 100,
                 seed: int = 42):
        self.segments = password_segments
        self.neg = neg_loader
        self.cfg = config
        self.sr = sample_rate
        self.rng = np.random.RandomState(seed)
        self.sigma_frames = max(1.0, 20.0 / 1000.0 * sample_rate)

        self.by_nkeys = {}
        for seg in self.segments:
            nk = seg['num_keys']
            if nk < 1:
                continue
            self.by_nkeys.setdefault(nk, []).append(seg)
        self.all_valid = [s for s in self.segments if s['num_keys'] >= 1]
        print(f"CTCSynthesizer: {len(self.all_valid)} segments, "
              f"key counts: {sorted(self.by_nkeys.keys())}")

    def _sample_segment(self, desired_len: int) -> Dict:
        if desired_len in self.by_nkeys and self.by_nkeys[desired_len]:
            pool = self.by_nkeys[desired_len]
            seg = pool[self.rng.randint(len(pool))]
        else:
            seg = self.all_valid[self.rng.randint(len(self.all_valid))]

        seg = dict(seg)
        imu = seg['imu'].copy()
        onsets = list(seg['onsets'])
        chars = list(seg['chars'])

        lo, hi = self.cfg.time_stretch_range
        rate = self.rng.uniform(lo, hi)
        if abs(rate - 1.0) > 0.01:
            old_len = len(imu)
            imu = time_stretch(imu, rate)
            ratio = len(imu) / max(old_len, 1)
            onsets = [int(o * ratio) for o in onsets]

        lo, hi = self.cfg.gain_range
        imu = imu * self.rng.uniform(lo, hi)

        if self.cfg.noise_std > 0:
            imu = imu + self.rng.randn(*imu.shape).astype(np.float32) * self.cfg.noise_std

        if len(onsets) > desired_len:
            onsets = onsets[:desired_len]
            chars = chars[:desired_len]
            if onsets:
                end = min(len(imu), onsets[-1] + int(0.3 * self.sr))
                imu = imu[:end]

        seg['imu'] = imu
        seg['onsets'] = onsets
        seg['chars'] = chars
        seg['num_keys'] = len(onsets)
        return seg

    def generate_one_episode(self) -> Dict:
        """Generate one synthetic password episode with CTC targets."""
        pw_len = self.rng.randint(self.cfg.min_password_len,
                                  self.cfg.max_password_len + 1)
        seg = self._sample_segment(pw_len)
        imu = seg['imu']
        T = len(imu)

        key_events = [
            {'ts_frame': min(o, T - 1), 'char': c}
            for o, c in zip(seg['onsets'], seg['chars'])
        ]

        hard_targets, soft_weights, ctc_target = build_frame_targets(
            T, key_events, self.sigma_frames
        )
        password = ''.join(seg['chars'][:len(seg['onsets'])])

        return {
            'imu': imu.astype(np.float32),
            'hard_targets': hard_targets,
            'soft_weights': soft_weights,
            'ctc_target': np.array(ctc_target, dtype=np.int64),
            'password': password,
        }

    def generate_one_session(self) -> List[Dict]:
        """Generate multiple episodes for one synthetic session."""
        n_pw = self.rng.randint(self.cfg.min_passwords,
                                self.cfg.max_passwords + 1)
        return [self.generate_one_episode() for _ in range(n_pw)]

    def generate_dataset(self, num_sessions: int, output_dir: str,
                         splits=(0.7, 0.15, 0.15)):
        out = Path(output_dir)
        all_episodes = []
        for i in range(num_sessions):
            if (i + 1) % 100 == 0:
                print(f"  generating session {i + 1}/{num_sessions}")
            all_episodes.extend(self.generate_one_session())

        indices = list(range(len(all_episodes)))
        self.rng.shuffle(indices)
        n_train = int(len(all_episodes) * splits[0])
        n_val = int(len(all_episodes) * splits[1])

        split_map = {
            'train': indices[:n_train],
            'val': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:],
        }

        for split, idxs in split_map.items():
            d = out / split
            d.mkdir(parents=True, exist_ok=True)
            for j, idx in enumerate(idxs):
                ep = all_episodes[idx]
                np.savez_compressed(
                    d / f"episode_{j:05d}.npz",
                    imu=ep['imu'],
                    hard_targets=ep['hard_targets'],
                    soft_weights=ep['soft_weights'],
                    ctc_target=ep['ctc_target'],
                    password=ep['password'],
                )

        meta = {
            'num_sessions': num_sessions,
            'num_episodes': len(all_episodes),
            'splits': {k: len(v) for k, v in split_map.items()},
            'sample_rate': self.sr,
            'format': 'ctc_episode',
        }
        with open(out / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"Saved CTC dataset to {output_dir}: " +
              ", ".join(f"{k}={len(v)}" for k, v in split_map.items()))
        return meta


def load_all_segments(password_dir: str) -> List[Dict]:
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
