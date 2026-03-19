"""
Variable-length synthetic session generator for episode-based Stage 2.

Frame-level labels (unchanged):
  0 = silence, 1 = typing (full episode interior, first key to last key).

NEW in this version:
  onset_targets: [T] float32 Gaussian-smoothed impulse target.
    Each key onset contributes a Gaussian bump with sigma = onset_sigma_ms
    (default 20ms). This is the supervision signal for the dual-head TCN's
    onset_head.

    Why Gaussian, not 0/1 spike?
      - Avoids near-all-zeros BCE imbalance (1 spike per ~100 silence frames).
      - Gaussian shape = soft spatial uncertainty → easier to optimise with MSE.
      - The resulting onset_head output is directly peak-pickable by find_peaks.
      - This is equivalent to how BMN / AudioSet event detection trains their
        temporal localization heads.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from data.loaders import NegativeLoader
from utils.signal_processing import splice_smooth, time_stretch
from configs.config import SynthesisConfig


class EpisodeSynthesizer:

    def __init__(self, password_segments: List[Dict],
                 neg_loader: NegativeLoader,
                 config: SynthesisConfig,
                 sample_rate: int = 100,
                 seed: int = 42):
        """
        password_segments: list of dicts with 'imu', 'onsets', 'chars', 'num_keys'
        """
        self.segments = password_segments
        self.neg = neg_loader
        self.cfg = config
        self.sr = sample_rate
        self.rng = np.random.RandomState(seed)

        self.by_nkeys = {}
        for seg in self.segments:
            nk = seg['num_keys']
            if nk < 1:
                continue
            self.by_nkeys.setdefault(nk, []).append(seg)

        self.all_valid = [s for s in self.segments if s['num_keys'] >= 1]
        print(f"EpisodeSynthesizer: {len(self.all_valid)} segments, "
              f"key counts: {sorted(self.by_nkeys.keys())}")

    def _sample_segment(self, desired_len: int) -> Dict:
        """Sample a password segment, preferring exact key-count match."""
        if desired_len in self.by_nkeys and self.by_nkeys[desired_len]:
            seg = self.by_nkeys[desired_len][
                self.rng.randint(len(self.by_nkeys[desired_len]))
            ]
        else:
            seg = self.all_valid[self.rng.randint(len(self.all_valid))]

        seg = dict(seg)
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

        # Truncate to desired_len keys
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

    def generate_one(self) -> Dict:
        """
        Generate one variable-length synthetic session with 2-class labels.

        Returns {
            'imu': [T, 6],
            'frame_labels': [T] int array (0=silence, 1=typing),
            'episodes': list of {'start', 'end', 'onsets', 'chars'},
            'num_passwords': int,
            'password_lengths': list of int,
        }
        """
        cfg = self.cfg

        n_pw = self.rng.randint(cfg.min_passwords, cfg.max_passwords + 1)
        pw_lengths = [self.rng.randint(cfg.min_password_len, cfg.max_password_len + 1)
                      for _ in range(n_pw)]
        pw_segs = [self._sample_segment(L) for L in pw_lengths]

        # Inter-password gap clips (silence)
        n_gaps = n_pw - 1
        gap_durations_s = self.rng.uniform(
            cfg.inter_pw_gap_min_s, cfg.inter_pw_gap_max_s, size=max(n_gaps, 0))
        gap_clips = [self.neg.sample(int(d * self.sr), self.rng) for d in gap_durations_s]

        # Context prefix/suffix (silence)
        pre_dur = self.rng.uniform(cfg.context_min_s, cfg.context_max_s)
        suf_dur = self.rng.uniform(cfg.context_min_s, cfg.context_max_s)
        prefix = self.neg.sample(int(pre_dur * self.sr), self.rng)
        suffix = self.neg.sample(int(suf_dur * self.sr), self.rng)

        # Assemble: prefix + [pw_0, gap_0, pw_1, gap_1, ..., pw_N-1] + suffix
        # ALL non-password parts get label 0 (silence)
        # Password keystroke regions get label 1 (typing)
        parts = []
        parts.append((prefix, 'silence', None))
        for i in range(n_pw):
            parts.append((pw_segs[i]['imu'], 'password', i))
            if i < n_gaps:
                parts.append((gap_clips[i], 'silence', None))
        parts.append((suffix, 'silence', None))

        episode_margin = int(cfg.keystroke_label_radius_ms / 1000.0 * self.sr)

        episodes_info = []
        all_imu = []
        all_labels = []
        global_offset = 0

        for imu_chunk, ptype, pw_idx in parts:
            T_chunk = len(imu_chunk)
            labels_chunk = np.zeros(T_chunk, dtype=np.int64)  # default: silence

            if ptype == 'password' and pw_idx is not None:
                seg = pw_segs[pw_idx]
                # Mark the whole password episode interior as typing=1.
                if seg['onsets']:
                    lo = max(0, seg['onsets'][0] - episode_margin)
                    hi = min(T_chunk, seg['onsets'][-1] + episode_margin + 1)
                    labels_chunk[lo:hi] = 1

                episodes_info.append({
                    'start': global_offset,
                    'end': global_offset + T_chunk,
                    'onsets': [o + global_offset for o in seg['onsets']],
                    'chars': seg['chars'][:len(seg['onsets'])],
                })

            all_imu.append(imu_chunk)
            all_labels.append(labels_chunk)
            global_offset += T_chunk

        full_imu = np.concatenate(all_imu, axis=0).astype(np.float32)
        full_labels = np.concatenate(all_labels, axis=0)

        # --- Build per-frame Gaussian onset targets (NEW) ---
        # Each onset gets a Gaussian bump: exp(-0.5 * (t - t_k)^2 / sigma^2)
        # sigma_ms is deliberately narrow so find_peaks can pick the center.
        # We use sigma = onset_gaussian_sigma_ms from config (default 20ms @100Hz = 2 frames).
        # This gives the onset_head a *learnable* target that is peak-pickable,
        # unlike the typing plateau which has no per-key structure.
        sigma_frames = max(1.0, cfg.onset_gaussian_sigma_ms / 1000.0 * self.sr)
        T_total = len(full_labels)
        onset_targets = np.zeros(T_total, dtype=np.float32)
        t = np.arange(T_total, dtype=np.float32)
        for ep_info in episodes_info:
            for onset_global in ep_info['onsets']:
                onset_targets += np.exp(
                    -0.5 * ((t - onset_global) / sigma_frames) ** 2
                )
        # Clip at 1.0: overlapping Gaussians (close keypresses) saturate gracefully
        onset_targets = np.clip(onset_targets, 0.0, 1.0).astype(np.float32)

        return {
            'imu': full_imu,
            'frame_labels': full_labels,
            'onset_targets': onset_targets,
            'episodes': episodes_info,
            'num_passwords': n_pw,
            'password_lengths': [len(ep['onsets']) for ep in episodes_info],
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
                    onset_targets=s['onset_targets'],
                    episodes_json=json.dumps(s['episodes']),
                    num_passwords=s['num_passwords'],
                    password_lengths=np.array(s['password_lengths']),
                )

        meta = {
            'num_sessions': num_sessions,
            'splits': {k: len(v) for k, v in split_map.items()},
            'sample_rate': self.sr,
            'label_scheme': '2-class: 0=silence, 1=typing + onset_targets (Gaussian)',
            'config': {
                'min_passwords': self.cfg.min_passwords,
                'max_passwords': self.cfg.max_passwords,
                'min_password_len': self.cfg.min_password_len,
                'max_password_len': self.cfg.max_password_len,
                'inter_pw_gap_min_s': self.cfg.inter_pw_gap_min_s,
                'inter_pw_gap_max_s': self.cfg.inter_pw_gap_max_s,
                'onset_gaussian_sigma_ms': self.cfg.onset_gaussian_sigma_ms,
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
