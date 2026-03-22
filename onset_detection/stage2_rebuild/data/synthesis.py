"""
Synthetic mixed session generator.

Creates training data for Stage 2A and 2B by splicing together:
- Password attempts from password/len_8
- Negative clips from onset_negative (idle, trackpad, shake, freetyping)

This simulates the structure of mixed2 without needing real mixed2 data.
"""
import os
import numpy as np
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from data.loaders import (
    NegativeDataLoader,
    load_all_password_blocks as _load_password_blocks,
    load_all_password_segments as _load_password_segments,
)
from utils.signal_processing import time_stretch
from configs.config import SynthesisConfig


class SyntheticMixedGenerator:
    """
    Generate synthetic mixed sessions that mimic mixed2 structure:
        [context] [password_1] [gap_1] [password_2] [gap_2] ... [password_5] [context]
    """

    def __init__(self,
                 password_segments: List[Dict],
                 negative_loader: NegativeDataLoader,
                 config: SynthesisConfig,
                 sample_rate: int = 190,
                 seed: int = 42):
        """
        Args:
            password_segments: list of dicts from PasswordSessionLoader.extract_attempt_segments()
            negative_loader: for sampling gap/context clips
            config: synthesis parameters
            sample_rate: IMU sample rate in Hz
            seed: random seed
        """
        self.password_segments = password_segments
        self.negative_loader = negative_loader
        self.config = config
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)

        if len(self.password_segments) == 0:
            raise ValueError("No password segments provided!")

        print(f"SyntheticMixedGenerator: {len(self.password_segments)} password segments available")

    def _sample_password_segments(self, n: int = 5) -> List[Dict]:
        """Sample n password segments, with optional time stretching."""
        indices = self.rng.choice(len(self.password_segments), size=n, replace=True)
        segments = []

        for idx in indices:
            seg = self.password_segments[idx].copy()
            imu = seg['imu'].copy()
            onsets = list(seg['key_onsets'])

            # Random time stretch
            lo, hi = self.config.time_stretch_range
            stretch_rate = self.rng.uniform(lo, hi)
            if abs(stretch_rate - 1.0) > 0.01:
                old_len = len(imu)
                imu = time_stretch(imu, stretch_rate)
                new_len = len(imu)
                # Adjust onset positions
                ratio = new_len / max(old_len, 1)
                onsets = [int(o * ratio) for o in onsets]

            # Random gain
            lo, hi = self.config.gain_augment_range
            gain = self.rng.uniform(lo, hi)
            imu = imu * gain

            # Random noise
            if self.config.noise_std > 0:
                imu = imu + self.rng.randn(*imu.shape).astype(np.float32) * self.config.noise_std

            seg['imu'] = imu
            seg['key_onsets'] = onsets
            segments.append(seg)

        return segments

    def generate_one(self) -> Dict:
        """
        Generate a single synthetic mixed session.

        Returns:
            {
                'imu': np.ndarray [T_total, 6],
                'group_labels': np.ndarray [T_total] binary (1=typing, 0=gap/context),
                'group_boundaries': list of (start, end) sample indices for each password group,
                'onset_positions': list of lists - onset sample indices per group (global coords),
                'onset_chars': list of lists - character labels per group,
                'num_groups': int,
                'keys_per_group': int,
            }
        """
        cfg = self.config
        n_pw = cfg.passwords_per_session

        # Sample password segments
        pw_segments = self._sample_password_segments(n_pw)

        # Sample gap clips (between passwords)
        n_gaps = n_pw - 1
        gap_durations_s = self.rng.uniform(cfg.gap_duration_min, cfg.gap_duration_max, size=n_gaps)
        gap_clips = []
        for dur in gap_durations_s:
            dur_samples = int(dur * self.sample_rate)
            clip = self.negative_loader.sample_clip(dur_samples, self.rng)
            gap_clips.append(clip)

        # Sample context (prefix and suffix)
        prefix_dur = self.rng.uniform(cfg.context_duration_min, cfg.context_duration_max)
        suffix_dur = self.rng.uniform(cfg.context_duration_min, cfg.context_duration_max)
        prefix = self.negative_loader.sample_clip(int(prefix_dur * self.sample_rate), self.rng)
        suffix = self.negative_loader.sample_clip(int(suffix_dur * self.sample_rate), self.rng)

        # Assemble the full session
        parts = []
        part_types = []  # 'context', 'password', 'gap'
        smooth = cfg.splice_smooth_samples

        # Prefix
        parts.append(prefix)
        part_types.append('context')

        for i in range(n_pw):
            parts.append(pw_segments[i]['imu'])
            part_types.append('password')

            if i < n_gaps:
                parts.append(gap_clips[i])
                part_types.append('gap')

        # Suffix
        parts.append(suffix)
        part_types.append('context')

        # Track labels on the simple-concat axis first, then smooth the signal itself.
        full_imu = np.concatenate(parts, axis=0)
        T = len(full_imu)

        # Build labels
        group_labels = np.zeros(T, dtype=np.float32)
        group_boundaries = []
        all_onsets = []
        all_chars = []

        offset = 0
        pw_idx = 0
        for i, (part, ptype) in enumerate(zip(parts, part_types)):
            part_len = len(part)
            if ptype == 'password':
                # Mark as typing
                group_labels[offset:offset + part_len] = 1.0

                # Record boundary
                group_boundaries.append((offset, offset + part_len))

                # Record onsets (offset to global coords)
                seg = pw_segments[pw_idx]
                global_onsets = [o + offset for o in seg['key_onsets']
                                 if 0 <= o < part_len]
                all_onsets.append(global_onsets)
                all_chars.append(seg.get('key_chars', ['?'] * len(global_onsets)))
                pw_idx += 1

            offset += part_len

        # Apply a short boundary smoothing pass without changing label coordinates.
        if smooth > 0 and len(parts) > 1:
            offset = len(parts[0])
            fade = np.linspace(0.0, 1.0, smooth, dtype=np.float32)[:, None]
            for part in parts[1:]:
                if 0 < smooth <= min(offset, len(part)):
                    left = full_imu[offset - smooth:offset].copy()
                    right = full_imu[offset:offset + smooth].copy()
                    blended = left * (1.0 - fade) + right * fade
                    full_imu[offset - smooth:offset] = blended
                offset += len(part)

        return {
            'imu': full_imu.astype(np.float32),
            'group_labels': group_labels,
            'group_boundaries': group_boundaries,
            'onset_positions': all_onsets,
            'onset_chars': all_chars,
            'num_groups': n_pw,
            'keys_per_group': cfg.keys_per_password,
        }

    def generate_dataset(self, num_sessions: int,
                         output_dir: str,
                         split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
        """
        Generate and save a full synthetic dataset.

        Saves to:
            output_dir/
                train/session_000.npz, session_001.npz, ...
                val/session_000.npz, ...
                test/session_000.npz, ...
                metadata.json
        """
        output_path = Path(output_dir)

        sessions = []
        for i in range(num_sessions):
            if (i + 1) % 20 == 0:
                print(f"  Generating session {i + 1}/{num_sessions}")
            sessions.append(self.generate_one())

        # Shuffle and split
        indices = list(range(num_sessions))
        self.rng.shuffle(indices)

        n_train = int(num_sessions * split_ratios[0])
        n_val = int(num_sessions * split_ratios[1])

        splits = {
            'train': indices[:n_train],
            'val': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:],
        }

        metadata = {
            'num_sessions': num_sessions,
            'splits': {k: len(v) for k, v in splits.items()},
            'config': {
                'passwords_per_session': self.config.passwords_per_session,
                'keys_per_password': self.config.keys_per_password,
                'sample_rate': self.sample_rate,
            }
        }

        for split_name, split_indices in splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for j, idx in enumerate(split_indices):
                s = sessions[idx]
                save_path = split_dir / f"session_{j:04d}.npz"
                np.savez_compressed(
                    save_path,
                    imu=s['imu'],
                    group_labels=s['group_labels'],
                    group_boundaries=np.array(s['group_boundaries']),
                    # Save onset positions as padded array
                    # Shape: [num_groups, max_onsets_per_group]
                    onset_positions=_pad_onset_lists(s['onset_positions']),
                    onset_chars=json.dumps(s['onset_chars']),
                    num_groups=s['num_groups'],
                    keys_per_group=s['keys_per_group'],
                )

        # Save metadata
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to {output_dir}")
        print(f"  Train: {len(splits['train'])} sessions")
        print(f"  Val:   {len(splits['val'])} sessions")
        print(f"  Test:  {len(splits['test'])} sessions")

        return metadata


def _pad_onset_lists(onset_lists: List[List[int]], pad_value: int = -1) -> np.ndarray:
    """Pad variable-length onset lists into a fixed 2D array."""
    if not onset_lists:
        return np.array([[]], dtype=np.int64)
    max_len = max(len(lst) for lst in onset_lists)
    padded = np.full((len(onset_lists), max(max_len, 1)), pad_value, dtype=np.int64)
    for i, lst in enumerate(onset_lists):
        for j, v in enumerate(lst):
            padded[i, j] = v
    return padded


def _apply_boundary_smoothing(full_imu: np.ndarray, part_lengths: List[int], smooth: int) -> np.ndarray:
    """Blend signal transitions while keeping label coordinates unchanged."""
    if smooth <= 0 or len(part_lengths) <= 1:
        return full_imu
    offset = part_lengths[0]
    fade = np.linspace(0.0, 1.0, smooth, dtype=np.float32)[:, None]
    for part_len in part_lengths[1:]:
        if 0 < smooth <= min(offset, part_len):
            left = full_imu[offset - smooth:offset].copy()
            right = full_imu[offset:offset + smooth].copy()
            blended = left * (1.0 - fade) + right * fade
            full_imu[offset - smooth:offset] = blended
        offset += part_len
    return full_imu


def load_all_password_segments(password_dir: str, target_rate_hz: int = 190) -> List[Dict]:
    return _load_password_segments(password_dir, target_rate_hz=target_rate_hz)


class BlockTemplateGenerator:
    """
    Generate pseudo-mixed sessions from real consecutive 5-password blocks.

    Compared with the original segment splicing strategy, this keeps the true
    within-session rhythm between neighboring passwords and only adds light
    contamination around the outside of the block.
    """

    def __init__(
        self,
        password_blocks: List[Dict],
        negative_loader: NegativeDataLoader,
        config: SynthesisConfig,
        sample_rate: int = 190,
        seed: int = 42,
    ):
        self.password_blocks = password_blocks
        self.negative_loader = negative_loader
        self.config = config
        self.sample_rate = sample_rate
        self.rng = np.random.RandomState(seed)
        if not self.password_blocks:
            raise ValueError("No password blocks provided!")
        print(f"BlockTemplateGenerator: {len(self.password_blocks)} block templates available")

    def _sample_context_sequence(self, stages: List[Tuple[str, Tuple[float, float]]]) -> np.ndarray:
        """Sample a protocol-like negative context from named clip categories."""
        parts = []
        for category, (dur_lo, dur_hi) in stages:
            dur_s = self.rng.uniform(dur_lo, dur_hi)
            dur_samples = max(1, int(round(dur_s * self.sample_rate)))
            parts.append(
                self.negative_loader.sample_clip(
                    dur_samples,
                    self.rng,
                    category=category,
                )
            )
        full = np.concatenate(parts, axis=0).astype(np.float32)
        return _apply_boundary_smoothing(
            full,
            [len(part) for part in parts],
            self.config.splice_smooth_samples,
        )

    def _sample_prefix_context(self) -> np.ndarray:
        # Mixed2-style lead-in: move/click/freetyping near the password block.
        stages = [
            ("idle", (0.20, 0.60)),
            ("trackpad_move", (0.25, 0.75)),
            ("trackpad_click", (0.08, 0.20)),
            ("freetyping", (0.45, 1.10)),
            ("idle", (0.10, 0.35)),
        ]
        return self._sample_context_sequence(stages)

    def _sample_suffix_context(self) -> np.ndarray:
        # Mixed2-style tail: brief settle, then shake/strong motion after submit.
        stages = [
            ("idle", (0.08, 0.25)),
            ("shake", (0.25, 0.85)),
            ("idle", (0.10, 0.35)),
        ]
        return self._sample_context_sequence(stages)

    def _sample_block(self) -> Dict:
        idx = int(self.rng.randint(len(self.password_blocks)))
        block = self.password_blocks[idx]
        imu = block["imu"].copy()
        labels = block["group_labels"].copy()
        boundaries = [tuple(x) for x in block["group_boundaries"]]
        onsets = [list(x) for x in block["onset_positions"]]
        chars = [list(x) for x in block["onset_chars"]]

        lo, hi = self.config.time_stretch_range
        stretch_rate = self.rng.uniform(lo, hi)
        if abs(stretch_rate - 1.0) > 0.01:
            old_len = len(imu)
            imu = time_stretch(imu, stretch_rate)
            new_len = len(imu)
            ratio = new_len / max(old_len, 1)
            labels = np.interp(
                np.linspace(0, len(labels) - 1, new_len),
                np.arange(len(labels)),
                labels,
            ).astype(np.float32)
            boundaries = [
                (int(round(s * ratio)), int(round(e * ratio)))
                for s, e in boundaries
            ]
            onsets = [[int(round(o * ratio)) for o in group] for group in onsets]

        gain = self.rng.uniform(*self.config.gain_augment_range)
        imu = imu * gain
        if self.config.noise_std > 0:
            imu = imu + self.rng.randn(*imu.shape).astype(np.float32) * self.config.noise_std

        return {
            "imu": imu.astype(np.float32),
            "group_labels": labels.astype(np.float32),
            "group_boundaries": boundaries,
            "onset_positions": onsets,
            "onset_chars": chars,
            "num_groups": block["num_groups"],
            "keys_per_group": block["keys_per_group"],
        }

    def generate_one(self) -> Dict:
        block = self._sample_block()

        prefix = self._sample_prefix_context()
        suffix = self._sample_suffix_context()

        full_imu = np.concatenate([prefix, block["imu"], suffix], axis=0).astype(np.float32)
        full_imu = _apply_boundary_smoothing(
            full_imu,
            [len(prefix), len(block["imu"]), len(suffix)],
            self.config.splice_smooth_samples,
        )
        shift = len(prefix)
        group_boundaries = [(s + shift, e + shift) for s, e in block["group_boundaries"]]
        onset_positions = [[o + shift for o in group] for group in block["onset_positions"]]

        group_labels = np.zeros(len(full_imu), dtype=np.float32)
        group_labels[shift:shift + len(block["group_labels"])] = block["group_labels"]

        return {
            "imu": full_imu,
            "group_labels": group_labels,
            "group_boundaries": group_boundaries,
            "onset_positions": onset_positions,
            "onset_chars": block["onset_chars"],
            "num_groups": block["num_groups"],
            "keys_per_group": block["keys_per_group"],
        }

    def generate_dataset(
        self,
        num_sessions: int,
        output_dir: str,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        output_path = Path(output_dir)
        sessions = []
        for i in range(num_sessions):
            if (i + 1) % 20 == 0:
                print(f"  Generating session {i + 1}/{num_sessions}")
            sessions.append(self.generate_one())

        indices = list(range(num_sessions))
        self.rng.shuffle(indices)
        n_train = int(num_sessions * split_ratios[0])
        n_val = int(num_sessions * split_ratios[1])
        splits = {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }

        metadata = {
            "num_sessions": num_sessions,
            "splits": {k: len(v) for k, v in splits.items()},
            "config": {
                "passwords_per_session": self.config.passwords_per_session,
                "keys_per_password": self.config.keys_per_password,
                "sample_rate": self.sample_rate,
                "template_mode": "real_block",
            },
        }

        for split_name, split_indices in splits.items():
            split_dir = output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for j, idx in enumerate(split_indices):
                s = sessions[idx]
                save_path = split_dir / f"session_{j:04d}.npz"
                np.savez_compressed(
                    save_path,
                    imu=s["imu"],
                    group_labels=s["group_labels"],
                    group_boundaries=np.array(s["group_boundaries"]),
                    onset_positions=_pad_onset_lists(s["onset_positions"]),
                    onset_chars=json.dumps(s["onset_chars"]),
                    num_groups=s["num_groups"],
                    keys_per_group=s["keys_per_group"],
                )

        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to {output_dir}")
        print(f"  Train: {len(splits['train'])} sessions")
        print(f"  Val:   {len(splits['val'])} sessions")
        print(f"  Test:  {len(splits['test'])} sessions")
        return metadata


def load_all_password_blocks(password_dir: str, target_rate_hz: int = 190) -> List[Dict]:
    return _load_password_blocks(password_dir, target_rate_hz=target_rate_hz)
