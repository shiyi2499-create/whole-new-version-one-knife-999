"""
PyTorch Dataset for frame-level CTC character decoding.

Each sample is one password episode:
  imu:           [T, 6] raw IMU
  hard_targets:  [T] int64 — char index at keystroke center, 0 (blank) elsewhere
  soft_weights:  [T] float32 — Gaussian loss weight around each keystroke
  ctc_target:    [num_keys] int64 — character sequence (no blank)
  password:      str — ground-truth string

Key difference from stage2_episode datasets:
  - Targets carry CHARACTER IDENTITY, not just onset presence
  - Per-key timestamp is used to place the character at the right frame
  - CTC target sequence is derived from the same events
"""
import os
import sys
import importlib.util
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Dict

# Ensure sibling package imports resolve even when called as a script entry-point.
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

try:
    from utils.signal_processing import preprocess
    from utils.vocab import BLANK_IDX, NUM_CLASSES, char_index, is_ignored_key
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
    preprocess = _sig.preprocess
    BLANK_IDX = _voc.BLANK_IDX
    NUM_CLASSES = _voc.NUM_CLASSES
    char_index = _voc.char_index
    is_ignored_key = _voc.is_ignored_key


def build_frame_targets(T: int, key_events: List[Dict],
                        sigma_frames: float = 2.0):
    """
    Construct per-frame training targets from per-key timestamps.

    Args:
        T: number of frames in episode
        key_events: list of {'ts_frame': int, 'char': str}
        sigma_frames: Gaussian width for soft weights (100Hz: 2.0 = 20ms)

    Returns:
        hard_targets: [T] int64 — char index at center frame, blank elsewhere
        soft_weights: [T] float32 — loss weight per frame
        ctc_target:   list[int] — character index sequence (no blank)
    """
    hard_targets = np.zeros(T, dtype=np.int64)  # all blank
    # Only supervise a local neighborhood around each key. Far-away frames
    # are left effectively unlabeled for CE so the model first learns local
    # character alignment instead of collapsing to blank everywhere.
    soft_weights = np.zeros(T, dtype=np.float32)
    best_scores = np.zeros(T, dtype=np.float32)

    ctc_target = []
    char_radius = max(1, int(round(max(sigma_frames, 1.5))))
    blank_band = max(char_radius + 1, int(round(2.5 * max(sigma_frames, 1.0))))

    for event in key_events:
        t_center = event['ts_frame']
        if is_ignored_key(event['char']):
            continue
        cidx = char_index(event['char'])
        ctc_target.append(cidx)

        # Supervise only a compact local band around each key.
        radius = max(blank_band, int(4 * sigma_frames))
        for dt in range(-radius, radius + 1):
            t = t_center + dt
            if 0 <= t < T:
                w = np.exp(-0.5 * (dt / max(sigma_frames, 0.5)) ** 2)
                # Near the center we supervise the character; slightly farther
                # out we supervise blank so the model learns a local "pulse"
                # shape instead of a broad character plateau.
                if abs(dt) <= char_radius and w >= best_scores[t]:
                    hard_targets[t] = cidx
                    best_scores[t] = w
                    soft_weights[t] = max(soft_weights[t], 1.0)
                elif abs(dt) <= blank_band:
                    soft_weights[t] = max(soft_weights[t], 0.35 * w)

    return hard_targets, soft_weights, ctc_target


class CTCEpisodeDataset(Dataset):
    """
    Loads episode-level .npz files with frame-level character targets.

    Each .npz contains:
        imu:           [T, 6] float32
        hard_targets:  [T] int64
        soft_weights:  [T] float32
        ctc_target:    [K] int64
        password:      str

    Also supports loading from old stage2_episode format .npz files that
    have episodes_json with onset + char info — auto-converts on the fly.
    """

    def __init__(self, data_dir: str, split: str = 'train',
                 sample_rate: int = 100, add_mag: bool = True,
                 norm: bool = True, norm_stats: Optional[dict] = None,
                 sigma_ms: float = 20.0):
        data_roots = [p.strip() for p in str(data_dir).split(',') if p.strip()]
        self.split_dirs = [Path(p) / split for p in data_roots]
        self.sr = sample_rate
        self.add_mag = add_mag
        self.norm = norm
        self.norm_stats = norm_stats
        self.sigma_frames = max(1.0, sigma_ms / 1000.0 * sample_rate)

        self.files = []
        for split_dir in self.split_dirs:
            if split_dir.exists():
                self.files.extend(sorted(split_dir.glob("*.npz")))

        if not self.files:
            joined = ", ".join(str(d) for d in self.split_dirs)
            print(f"Warning: no files in {joined}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)

        # New CTC format: has hard_targets directly
        if 'hard_targets' in d:
            imu = d['imu'].astype(np.float32)
            hard_targets = d['hard_targets'].astype(np.int64)
            soft_weights = d['soft_weights'].astype(np.float32)
            ctc_target = d['ctc_target'].astype(np.int64)
        else:
            # Backward compat: convert from stage2_episode format
            imu, hard_targets, soft_weights, ctc_target = \
                self._convert_episode_format(d)

        proc, _ = preprocess(imu, self.sr, self.add_mag, self.norm,
                             self.norm_stats)

        x = torch.from_numpy(proc.T).float()                    # [C, T]
        ht = torch.from_numpy(hard_targets)                      # [T]
        sw = torch.from_numpy(soft_weights)                      # [T]
        ct = torch.from_numpy(ctc_target.copy())                 # [K]
        return x, ht, sw, ct

    def _convert_episode_format(self, d):
        """Convert old stage2_episode .npz with episodes_json to CTC format."""
        imu = d['imu'].astype(np.float32)
        T = len(imu)

        episodes_json = str(d.get('episodes_json', '[]'))
        episodes = json.loads(episodes_json)

        # Combine all episodes in this sample into one target sequence
        all_events = []
        for ep in episodes:
            onsets = ep.get('onsets', [])
            chars = ep.get('chars', [])
            for onset, char in zip(onsets, chars):
                all_events.append({'ts_frame': int(onset), 'char': str(char)})

        hard_targets, soft_weights, ctc_target = build_frame_targets(
            T, all_events, self.sigma_frames
        )
        return imu, hard_targets, soft_weights, np.array(ctc_target, dtype=np.int64)

    @staticmethod
    def collate(batch):
        """
        Variable-length collation with padding.

        Returns:
            x:                   [B, C, T_max] padded IMU
            hard_targets:        [B, T_max] padded (0 = blank)
            soft_weights:        [B, T_max] padded (0.0)
            mask:                [B, T_max] valid frame mask
            ctc_targets:         [sum(K_i)] concatenated target sequences
            ctc_target_lengths:  [B] length of each target sequence
            input_lengths:       [B] actual T for each sample
        """
        xs, hts, sws, cts = zip(*batch)

        B = len(xs)
        C = xs[0].shape[0]
        max_T = max(x.shape[1] for x in xs)

        x_pad = torch.zeros(B, C, max_T)
        ht_pad = torch.zeros(B, max_T, dtype=torch.long)
        sw_pad = torch.zeros(B, max_T)
        mask = torch.zeros(B, max_T)
        input_lengths = torch.zeros(B, dtype=torch.long)

        ctc_targets_list = []
        ctc_target_lengths = torch.zeros(B, dtype=torch.long)

        for i in range(B):
            T = xs[i].shape[1]
            x_pad[i, :, :T] = xs[i]
            ht_pad[i, :T] = hts[i]
            sw_pad[i, :T] = sws[i]
            mask[i, :T] = 1.0
            input_lengths[i] = T
            ctc_targets_list.append(cts[i])
            ctc_target_lengths[i] = len(cts[i])

        ctc_targets_cat = torch.cat(ctc_targets_list)

        return (x_pad, ht_pad, sw_pad, mask,
                ctc_targets_cat, ctc_target_lengths, input_lengths)
