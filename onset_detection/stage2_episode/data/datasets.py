"""
PyTorch Dataset for the 2-class frame-level episode task.
Each sample: (imu [C, T], labels [T], onset_targets [T], mask [T]).
Labels: 0=silence, 1=typing.
onset_targets: float32 Gaussian impulse per-frame target for onset_head.
               May be all-zeros for old data that pre-dates the dual-head design.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional

from utils.signal_processing import preprocess


class EpisodeFrameDataset(Dataset):
    """
    Loads synthetic or real episode sessions.
    Labels: 0=silence, 1=typing (2-class).

    Compatible with both old 3-class data (maps separator→silence)
    and new 2-class data.

    Returns onset_targets as a zero tensor for legacy files that don't have it.
    The trainer checks whether any onset_targets are non-zero before computing
    the onset loss, so old data trains only the typing head — no breakage.
    """

    def __init__(self, data_dir: str, split: str = 'train',
                 sample_rate: int = 100, add_mag: bool = True,
                 norm: bool = True,
                 norm_stats: Optional[dict] = None):
        data_roots = [p.strip() for p in str(data_dir).split(',') if p.strip()]
        self.split_dirs = [Path(p) / split for p in data_roots]
        self.sr = sample_rate
        self.add_mag = add_mag
        self.norm = norm
        self.norm_stats = norm_stats
        self.files = []
        for split_dir in self.split_dirs:
            self.files.extend(sorted(split_dir.glob("session_*.npz")))
        if not self.files:
            joined = ", ".join(str(d) for d in self.split_dirs)
            print(f"Warning: no files in {joined}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)
        imu = d['imu']                  # [T, 6]
        labels = d['frame_labels']      # [T]

        # Backward compat: if data has 3-class labels, convert:
        #   0 (gap) → 0 (silence)
        #   1 (keystroke) → 1 (typing)
        #   2 (separator) → 0 (silence)  ← key change
        if labels.max() > 1:
            labels = (labels == 1).astype(np.int64)

        # Onset Gaussian targets (new dual-head data) or zeros (legacy data)
        if 'onset_targets' in d:
            onset_targets = d['onset_targets'].astype(np.float32)
        else:
            onset_targets = np.zeros(len(labels), dtype=np.float32)

        proc, _ = preprocess(imu, self.sr, self.add_mag, self.norm, self.norm_stats)

        x = torch.from_numpy(proc.T).float()                          # [C, T]
        y = torch.from_numpy(labels.astype(np.int64))                  # [T]
        ot = torch.from_numpy(onset_targets)                           # [T]
        return x, y, ot

    @staticmethod
    def collate(batch):
        xs, ys, ots = zip(*batch)
        max_T = max(x.shape[1] for x in xs)
        B = len(xs)
        C = xs[0].shape[0]

        xp = torch.zeros(B, C, max_T)
        yp = torch.full((B, max_T), 0, dtype=torch.long)
        otp = torch.zeros(B, max_T)
        mask = torch.zeros(B, max_T)

        for i, (x, y, ot) in enumerate(zip(xs, ys, ots)):
            T = x.shape[1]
            xp[i, :, :T] = x
            yp[i, :T] = y
            otp[i, :T] = ot
            mask[i, :T] = 1.0

        return xp, yp, otp, mask
