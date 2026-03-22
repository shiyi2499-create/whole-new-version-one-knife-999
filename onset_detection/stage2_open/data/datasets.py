"""
PyTorch Dataset for the open 3-class frame-level task.
Each sample: (imu [C, T], labels [T], mask [T]).
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple

from utils.signal_processing import preprocess


class OpenFrameDataset(Dataset):
    """
    Loads synthetic open sessions.
    Labels: 0=gap, 1=keystroke, 2=separator.
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

        proc, _ = preprocess(imu, self.sr, self.add_mag, self.norm, self.norm_stats)

        x = torch.from_numpy(proc.T).float()          # [C, T]
        y = torch.from_numpy(labels.astype(np.int64))  # [T]
        return x, y

    @staticmethod
    def collate(batch):
        xs, ys = zip(*batch)
        max_T = max(x.shape[1] for x in xs)
        B = len(xs)
        C = xs[0].shape[0]

        xp = torch.zeros(B, C, max_T)
        yp = torch.full((B, max_T), 0, dtype=torch.long)  # pad with 'gap'
        mask = torch.zeros(B, max_T)

        for i, (x, y) in enumerate(zip(xs, ys)):
            T = x.shape[1]
            xp[i, :, :T] = x
            yp[i, :T] = y
            mask[i, :T] = 1.0

        return xp, yp, mask
