"""
PyTorch Dataset classes for Stage 2A (group segmentation) and Stage 2B (onset detection).
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple

from utils.signal_processing import preprocess_imu


class Stage2ADataset(Dataset):
    """
    Dataset for Stage 2A: Group Segmentation.

    Each sample is a full synthetic mixed session:
        Input:  [T, C] preprocessed IMU
        Target: [T] binary labels (1=typing, 0=gap/context)
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 sample_rate: int = 100,
                 normalize: bool = True,
                 add_magnitude: bool = True,
                 max_length: Optional[int] = None,
                 norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Args:
            data_dir: path to synthetic_mixed/ directory
            split: 'train', 'val', or 'test'
            sample_rate: Hz
            max_length: if set, truncate/pad to this length
            norm_stats: (mean, std) for normalization; if None, compute per-sample
        """
        self.split_dir = Path(data_dir) / split
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.add_magnitude = add_magnitude
        self.max_length = max_length
        self.norm_stats = norm_stats

        # Discover all .npz files
        self.files = sorted(self.split_dir.glob("session_*.npz"))
        if len(self.files) == 0:
            print(f"Warning: No files found in {self.split_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True)

        imu = data['imu']  # [T, 6]
        labels = data['group_labels']  # [T]

        # Preprocess
        processed, _ = preprocess_imu(
            imu,
            sample_rate=self.sample_rate,
            normalize=self.normalize,
            bandpass=False,  # already clean synthetic data
            add_magnitude=self.add_magnitude,
            norm_stats=self.norm_stats,
        )

        # Optional length control
        T = len(processed)
        if self.max_length is not None and T > self.max_length:
            # Random crop for training
            start = np.random.randint(0, T - self.max_length)
            processed = processed[start:start + self.max_length]
            labels = labels[start:start + self.max_length]

        # To tensors: [C, T] for Conv1D
        x = torch.from_numpy(processed.T).float()  # [C, T]
        y = torch.from_numpy(labels).float()  # [T]

        return x, y

    @staticmethod
    def collate_fn(batch):
        """
        Pad sequences to same length within a batch.
        """
        xs, ys = zip(*batch)

        # Find max length in this batch
        max_T = max(x.shape[1] for x in xs)

        padded_xs = []
        padded_ys = []
        masks = []

        for x, y in zip(xs, ys):
            T = x.shape[1]
            pad_len = max_T - T
            if pad_len > 0:
                x = torch.nn.functional.pad(x, (0, pad_len), value=0.0)
                y = torch.nn.functional.pad(y, (0, pad_len), value=0.0)
                mask = torch.cat([torch.ones(T), torch.zeros(pad_len)])
            else:
                mask = torch.ones(T)

            padded_xs.append(x)
            padded_ys.append(y)
            masks.append(mask)

        return (torch.stack(padded_xs),    # [B, C, T]
                torch.stack(padded_ys),    # [B, T]
                torch.stack(masks))         # [B, T]


class Stage2BDataset(Dataset):
    """
    Dataset for Stage 2B: Onset Detection within a single password group.

    Each sample is one password group segment:
        Input:  [T_group, C] preprocessed IMU
        Target: [T_group] Gaussian peak target (soft onset labels)
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 sample_rate: int = 100,
                 gaussian_sigma_ms: float = 15.0,
                 normalize: bool = True,
                 add_magnitude: bool = True,
                 expected_onsets: int = 8,
                 norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Extracts individual password group segments from synthetic mixed sessions.
        """
        self.split_dir = Path(data_dir) / split
        self.sample_rate = sample_rate
        self.sigma_samples = gaussian_sigma_ms / 1000.0 * sample_rate
        self.normalize = normalize
        self.add_magnitude = add_magnitude
        self.expected_onsets = expected_onsets
        self.norm_stats = norm_stats

        # Pre-extract all group segments
        self.segments = []
        files = sorted(self.split_dir.glob("session_*.npz"))

        for f in files:
            try:
                self._extract_segments(f)
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")

        print(f"Stage2BDataset [{split}]: {len(self.segments)} group segments from {len(files)} sessions")

    def _extract_segments(self, npz_path: Path):
        """Extract individual group segments from a session file."""
        data = np.load(npz_path, allow_pickle=True)

        imu = data['imu']  # [T, 6]
        boundaries = data['group_boundaries']  # [N, 2]
        onset_positions = data['onset_positions']  # [N, max_onsets]

        onset_chars_str = str(data['onset_chars'])
        try:
            onset_chars = json.loads(onset_chars_str)
        except:
            onset_chars = [['?'] * 8] * len(boundaries)

        for g in range(len(boundaries)):
            start, end = int(boundaries[g][0]), int(boundaries[g][1])
            if end <= start or end - start < 10:
                continue

            seg_imu = imu[start:end]

            # Get onsets for this group, convert to local coords
            if g < len(onset_positions):
                global_onsets = onset_positions[g]
                local_onsets = [int(o - start) for o in global_onsets
                                if o >= 0 and 0 <= o - start < len(seg_imu)]
            else:
                local_onsets = []

            chars = onset_chars[g] if g < len(onset_chars) else []

            self.segments.append({
                'imu': seg_imu.astype(np.float32),
                'onsets': local_onsets,
                'chars': chars,
            })

    def _make_gaussian_target(self, T: int, onsets: list) -> np.ndarray:
        """Create Gaussian peak target from onset positions."""
        target = np.zeros(T, dtype=np.float32)
        for o in onsets:
            if 0 <= o < T:
                t = np.arange(T)
                target += np.exp(-((t - o) ** 2) / (2 * self.sigma_samples ** 2))
        # Clip to [0, 1]
        target = np.clip(target, 0.0, 1.0)
        return target

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        imu = seg['imu']
        onsets = seg['onsets']

        # Preprocess
        processed, _ = preprocess_imu(
            imu,
            sample_rate=self.sample_rate,
            normalize=self.normalize,
            bandpass=False,
            add_magnitude=self.add_magnitude,
            norm_stats=self.norm_stats,
        )

        T = len(processed)
        target = self._make_gaussian_target(T, onsets)

        # To tensors
        x = torch.from_numpy(processed.T).float()  # [C, T]
        y = torch.from_numpy(target).float()  # [T]

        # Also return onset positions for metric computation
        onset_array = np.array(onsets, dtype=np.int64)

        return x, y, onset_array

    @staticmethod
    def collate_fn(batch):
        """Pad to same length, handle variable onset counts."""
        xs, ys, onsets_list = zip(*batch)

        max_T = max(x.shape[1] for x in xs)

        padded_xs = []
        padded_ys = []
        masks = []

        for x, y in zip(xs, ys):
            T = x.shape[1]
            pad_len = max_T - T
            if pad_len > 0:
                x = torch.nn.functional.pad(x, (0, pad_len), value=0.0)
                y = torch.nn.functional.pad(y, (0, pad_len), value=0.0)
                mask = torch.cat([torch.ones(T), torch.zeros(pad_len)])
            else:
                mask = torch.ones(T)
            padded_xs.append(x)
            padded_ys.append(y)
            masks.append(mask)

        return (torch.stack(padded_xs),
                torch.stack(padded_ys),
                torch.stack(masks),
                list(onsets_list))  # keep as list of variable-length arrays
