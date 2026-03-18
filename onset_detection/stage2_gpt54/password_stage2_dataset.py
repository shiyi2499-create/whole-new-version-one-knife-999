"""
Password Stage 2 Sequence Dataset
=================================

Variable-length sequence dataset for dense patch labeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PasswordStage2Sequence:
    features: np.ndarray          # [T, C]
    patch_times_ns: np.ndarray    # [T]
    key_target: np.ndarray        # [T]
    boundary_target: np.ndarray   # [T]
    inside_target: np.ndarray     # [T]
    session_id: str
    source: str


class PasswordStage2Dataset(Dataset):
    def __init__(self, sequences: list[PasswordStage2Sequence], normalize: bool = True, means: np.ndarray | None = None, stds: np.ndarray | None = None):
        self.sequences = sequences
        self.normalize = normalize
        if not sequences:
            self.means = np.zeros((1,), dtype=np.float32)
            self.stds = np.ones((1,), dtype=np.float32)
            return

        stacked = np.concatenate([s.features for s in sequences], axis=0)
        if means is None:
            means = stacked.mean(axis=0)
        if stds is None:
            stds = stacked.std(axis=0)
        self.means = means.astype(np.float32)
        self.stds = np.maximum(stds.astype(np.float32), 1e-6)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq = self.sequences[idx]
        x = seq.features.astype(np.float32)
        if self.normalize:
            x = (x - self.means) / self.stds
        return {
            "features": torch.from_numpy(x),
            "patch_times_ns": torch.from_numpy(seq.patch_times_ns.astype(np.int64)),
            "key_target": torch.from_numpy(seq.key_target.astype(np.float32)),
            "boundary_target": torch.from_numpy(seq.boundary_target.astype(np.float32)),
            "inside_target": torch.from_numpy(seq.inside_target.astype(np.float32)),
            "session_id": seq.session_id,
            "source": seq.source,
        }


def pad_collate_password_stage2(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")

    lengths = [item["features"].shape[0] for item in batch]
    max_len = max(lengths)
    feat_dim = batch[0]["features"].shape[1]

    x = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    patch_times = torch.zeros(len(batch), max_len, dtype=torch.int64)
    key_t = torch.zeros(len(batch), max_len, dtype=torch.float32)
    boundary_t = torch.zeros(len(batch), max_len, dtype=torch.float32)
    inside_t = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    session_ids: list[str] = []
    sources: list[str] = []
    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        x[i, :n] = item["features"]
        patch_times[i, :n] = item["patch_times_ns"]
        key_t[i, :n] = item["key_target"]
        boundary_t[i, :n] = item["boundary_target"]
        inside_t[i, :n] = item["inside_target"]
        mask[i, :n] = True
        session_ids.append(str(item["session_id"]))
        sources.append(str(item["source"]))

    return {
        "features": x,
        "patch_times_ns": patch_times,
        "key_target": key_t,
        "boundary_target": boundary_t,
        "inside_target": inside_t,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "session_ids": session_ids,
        "sources": sources,
    }
