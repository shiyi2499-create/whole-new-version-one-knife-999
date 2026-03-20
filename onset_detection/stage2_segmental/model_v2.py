"""
stage2_segmental model_v2: Learned Overlapping Windows

Key change vs v1 (model.py):
  v1 partitions the episode into non-overlapping segments (shared boundaries).
  v2 gives each key an INDEPENDENT window that can freely overlap with neighbors.

Why this matters:
  The classifier was trained on ~300ms windows (100ms pre + 200ms post) centered
  on key timestamps.  Adjacent keys are often <100ms apart, so their training
  windows overlap 50-80%.  The v1 partition model forced non-overlap, giving
  fast-typed keys only 5-10 frames each (11x upsampling to 57 -> destroyed signal).

  v2 initializes each key's window to match the classifier's training distribution
  (offset=0, width=prior_pre+prior_post).  The learned offset/width can then
  refine per-key alignment without breaking the classifier's expectations.

Architecture:
  episode IMU [T, 6]
       |
  EpisodeEncoder -> frame features [T, H]
       |
  For each key i at frame t_i:
    - gather local features around t_i
    - predict offset_i  (small shift, init~0)
    - predict width_i   (scale of default window, init~1.0)
    - window_start = t_i + offset_i - width_i * trigger_ratio
    - window_end   = t_i + offset_i + width_i * (1 - trigger_ratio)
       |
  grid_sample each window -> [target_len, 6]
       |
  frozen classifier -> logits [K, num_classes]
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "phase3_password_inception")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phase3_password_inception.run_password_closure_inception import (
    InceptionTimeClassifier,
    augment_batch,
)

# Reuse building blocks from v1
from onset_detection.stage2_segmental.model import (
    EpisodeEncoder,
    ResidualDilatedBlock,
    SegmentalClassifier,
    build_classifier,
    load_external_inception,
    save_classifier_checkpoint,
    train_classifier,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class OverlapConfig:
    input_channels: int = 6
    encoder_hidden: int = 96
    encoder_blocks: int = 6
    encoder_kernel: int = 7
    encoder_dropout: float = 0.10
    target_len: int = 57

    # default window = classifier's training distribution
    trigger_ratio: float = 1.0 / 3.0       # key sits at 1/3 of window
    prior_pre_ms: float = 100.0
    prior_post_ms: float = 200.0

    # offset bounds: max learnable shift in ms
    max_offset_ms: float = 60.0

    # width bounds: multiplicative range around prior width
    max_width_scale: float = 2.00

    # local context for per-key features (in frames)
    context_radius: int = 10

    # loss weights
    loss_char: float = 1.0
    loss_offset: float = 0.10       # L2 regularization on offset
    loss_width: float = 0.08        # L2 on log(width_scale)
    loss_consistency: float = 0.05  # smooth neighboring offsets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class OverlapWindowModel(nn.Module):
    """Per-key learned overlapping windows + frozen classifier."""

    def __init__(self, config: OverlapConfig, classifier: SegmentalClassifier):
        super().__init__()
        self.cfg = config
        self.classifier = classifier

        self.encoder = EpisodeEncoder(
            input_channels=config.input_channels,
            hidden=config.encoder_hidden,
            n_blocks=config.encoder_blocks,
            kernel=config.encoder_kernel,
            dropout=config.encoder_dropout,
        )

        h = config.encoder_hidden
        # local feature: key_feat + left_ctx + right_ctx + global = 4*h
        feat_dim = h * 4

        # Heads predict per-key offset and width_scale
        self.offset_head = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(h, 1),
        )
        self.width_head = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(h, 1),
        )

        # Initialize heads to output near-zero (offset~0, width_scale~1.0)
        self._init_heads()

    def _init_heads(self):
        """Initialize so offset~0 and width_scale~1.0 at start."""
        for head in [self.offset_head, self.width_head]:
            final = head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def freeze_classifier(self, freeze: bool = True):
        for p in self.classifier.parameters():
            p.requires_grad = not freeze

    def _gather_feature(self, feats: torch.Tensor, index: float) -> torch.Tensor:
        """Gather feature at a frame index with boundary clamping."""
        T = feats.shape[0]
        idx = int(round(float(index)))
        idx = min(max(idx, 0), T - 1)
        return feats[idx]

    def _local_pool(self, feats: torch.Tensor, center: float, radius: int) -> torch.Tensor:
        """Average-pool features in [center-radius, center+radius]."""
        T = feats.shape[0]
        c = int(round(float(center)))
        lo = max(0, c - radius)
        hi = min(T, c + radius + 1)
        if hi <= lo:
            return feats[min(max(c, 0), T - 1)]
        return feats[lo:hi].mean(dim=0)

    def predict_windows(
        self,
        feats: torch.Tensor,
        key_frames: torch.Tensor,
        sample_rate_hz: float,
    ) -> dict:
        """
        Predict per-key (start, end) windows.  Windows CAN overlap.

        Returns dict with:
          starts:  [K] tensor of window start frames
          ends:    [K] tensor of window end frames
          offsets: [K] learned offsets in frames
          widths:  [K] learned widths in frames
          width_scales: [K] multiplicative scale factors
        """
        device = feats.device
        t = key_frames.float().to(device)
        T = feats.shape[0]
        K = len(t)

        prior_pre = self.cfg.prior_pre_ms / 1000.0 * sample_rate_hz
        prior_post = self.cfg.prior_post_ms / 1000.0 * sample_rate_hz
        prior_width = prior_pre + prior_post
        max_offset = self.cfg.max_offset_ms / 1000.0 * sample_rate_hz
        trigger = self.cfg.trigger_ratio

        global_feat = feats.mean(dim=0)  # [H]
        radius = self.cfg.context_radius

        offsets_list = []
        width_scales_list = []

        for i in range(K):
            ti = float(t[i])
            key_feat = self._gather_feature(feats, ti)
            left_ctx = self._local_pool(feats, ti - radius, radius // 2)
            right_ctx = self._local_pool(feats, ti + radius, radius // 2)

            feat_vec = torch.cat([key_feat, left_ctx, right_ctx, global_feat], dim=0)

            # offset: tanh * max_offset -> bounded [-max_offset, +max_offset]
            raw_offset = self.offset_head(feat_vec).squeeze(-1)
            offset = torch.tanh(raw_offset) * max_offset
            offsets_list.append(offset)

            # width_scale: exp(tanh(raw) * log(max_scale))
            # At raw=0: scale=1.0.  Bounded in [1/max_scale, max_scale].
            raw_width = self.width_head(feat_vec).squeeze(-1)
            log_range = math.log(max(self.cfg.max_width_scale, 1.01))
            width_scale = torch.exp(torch.tanh(raw_width) * log_range)
            width_scales_list.append(width_scale)

        offsets = torch.stack(offsets_list)         # [K]
        width_scales = torch.stack(width_scales_list)  # [K]
        widths = prior_width * width_scales          # [K]

        # Per-key window boundaries (can overlap!)
        centers = t + offsets
        starts = centers - widths * trigger
        ends = centers + widths * (1.0 - trigger)

        # Clamp to valid signal range
        starts = torch.clamp(starts, min=0.0)
        ends = torch.clamp(ends, max=float(T - 1))
        # Ensure minimum window (at least 20% of prior)
        min_win = max(4.0, prior_width * 0.2)
        ends = torch.max(ends, starts + min_win)

        return {
            "starts": starts,
            "ends": ends,
            "offsets": offsets,
            "widths": widths,
            "width_scales": width_scales,
            "prior_width": torch.tensor(prior_width, dtype=torch.float32, device=device),
        }

    def _sample_window(
        self,
        signal_tc: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable resample of signal[start:end] -> [target_len, C]."""
        T = signal_tc.shape[0]
        signal = signal_tc.transpose(0, 1).unsqueeze(0).unsqueeze(2)  # [1, C, T, 1]
        steps = torch.linspace(0.0, 1.0, self.cfg.target_len, device=signal_tc.device)
        positions = start + steps * torch.clamp(end - start, min=1.0)
        # Keep sampling locations inside the valid range so the learned window
        # behavior stays the same across backends; this also lets us avoid MPS'
        # lack of border-padding support in grid_sample.
        positions = torch.clamp(positions, 0.0, float(max(T - 1, 0)))
        norm_x = 2.0 * positions / max(T - 1, 1) - 1.0
        grid = torch.stack([norm_x, torch.zeros_like(norm_x)], dim=-1)
        grid = grid.view(1, 1, self.cfg.target_len, 2)
        padding_mode = "border"
        if signal_tc.device.type == "mps":
            padding_mode = "zeros"
        sampled = F.grid_sample(
            signal, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True,
        )
        return sampled[0, :, 0, :].transpose(0, 1)  # [target_len, C]

    def extract_windows(
        self,
        signal_tc: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """Extract K overlapping windows -> [K, target_len, C]."""
        windows = []
        for i in range(len(starts)):
            windows.append(self._sample_window(signal_tc, starts[i], ends[i]))
        return torch.stack(windows, dim=0)

    def forward_episode(
        self,
        imu: torch.Tensor,
        key_frames: torch.Tensor,
        sample_rate_hz: float,
    ) -> dict:
        feats = self.encoder(imu.unsqueeze(0))[0]  # [T, H]
        win_out = self.predict_windows(feats, key_frames, sample_rate_hz)
        windows = self.extract_windows(imu, win_out["starts"], win_out["ends"])
        logits = self.classifier(windows)
        return {
            **win_out,
            "windows": windows,
            "logits": logits,
        }

    def compute_loss(
        self, out: dict, labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = labels.long().to(out["logits"].device)
        K = len(labels)

        # Main signal: character classification
        char_loss = F.cross_entropy(out["logits"], labels)

        # Offset regularization: keep offsets small (prior = 0)
        offset_loss = (out["offsets"] ** 2).mean()

        # Width regularization: keep width_scale near 1.0
        # log(width_scale)^2 -> 0 when scale=1.0
        width_loss = (torch.log(out["width_scales"]) ** 2).mean()

        # Consistency: neighboring keys should have similar offsets
        if K > 1:
            consistency_loss = ((out["offsets"][1:] - out["offsets"][:-1]) ** 2).mean()
        else:
            consistency_loss = torch.zeros((), device=out["logits"].device)

        total = (
            self.cfg.loss_char * char_loss
            + self.cfg.loss_offset * offset_loss
            + self.cfg.loss_width * width_loss
            + self.cfg.loss_consistency * consistency_loss
        )

        metrics = {
            "loss": float(total.detach().cpu()),
            "char_loss": float(char_loss.detach().cpu()),
            "offset_loss": float(offset_loss.detach().cpu()),
            "width_loss": float(width_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "mean_offset_frames": float(out["offsets"].detach().mean().cpu()),
            "mean_width_scale": float(out["width_scales"].detach().mean().cpu()),
            "mean_width_frames": float(out["widths"].detach().mean().cpu()),
        }
        return total, metrics

    def checkpoint_payload(self) -> dict:
        return {
            "model_state": self.state_dict(),
            "config": self.cfg.__dict__,
            "classifier_meta": self.classifier.meta(),
        }


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------
def load_overlap_checkpoint(path: str, device: torch.device) -> OverlapWindowModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = OverlapConfig(**ckpt["config"])
    meta = ckpt["classifier_meta"]
    classifier = build_classifier(
        target_len=int(meta["target_len"]),
        classes=[str(x) for x in meta["classes"]],
        means=np.asarray(meta["means"], dtype=np.float32),
        stds=np.asarray(meta["stds"], dtype=np.float32),
    ).to(device)
    model = OverlapWindowModel(cfg, classifier).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model
