"""
Password Stage 2 Dense Temporal Model
=====================================

A lightweight multi-head temporal model for Stage 2 inside the coarse
password region.

Input:
    x: [B, T, C]

Outputs:
    {
        "key_logits": [B, T],
        "boundary_logits": [B, T],
        "inside_logits": [B, T],
    }

This file is intentionally separate from the legacy onset model so that the
existing Stage 1 / onset baselines remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DualDilatedResidualBlock(nn.Module):
    """A compact DDL-style block inspired by MS-TCN++ style refinement."""

    def __init__(self, channels: int, dilation_a: int, dilation_b: int, dropout: float = 0.1):
        super().__init__()
        self.branch_a = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation_a, dilation=dilation_a)
        self.branch_b = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation_b, dilation=dilation_b)
        self.mix = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(2 * channels, channels, kernel_size=1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        y = torch.cat([a, b], dim=1)
        y = self.mix(y)
        return x + y


class SingleStageTemporalNet(nn.Module):
    def __init__(self, channels: int = 64, depth: int = 8, dropout: float = 0.1):
        super().__init__()
        blocks = []
        for i in range(depth):
            d1 = 2 ** min(i, 5)
            d2 = 2 ** max(depth - i - 1, 0)
            blocks.append(DualDilatedResidualBlock(channels, dilation_a=d1, dilation_b=d2, dropout=dropout))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RefinementStage(nn.Module):
    def __init__(self, n_heads: int, channels: int = 64, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(n_heads, channels, kernel_size=1)
        self.tcn = SingleStageTemporalNet(channels=channels, depth=depth, dropout=dropout)
        self.out_proj = nn.Conv1d(channels, n_heads, kernel_size=1)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(probs)
        x = self.tcn(x)
        return self.out_proj(x)


@dataclass
class PasswordStage2Output:
    key_logits: torch.Tensor
    boundary_logits: torch.Tensor
    inside_logits: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "key_logits": self.key_logits,
            "boundary_logits": self.boundary_logits,
            "inside_logits": self.inside_logits,
        }


class PasswordStage2TCN(nn.Module):
    """
    Lightweight dense temporal model.

    Design:
      stem -> temporal trunk -> multi-head logits -> optional refinement stages
    """

    def __init__(
        self,
        n_channels: int,
        hidden_dim: int = 64,
        trunk_depth: int = 8,
        refine_stages: int = 2,
        refine_depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.trunk = SingleStageTemporalNet(channels=hidden_dim, depth=trunk_depth, dropout=dropout)
        self.heads = nn.Conv1d(hidden_dim, 3, kernel_size=1)
        self.refinement = nn.ModuleList(
            [RefinementStage(n_heads=3, channels=hidden_dim, depth=refine_depth, dropout=dropout) for _ in range(refine_stages)]
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, C], got {tuple(x.shape)}")
        z = x.transpose(1, 2)  # [B, C, T]
        z = self.stem(z)
        z = self.trunk(z)
        logits = self.heads(z)

        # refinement operates on probabilities as in MS-TCN-style staged refinement
        for stage in self.refinement:
            probs = torch.sigmoid(logits)
            logits = logits + stage(probs)

        key_logits = logits[:, 0, :]
        boundary_logits = logits[:, 1, :]
        inside_logits = logits[:, 2, :]
        return PasswordStage2Output(
            key_logits=key_logits,
            boundary_logits=boundary_logits,
            inside_logits=inside_logits,
        ).as_dict()


def temporal_smoothing_loss(logits: torch.Tensor, mask: torch.Tensor | None = None, clamp: float = 4.0) -> torch.Tensor:
    """
    Truncated temporal smoothing loss in the spirit of MS-TCN++.
    logits: [B, T]
    mask:   [B, T] boolean or float, 1 for valid positions
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected [B, T], got {tuple(logits.shape)}")
    diffs = logits[:, 1:] - logits[:, :-1]
    diffs = torch.clamp(diffs.square(), max=clamp)
    if mask is not None:
        mask = mask.float()
        valid = mask[:, 1:] * mask[:, :-1]
        return (diffs * valid).sum() / valid.sum().clamp_min(1.0)
    return diffs.mean()


def build_password_stage2_model(n_channels: int, **kwargs) -> PasswordStage2TCN:
    return PasswordStage2TCN(n_channels=n_channels, **kwargs)
