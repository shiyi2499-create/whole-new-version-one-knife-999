"""
Onset Detection Models
======================

This module keeps the original onset detector intact and adds a dedicated
password-boundary model for extracting password episodes from mixed streams.

Supported tasks:
  - onset              : binary keystroke onset detection
  - activity           : binary keyboard-active detection (legacy)
  - password_boundary  : 4-way password boundary classification
                         [non_password, password_start,
                          password_active, password_end]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder1D(nn.Module):
    """Small reusable 1-D CNN encoder."""

    def __init__(
        self,
        n_channels: int,
        channels: list[int],
        kernels: list[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        assert len(channels) == len(kernels)
        layers = []
        in_ch = n_channels
        for i, (out_ch, k) in enumerate(zip(channels, kernels)):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ])
            if dropout > 0 and i < len(channels) - 1:
                layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OnsetCNN(nn.Module):
    """3-layer binary onset detector."""

    def __init__(self, n_channels: int = 6, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        self.encoder = ConvEncoder1D(
            n_channels=n_channels,
            channels=[32, 64, 64],
            kernels=[5, 5, 3],
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            if logits.shape[-1] == 1:
                return torch.sigmoid(logits).squeeze(-1)
            return torch.softmax(logits, dim=-1)


class OnsetCNNLarge(nn.Module):
    """Slightly larger onset model for ablations."""

    def __init__(self, n_channels: int = 6, dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        self.encoder = ConvEncoder1D(
            n_channels=n_channels,
            channels=[32, 64, 128, 128, 64],
            kernels=[7, 5, 5, 3, 3],
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_channels, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            if logits.shape[-1] == 1:
                return torch.sigmoid(logits).squeeze(-1)
            return torch.softmax(logits, dim=-1)


class ActivitySegmentCNN(nn.Module):
    """Legacy binary keyboard-active detector."""

    def __init__(self, n_channels: int = 6, dropout: float = 0.3, out_dim: int = 1):
        super().__init__()
        self.encoder = ConvEncoder1D(
            n_channels=n_channels,
            channels=[32, 64, 64, 32],
            kernels=[9, 7, 5, 3],
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            if logits.shape[-1] == 1:
                return torch.sigmoid(logits).squeeze(-1)
            return torch.softmax(logits, dim=-1)


class PasswordBoundaryCNN(nn.Module):
    """
    Dedicated 4-way password-boundary classifier.

    The window label is one of:
      0 non_password
      1 password_start
      2 password_active
      3 password_end

    Wider kernels than the onset detector are used so the model can see
    enough pre/post context around the password boundary.
    """

    def __init__(
        self,
        n_channels: int = 6,
        dropout: float = 0.3,
        out_dim: int = 4,
    ):
        super().__init__()
        self.encoder = ConvEncoder1D(
            n_channels=n_channels,
            channels=[48, 96, 96, 64],
            kernels=[11, 9, 7, 5],
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.out_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


class EnergyOnsetBaseline:
    """Simple energy-threshold onset detector for ablation comparison."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def compute_energy(self, windows: "np.ndarray") -> "np.ndarray":
        import numpy as np
        return np.sqrt(np.mean(windows ** 2, axis=(1, 2)))

    def predict(self, windows: "np.ndarray") -> "np.ndarray":
        import numpy as np
        energy = self.compute_energy(windows)
        return (energy > self.threshold).astype(np.int32)

    def calibrate(self, windows: "np.ndarray", labels: "np.ndarray"):
        pos_energy = self.compute_energy(windows[labels == 1])
        neg_energy = self.compute_energy(windows[labels == 0])
        if len(pos_energy) == 0 or len(neg_energy) == 0:
            self.threshold = 0.0
            return
        self.threshold = float((pos_energy.mean() + neg_energy.mean()) / 2.0)


MODEL_TASK_DEFAULTS = {
    "onset": "cnn",
    "activity": "activity_cnn",
    "password_boundary": "password_boundary_cnn",
}


def build_onset_model(
    name: str = "cnn",
    n_channels: int = 6,
    n_classes: int = 1,
    task: str | None = None,
    **kwargs,
) -> nn.Module:
    """Build a detector / segmenter model by name or task alias."""
    if name == "auto" and task:
        name = MODEL_TASK_DEFAULTS.get(task, "cnn")
    if task == "password_boundary" and name == "cnn":
        name = "password_boundary_cnn"
    if task == "activity" and name == "cnn":
        name = "activity_cnn"

    if name == "cnn":
        return OnsetCNN(n_channels=n_channels, out_dim=n_classes, **kwargs)
    if name == "cnn_large":
        return OnsetCNNLarge(n_channels=n_channels, out_dim=n_classes, **kwargs)
    if name == "activity_cnn":
        return ActivitySegmentCNN(n_channels=n_channels, out_dim=n_classes, **kwargs)
    if name == "password_boundary_cnn":
        return PasswordBoundaryCNN(n_channels=n_channels, out_dim=n_classes, **kwargs)
    raise ValueError(f"Unknown onset model: {name}")
