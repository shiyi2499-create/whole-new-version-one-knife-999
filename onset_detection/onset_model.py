"""
Onset Detection Model
=====================
Lightweight 1D-CNN binary classifier for keystroke onset detection.

Input:  (batch, timesteps, 6)   – 6-channel IMU window
Output: (batch, 1)              – sigmoid probability of keystroke onset

Design rationale:
- Detection windows are short (~29 samples @ 190 Hz = 150ms)
- The task is *binary* (onset vs not-onset), much simpler than 36-class key ID
- Inference must be fast enough for real-time sliding-window scanning
- A simple 3-layer 1D-CNN with ~25k params is more than sufficient

Also provides an energy-based baseline for ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OnsetCNN(nn.Module):
    """
    3-layer 1D-CNN onset detector.

    Architecture:
        Conv1d(6→32, k=5) → BN → ReLU → Dropout
        Conv1d(32→64, k=5) → BN → ReLU → Dropout
        Conv1d(64→64, k=3) → BN → ReLU
        GlobalAvgPool → FC(64→1) → Sigmoid
    """

    def __init__(self, n_channels: int = 6, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, timesteps, channels)
        Returns:
            logits: (batch, 1) – raw logits (use sigmoid for probability)
        """
        # Conv1d expects (batch, channels, timesteps)
        x = x.permute(0, 2, 1)
        x = self.features(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities, shape (batch,)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)


class OnsetCNNLarge(nn.Module):
    """
    Slightly larger variant (5 layers, ~80k params) for ablation.
    Use if OnsetCNN under-fits on more complex mixed-stream data.
    """

    def __init__(self, n_channels: int = 6, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.features(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(-1)


# ── Energy-based baseline (no learned parameters) ────────────

class EnergyOnsetBaseline:
    """
    Simple energy-threshold onset detector for ablation comparison.

    Computes per-window RMS energy across all channels, then thresholds.
    No training required – just needs a threshold calibrated on validation data.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def compute_energy(self, windows: "np.ndarray") -> "np.ndarray":
        """
        Args:
            windows: (N, timesteps, channels)
        Returns:
            energy: (N,)
        """
        import numpy as np
        return np.sqrt(np.mean(windows ** 2, axis=(1, 2)))

    def predict(self, windows: "np.ndarray") -> "np.ndarray":
        """Binary prediction: 1 if energy > threshold."""
        import numpy as np
        energy = self.compute_energy(windows)
        return (energy > self.threshold).astype(np.int32)

    def calibrate(self, windows: "np.ndarray", labels: "np.ndarray"):
        """
        Set threshold as the midpoint between mean energy of positive
        and negative windows.
        """
        import numpy as np
        pos_energy = self.compute_energy(windows[labels == 1])
        neg_energy = self.compute_energy(windows[labels == 0])
        if len(pos_energy) == 0 or len(neg_energy) == 0:
            self.threshold = 0.0
            return
        self.threshold = (pos_energy.mean() + neg_energy.mean()) / 2.0


# ── Factory ──────────────────────────────────────────────────

def build_onset_model(name: str = "cnn", n_channels: int = 6, **kwargs) -> nn.Module:
    """Build an onset detector by name."""
    if name == "cnn":
        return OnsetCNN(n_channels=n_channels, **kwargs)
    elif name == "cnn_large":
        return OnsetCNNLarge(n_channels=n_channels, **kwargs)
    else:
        raise ValueError(f"Unknown onset model: {name}")
