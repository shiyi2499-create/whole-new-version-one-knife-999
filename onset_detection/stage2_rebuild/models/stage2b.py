"""
Stage 2B: Onset Detector

Takes a single password group segment and detects exactly K key onsets.

Architecture: Small TCN with sigmoid output → Gaussian peak prediction.
Post-processing: constrained peak picking to extract exactly K onsets.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from models.tcn import SingleStageTCN
from utils.postprocess import pick_onset_peaks
from configs.config import Stage2BConfig


class OnsetDetector(nn.Module):
    """
    Stage 2B model: frame-wise onset probability prediction.
    """

    def __init__(self, config: Stage2BConfig):
        super().__init__()
        self.config = config

        self.tcn = SingleStageTCN(
            input_channels=config.input_channels,
            hidden_channels=config.hidden_channels,
            output_channels=1,
            num_layers=config.num_layers,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )

    def forward(self, x):
        """
        x: [B, C, T_group]
        Returns: logits [B, 1, T_group]
        """
        return self.tcn(x)

    def predict_probs(self, x):
        """Get sigmoid probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(1)  # [B, T]

    @staticmethod
    def pick_peaks(probs: np.ndarray,
                   expected_onsets: int = 8,
                   min_iki_samples: int = 5,
                   base_threshold: float = 0.3,
                   fallback_thresholds: Optional[List[float]] = None,
                   ) -> np.ndarray:
        """Delegate to shared utility (see utils/postprocess.py)."""
        return pick_onset_peaks(
            probs, expected_onsets, min_iki_samples,
            base_threshold, fallback_thresholds,
        )
