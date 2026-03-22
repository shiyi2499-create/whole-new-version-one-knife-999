"""
Stage 2A: Group Segmentor

Takes the coarse password region from Stage 1 and segments it into
individual password groups (typing segments separated by gaps).

Architecture: Single-stage or Multi-stage TCN with binary output.
Post-processing: median filter → threshold → extract top-K groups.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

from models.tcn import SingleStageTCN, MultiStageTCN
from utils.postprocess import extract_groups_from_probs
from configs.config import Stage2AConfig


class GroupSegmentor(nn.Module):
    """
    Stage 2A model: frame-wise binary classification (typing vs gap).
    """

    def __init__(self, config: Stage2AConfig, use_multistage: bool = False):
        super().__init__()
        self.config = config

        if use_multistage:
            self.tcn = MultiStageTCN(
                input_channels=config.input_channels,
                hidden_channels=config.hidden_channels,
                output_channels=config.num_classes,
                num_layers_gen=config.num_layers,
                num_layers_refine=config.num_layers,
                num_refine_stages=2,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            )
            self.multistage = True
        else:
            self.tcn = SingleStageTCN(
                input_channels=config.input_channels,
                hidden_channels=config.hidden_channels,
                output_channels=config.num_classes,
                num_layers=config.num_layers,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            )
            self.multistage = False

    def forward(self, x):
        """
        x: [B, C, T] preprocessed IMU
        Returns: logits [B, 1, T] (or list of [B, 1, T] for multistage)
        """
        if self.multistage:
            return self.tcn(x)  # list of [B, 1, T]
        else:
            return self.tcn(x)  # [B, 1, T]

    def predict_probs(self, x):
        """Get sigmoid probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if isinstance(logits, list):
                logits = logits[-1]  # last stage
            return torch.sigmoid(logits).squeeze(1)  # [B, T]

    @staticmethod
    def post_process(probs: np.ndarray,
                     sample_rate: int = 100,
                     median_kernel: int = 21,
                     threshold: float = 0.5,
                     min_group_duration_s: float = 0.8,
                     expected_groups: int = 5,
                     ) -> List[Tuple[int, int]]:
        """Delegate to shared utility (see utils/postprocess.py)."""
        return extract_groups_from_probs(
            probs, sample_rate, median_kernel, threshold,
            min_group_duration_s, expected_groups,
        )
