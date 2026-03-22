"""
Temporal Convolutional Network building blocks.

Based on MS-TCN (Farha & Gall, CVPR 2019) and MS-TCN++ (Li et al., TPAMI 2020).
Adapted for 1D IMU signal processing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DilatedResidualLayer(nn.Module):
    """
    Single dilated residual layer.
    Conv1D(dilation=d) → ReLU → Conv1D(1x1) → Dropout → Residual add
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # same padding

        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        """
        x: [B, C, T]
        """
        residual = x
        out = self.conv_dilated(x)
        out = F.relu(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return self.norm(out + residual)


class DualDilatedLayer(nn.Module):
    """
    Dual Dilated Layer from MS-TCN++.
    Uses two dilations: 2^l and 2^(L-l) to capture both local and global features.
    """

    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation_small: int = 1, dilation_large: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        pad_small = dilation_small * (kernel_size - 1) // 2
        pad_large = dilation_large * (kernel_size - 1) // 2

        self.conv_small = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad_small, dilation=dilation_small
        )
        self.conv_large = nn.Conv1d(
            channels, channels, kernel_size,
            padding=pad_large, dilation=dilation_large
        )
        self.conv_fuse = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out_s = F.relu(self.conv_small(x))
        out_l = F.relu(self.conv_large(x))
        out = out_s + out_l
        out = self.conv_fuse(out)
        out = self.dropout(out)
        return self.norm(out + residual)


class SingleStageTCN(nn.Module):
    """
    Single-stage TCN with stacked dilated residual layers.
    This is the core building block for both Stage 2A and Stage 2B.
    """

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int = 64,
                 output_channels: int = 1,
                 num_layers: int = 10,
                 kernel_size: int = 3,
                 dropout: float = 0.3,
                 use_dual_dilated: bool = False):
        super().__init__()

        # Input projection
        self.input_conv = nn.Conv1d(input_channels, hidden_channels, 1)
        self.input_norm = nn.BatchNorm1d(hidden_channels)

        # Stacked dilated layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i

            if use_dual_dilated:
                dilation_large = 2 ** (num_layers - 1 - i)
                self.layers.append(DualDilatedLayer(
                    hidden_channels, kernel_size,
                    dilation_small=dilation,
                    dilation_large=dilation_large,
                    dropout=dropout
                ))
            else:
                self.layers.append(DilatedResidualLayer(
                    hidden_channels, kernel_size,
                    dilation=dilation, dropout=dropout
                ))

        # Output projection
        self.output_conv = nn.Conv1d(hidden_channels, output_channels, 1)

    def forward(self, x):
        """
        x: [B, C_in, T]
        Returns: [B, C_out, T]
        """
        out = F.relu(self.input_norm(self.input_conv(x)))

        for layer in self.layers:
            out = layer(out)

        return self.output_conv(out)


class MultiStageTCN(nn.Module):
    """
    Multi-stage TCN (MS-TCN style).
    First stage: prediction generator.
    Subsequent stages: refinement stages that take previous stage's output.
    """

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int = 64,
                 output_channels: int = 1,
                 num_layers_gen: int = 10,
                 num_layers_refine: int = 10,
                 num_refine_stages: int = 2,
                 kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()

        # Prediction generator (first stage)
        self.generator = SingleStageTCN(
            input_channels, hidden_channels, output_channels,
            num_layers_gen, kernel_size, dropout,
            use_dual_dilated=True
        )

        # Refinement stages
        self.refinement_stages = nn.ModuleList()
        for _ in range(num_refine_stages):
            # Refinement takes previous output + original features
            self.refinement_stages.append(SingleStageTCN(
                output_channels + hidden_channels,  # prev output + features
                hidden_channels, output_channels,
                num_layers_refine, kernel_size, dropout,
                use_dual_dilated=False
            ))

        # Feature extractor (shared) for refinement input
        self.feature_extractor = nn.Conv1d(input_channels, hidden_channels, 1)

    def forward(self, x):
        """
        x: [B, C_in, T]
        Returns: list of [B, C_out, T] predictions from each stage
        """
        features = F.relu(self.feature_extractor(x))

        # First stage
        out = self.generator(x)
        outputs = [out]

        # Refinement stages
        for stage in self.refinement_stages:
            # Concatenate previous prediction with features
            refined_input = torch.cat([F.sigmoid(out), features], dim=1)
            out = stage(refined_input)
            outputs.append(out)

        return outputs
