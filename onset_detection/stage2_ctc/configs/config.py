"""
Configuration for Stage 2 CTC: frame-level character decoding.

Core change from stage2_episode:
  - No onset detection, no per-key window cutting
  - Model outputs P(char|frame) at every frame
  - CTC loss + frame-level CE loss with per-key timestamp supervision
  - Decoding via CTC greedy/beam search
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SignalConfig:
    sample_rate: int = 100          # Mac internal IMU typical rate
    num_channels: int = 6           # accel_xyz + gyro_xyz
    use_magnitude: bool = True      # -> 8 channels total
    normalize: bool = True

    @property
    def input_channels(self) -> int:
        return self.num_channels + (2 if self.use_magnitude else 0)


@dataclass
class ModelConfig:
    """Frame-level character TCN."""
    input_channels: int = 8
    hidden_channels: int = 128      # wider than onset TCN (was 64)
    num_layers: int = 12            # deeper than onset TCN (was 10)
    kernel_size: int = 3
    dropout: float = 0.25
    num_classes: int = 38           # blank + 26 letters + 10 digits + unk


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 200
    patience: int = 30

    # Loss weights
    frame_ce_weight: float = 1.0    # frame-level CE (primary, uses per-key ts)
    ctc_weight: float = 0.05        # keep CE dominant early; CTC is auxiliary
    label_smoothing: float = 0.01
    blank_ce_weight: float = 0.02   # blank frames are extremely common
    ctc_warmup_epochs: int = 3      # frame-only warmup
    ctc_ramp_epochs: int = 6        # then ramp CTC in gradually

    # Gaussian sigma for frame-level soft weights around keystroke
    # At 100Hz, 20ms = 2 frames
    keystroke_sigma_ms: float = 20.0

    # Backbone init from existing onset checkpoint (optional)
    onset_checkpoint: str = ''
    resume_checkpoint: str = ''
    freeze_backbone: bool = False


@dataclass
class DataConfig:
    """Dataset building parameters."""
    # Episode extraction margin around first/last keystroke
    episode_margin_ms: float = 300.0
    # Padding before/after password block
    pad_s: float = 0.5
    # Synthesis
    num_synth_sessions: int = 600
    min_passwords: int = 1
    max_passwords: int = 6
    min_password_len: int = 4
    max_password_len: int = 12
    inter_pw_gap_min_s: float = 0.8
    inter_pw_gap_max_s: float = 4.0
    context_min_s: float = 0.3
    context_max_s: float = 2.0
    # Augmentation
    time_stretch_range: Tuple[float, float] = (0.85, 1.15)
    gain_range: Tuple[float, float] = (0.7, 1.3)
    noise_std: float = 0.03
