"""
Configuration for Stage 2 Open: variable-length password stream recovery.
No fixed password count or length assumptions.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SignalConfig:
    sample_rate: int = 100
    num_channels: int = 6           # accel_xyz + gyro_xyz
    use_magnitude: bool = True      # → 8 channels total
    normalize: bool = True

    @property
    def input_channels(self) -> int:
        return self.num_channels + (2 if self.use_magnitude else 0)


@dataclass
class SynthesisConfig:
    """Variable-length synthetic session generation."""
    num_sessions: int = 300
    # Password count per session: uniform [min, max]
    min_passwords: int = 2
    max_passwords: int = 8
    # Password length (characters): uniform [min, max]
    min_password_len: int = 4
    max_password_len: int = 12
    # Inter-password gap (seconds)
    gap_min_s: float = 0.5
    gap_max_s: float = 3.0
    # Context padding before/after password block
    context_min_s: float = 0.2
    context_max_s: float = 2.0
    # Augmentation
    time_stretch_range: Tuple[float, float] = (0.9, 1.1)
    gain_range: Tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.02
    splice_smooth_samples: int = 10
    # Frame-label parameters
    keystroke_label_radius_ms: float = 30.0  # how wide the '1' label is around each press


@dataclass
class ModelConfig:
    """Frame-wise 3-class TCN."""
    input_channels: int = 8
    hidden_channels: int = 64
    num_layers: int = 10
    kernel_size: int = 3
    dropout: float = 0.3
    num_classes: int = 3            # 0=gap, 1=keystroke, 2=separator


@dataclass
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 120
    patience: int = 15
    # Loss weights for class imbalance: [gap, keystroke, separator]
    # keystroke frames are rare, separator less rare, gap dominant
    class_weights: Tuple[float, float, float] = (1.0, 5.0, 3.0)
    smoothing_weight: float = 0.15


@dataclass
class DecoderConfig:
    """Rule-based decoder: frame labels → onsets + group boundaries."""
    min_keystroke_run: int = 2          # minimum consecutive '1' frames to count as onset
    min_separator_run_ms: float = 150   # minimum separator duration to split groups
    min_gap_between_onsets_ms: float = 40   # merge onsets closer than this
    # No expected_groups or expected_onsets — fully open
