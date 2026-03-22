"""
Hyperparameter configurations for Stage 2A and Stage 2B.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SignalConfig:
    """IMU signal processing parameters."""
    sample_rate: int = 190          # Hz - aligned with the main repo protocol
    num_channels: int = 6           # accel_xyz + gyro_xyz
    normalize: bool = True          # per-channel z-score normalization
    bandpass_low: float = 0.5       # Hz, high-pass cutoff
    bandpass_high: float = 45.0     # Hz, low-pass cutoff
    use_magnitude: bool = True      # append ||accel|| and ||gyro|| as extra channels
    # If use_magnitude=True, effective channels = 8


@dataclass
class SynthesisConfig:
    """Synthetic mixed session generation parameters."""
    num_sessions: int = 200
    passwords_per_session: int = 5
    keys_per_password: int = 8
    # Gap between passwords (seconds)
    gap_duration_min: float = 0.5
    gap_duration_max: float = 3.0
    # Prefix/suffix idle duration (seconds)
    context_duration_min: float = 0.3
    context_duration_max: float = 2.0
    # Overlap-add smoothing at splice points (samples)
    splice_smooth_samples: int = 10
    # Random gain augmentation
    gain_augment_range: tuple = (0.8, 1.2)
    # Time stretch augmentation
    time_stretch_range: tuple = (0.9, 1.1)
    # Noise injection std
    noise_std: float = 0.02


@dataclass
class Stage2AConfig:
    """Stage 2A: Group Segmentor (TCN) configuration."""
    # Model
    input_channels: int = 8         # 6 raw + 2 magnitude (if use_magnitude)
    hidden_channels: int = 64
    num_layers: int = 10            # dilated conv layers
    kernel_size: int = 3
    dropout: float = 0.3
    # Output
    num_classes: int = 1            # binary: typing vs gap

    # Training
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 100
    patience: int = 15              # early stopping

    # Loss
    bce_weight: float = 1.0
    smoothing_weight: float = 0.15  # TMSE smoothing loss (MS-TCN style)

    # Post-processing
    median_filter_kernel: int = 21
    threshold: float = 0.5
    min_group_duration_s: float = 0.8   # minimum password duration in seconds
    expected_groups: int = 5


@dataclass
class Stage2BConfig:
    """Stage 2B: Onset Detector (TCN + Gaussian peak) configuration."""
    # Model
    input_channels: int = 8
    hidden_channels: int = 64
    num_layers: int = 6             # smaller receptive field needed
    kernel_size: int = 3
    dropout: float = 0.3

    # Gaussian target
    gaussian_sigma_ms: float = 15.0  # sigma in milliseconds
    # Will be converted to samples based on sample_rate

    # Training
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 150
    patience: int = 20

    # Loss
    use_focal_loss: bool = True
    focal_alpha: float = 0.75       # weight for positive class
    focal_gamma: float = 2.0

    # Peak picking
    min_iki_ms: float = 50.0        # minimum inter-keystroke interval
    peak_height_threshold: float = 0.3
    expected_onsets: int = 8
    # Fallback thresholds if we get != 8 peaks
    fallback_thresholds: List[float] = field(
        default_factory=lambda: [0.2, 0.15, 0.1, 0.05]
    )


@dataclass
class PipelineConfig:
    """Full E2E pipeline configuration."""
    signal: SignalConfig = field(default_factory=SignalConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    stage2a: Stage2AConfig = field(default_factory=Stage2AConfig)
    stage2b: Stage2BConfig = field(default_factory=Stage2BConfig)

    # Paths (override at runtime)
    password_data_dir: str = "data/raw/password/len_8"
    negative_data_dir: str = "data/raw/onset_negative"
    synthetic_data_dir: str = "data/processed/stage2_synthetic_mixed"
    mixed_training_dir: str = "data/raw/mixed_training"
    mixed2_dir: str = "data/raw/onset_mixed2"
    output_dir: str = "results/stage2_rebuild"
