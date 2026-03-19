"""
Configuration for Stage 2 Episode: password episode detection + onset recovery.

Key design change from stage2_open:
  - Frame model: 2-class (typing vs silence) instead of 3-class
  - Episode detection: rule-based temporal clustering on typing runs
  - No 'separator' class — the model only needs to learn what typing looks like

Rationale:
  'separator' vs 'gap' have identical IMU signatures (both are silence).
  The only difference is duration, which is better handled as a post-hoc rule
  than forced into the frame classifier.
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
    num_sessions: int = 400
    # Password count per session: uniform [min, max]
    min_passwords: int = 1
    max_passwords: int = 8
    # Password length (characters): uniform [min, max]
    min_password_len: int = 4
    max_password_len: int = 16
    # Inter-password gap (seconds) — this becomes the episode boundary signal
    inter_pw_gap_min_s: float = 0.8
    inter_pw_gap_max_s: float = 4.0
    # Intra-password inter-keystroke gap variation
    # (baked into real segments, but for synthetic onset timing)
    intra_key_gap_min_ms: float = 80.0
    intra_key_gap_max_ms: float = 400.0
    # Context padding before/after entire password block
    context_min_s: float = 0.3
    context_max_s: float = 3.0
    # Augmentation
    time_stretch_range: Tuple[float, float] = (0.85, 1.15)
    gain_range: Tuple[float, float] = (0.7, 1.3)
    noise_std: float = 0.03
    splice_smooth_samples: int = 10
    # Frame-label parameters: how wide the typing=1 label is around each press
    keystroke_label_radius_ms: float = 40.0
    # Onset Gaussian target: keep this deliberately narrow so the onset head
    # learns sparse key centers instead of a broad activation band.
    # At 100Hz, 12ms ~= 1.2 frames.
    onset_gaussian_sigma_ms: float = 12.0


@dataclass
class ModelConfig:
    """Frame-wise 2-class TCN: typing vs silence."""
    input_channels: int = 8
    hidden_channels: int = 64
    num_layers: int = 10
    kernel_size: int = 3
    dropout: float = 0.3
    num_classes: int = 2            # 0=silence, 1=typing


@dataclass
class TrainConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 150
    patience: int = 20
    # Loss weights: [silence, typing]. Typing frames are rarer.
    class_weights: Tuple[float, float] = (1.0, 4.0)
    smoothing_weight: float = 0.15
    # Stronger onset supervision: the onset head is now sparse BCE + local-shape
    # regression, so we can give it a bit more weight without washing out the
    # typing head.
    onset_loss_weight: float = 3.0


@dataclass
class EpisodeConfig:
    """
    Rule-based episode detector: frame predictions → password episodes.

    The core idea:
      1. Find contiguous runs of typing=1 frames ("typing runs").
      2. Merge typing runs separated by < merge_gap_ms (intra-password pauses).
      3. The merged runs are candidate episodes.
      4. Within each episode, find individual onset peaks.

    episode_gap_ms is the KEY threshold:
      - Gaps shorter than this between typing runs → same episode (intra-password)
      - Gaps longer than this → different episodes (inter-password)

    This replaces the old 'separator' class entirely.
    """
    # Smoothing
    median_kernel: int = 7

    # Minimum typing run to be non-spurious
    min_typing_run_ms: float = 30.0

    # THE critical threshold: max silence within one password episode
    # Gaps shorter than this are merged into the same episode.
    # Typical inter-keystroke gap during password typing: 100-400ms
    # Typical inter-password gap: 800ms - 3s+
    # So 500-700ms is a good boundary.
    episode_gap_ms: float = 600.0

    # Onset detection within episodes
    min_onset_gap_ms: float = 50.0     # minimum inter-onset distance
    min_episode_keys: int = 2           # discard episodes with fewer onsets

    # Optional: minimum episode duration to keep
    min_episode_duration_ms: float = 200.0
