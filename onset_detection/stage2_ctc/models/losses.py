"""
Combined Frame-level CE + CTC loss for frame-level character decoding.

Frame-level CE:
  Uses per-key timestamps to directly supervise each frame.
  Keystroke center frames get the character label; all other frames get blank.
  Soft weights (Gaussian around each keystroke) focus the loss on informative frames.

CTC loss:
  Sequence-level consistency: ensures the decoded character sequence matches
  the target, even if individual frame predictions are slightly misaligned.

Design rationale:
  We have per-key timestamps (unlike standard ASR which has no alignment).
  So frame-level CE is the PRIMARY signal — it tells the model exactly where
  each character is. CTC is AUXILIARY — it provides global sequence consistency
  and smooth gradients for alignment refinement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameCTCLoss(nn.Module):

    def __init__(self, num_classes=38, blank_idx=0,
                 frame_weight=1.0, ctc_weight=0.3,
                 blank_ce_weight=0.15, label_smoothing=0.05,
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.blank_idx = blank_idx
        self.frame_weight = frame_weight
        self.ctc_weight = ctc_weight
        self.num_classes = num_classes

        if class_weights is None:
            class_weights = torch.ones(num_classes)
            class_weights[blank_idx] = blank_ce_weight
        else:
            class_weights = class_weights.clone().float()
            class_weights[blank_idx] = blank_ce_weight
        self.register_buffer('class_weights', class_weights)
        self.label_smoothing = label_smoothing

        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    def forward(self, logits, hard_targets, soft_weights, mask,
                ctc_targets, ctc_target_lengths, input_lengths,
                ctc_weight_override=None):
        """
        Args:
            logits:              [B, C, T] raw model output
            hard_targets:        [B, T] int64, per-frame char index (0=blank)
            soft_weights:        [B, T] float32, Gaussian loss weights
            mask:                [B, T] float32, valid frame mask (padding=0)
            ctc_targets:         [sum(target_lengths)] int64, concatenated
            ctc_target_lengths:  [B] int64
            input_lengths:       [B] int64, actual lengths per sample

        Returns:
            dict with 'loss', 'frame_ce', 'ctc'
        """
        B, C, T = logits.shape

        # --- Frame-level CE with per-frame weighting ---
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*T, C]
        targets_flat = hard_targets.reshape(-1)                 # [B*T]
        weights_flat = soft_weights.reshape(-1) * mask.reshape(-1)  # [B*T]

        ce_per_frame = F.cross_entropy(
            logits_flat, targets_flat,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction='none',
        )  # [B*T]

        weighted_ce = (ce_per_frame * weights_flat).sum() / (weights_flat.sum() + 1e-8)

        # --- CTC loss ---
        # CTC expects [T, B, C] log-probs
        log_probs = F.log_softmax(logits, dim=1).permute(2, 0, 1)  # [T, B, C]

        ctc = self.ctc_loss(
            log_probs,
            ctc_targets,
            input_lengths,
            ctc_target_lengths,
        )

        # Guard against NaN from degenerate CTC situations
        if torch.isnan(ctc) or torch.isinf(ctc):
            ctc = torch.tensor(0.0, device=logits.device)

        ctc_weight = self.ctc_weight if ctc_weight_override is None else float(ctc_weight_override)
        total = self.frame_weight * weighted_ce + ctc_weight * ctc

        return {
            'loss': total,
            'frame_ce': weighted_ce.detach(),
            'ctc': ctc.detach(),
            'ctc_weight': torch.tensor(ctc_weight, device=logits.device),
        }
