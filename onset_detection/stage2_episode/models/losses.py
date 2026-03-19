"""
Dual-head loss for 2-class frame-level + onset-impulse prediction.

Loss = typing_loss + onset_weight * onset_loss

typing_loss: weighted CE + temporal smoothing (MS-TCN-style TMSE).
onset_loss:  sparse onset supervision tuned for "few sharp peaks" rather than
             a broad activation plateau.

Why this version:
  The previous onset loss was plain full-frame MSE against a 20ms Gaussian.
  On real data that encouraged the model to light up wide regions inside an
  episode, because being "a bit high everywhere" can still keep MSE low.

  The new loss separates three pressures:
    1. weighted BCE on a hard peak mask      -> make true key centers stand out
    2. local shape MSE on positive support   -> keep each peak centered/narrow
    3. sparsity penalty on negative frames   -> push background back to zero
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodeFrameLoss(nn.Module):
    """
    Dual-head loss:
      - typing_loss: weighted CE + TMSE smoothing (for episode boundary learning)
      - onset_loss:  masked MSE against Gaussian impulse targets (for key localization)

    If onset targets are not provided (None), only the typing loss is computed.
    This preserves backward compatibility with old training data that has no
    onset_targets field.
    """

    def __init__(self, class_weights=(1.0, 4.0), smooth_weight=0.15,
                 smooth_tau=4.0, onset_weight=2.0,
                 onset_pos_thresh=0.55, onset_support_thresh=0.10,
                 onset_pos_weight=6.0, onset_neg_weight=1.0,
                 onset_shape_weight=0.35, onset_sparse_weight=0.25):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer('weights', w)
        self.smooth_weight = smooth_weight
        self.tau = smooth_tau
        self.onset_weight = onset_weight
        self.onset_pos_thresh = onset_pos_thresh
        self.onset_support_thresh = onset_support_thresh
        self.onset_pos_weight = onset_pos_weight
        self.onset_neg_weight = onset_neg_weight
        self.onset_shape_weight = onset_shape_weight
        self.onset_sparse_weight = onset_sparse_weight

    def forward(self, typing_logits, targets, mask, onset_logits=None,
                onset_targets=None):
        """
        typing_logits:  [B, 2, T]
        targets:        [B, T] long (0=silence, 1=typing)
        mask:           [B, T] float
        onset_logits:   [B, 1, T] or None
        onset_targets:  [B, T] float in [0, 1] (Gaussian impulse), or None
        """
        B, C, T = typing_logits.shape

        # --- Typing head loss (unchanged) ---
        logits_flat = typing_logits.permute(0, 2, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        mask_flat = mask.reshape(-1)

        ce = F.cross_entropy(logits_flat, targets_flat,
                             weight=self.weights, reduction='none')
        ce = (ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        # Temporal smoothing: penalize rapid label changes
        log_probs = F.log_softmax(typing_logits, dim=1)
        diff = (log_probs[:, :, 1:] - log_probs[:, :, :-1]) ** 2
        diff = torch.clamp(diff.mean(dim=1), max=self.tau)
        mask_adj = mask[:, 1:] * mask[:, :-1]
        smooth = (diff * mask_adj).sum() / (mask_adj.sum() + 1e-8)

        typing_loss = ce + self.smooth_weight * smooth

        # --- Onset head loss ---
        onset_loss = torch.tensor(0.0, device=typing_logits.device)
        if onset_logits is not None and onset_targets is not None:
            onset_probs = torch.sigmoid(onset_logits.squeeze(1))  # [B, T]

            # Hard peak mask: only the center of each Gaussian should really be
            # "on". This prevents the model from learning a wide elevated band.
            peak_mask = (onset_targets >= self.onset_pos_thresh).float()
            support_mask = (onset_targets >= self.onset_support_thresh).float()
            neg_mask = mask * (1.0 - support_mask)

            eps = 1e-6
            bce_pos = -(peak_mask * torch.log(onset_probs + eps))
            bce_neg = -((1.0 - peak_mask) * torch.log(1.0 - onset_probs + eps))
            bce = (
                self.onset_pos_weight * (bce_pos * mask).sum() +
                self.onset_neg_weight * (bce_neg * mask).sum()
            ) / (mask.sum() + 1e-8)

            # Keep the predicted peak shape locally aligned only near true peaks.
            sq_err = (onset_probs - onset_targets) ** 2
            shape_loss = (sq_err * support_mask * mask).sum() / (
                (support_mask * mask).sum() + 1e-8
            )

            # Explicitly punish non-zero background away from peak support.
            sparse_loss = (onset_probs * neg_mask).sum() / (neg_mask.sum() + 1e-8)

            onset_loss = bce + self.onset_shape_weight * shape_loss + self.onset_sparse_weight * sparse_loss

        return typing_loss + self.onset_weight * onset_loss, {
            'typing_loss': typing_loss.item(),
            'onset_loss': onset_loss.item() if isinstance(onset_loss, torch.Tensor) else 0.0,
        }
