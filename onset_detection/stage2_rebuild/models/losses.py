"""
Loss functions for Stage 2A and Stage 2B.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error loss from MS-TCN.
    Penalizes over-segmentation by encouraging smooth predictions.
    L_smooth = (1/T) * sum_t clamp(|log(p_t) - log(p_{t-1})|^2, 0, τ)
    """

    def __init__(self, tau: float = 4.0):
        super().__init__()
        self.tau = tau

    def forward(self, pred: torch.Tensor, mask: torch.Tensor = None):
        """
        pred: [B, T] probabilities (after sigmoid)
        mask: [B, T] valid mask (1=valid, 0=padding)
        """
        # Log probabilities (clamp for numerical stability)
        log_pred = torch.log(torch.clamp(pred, min=1e-7))

        # Temporal difference
        diff = (log_pred[:, 1:] - log_pred[:, :-1]) ** 2
        diff = torch.clamp(diff, max=self.tau)

        if mask is not None:
            # Use minimum of adjacent masks
            valid = mask[:, 1:] * mask[:, :-1]
            diff = diff * valid
            return diff.sum() / (valid.sum() + 1e-8)
        else:
            return diff.mean()


class Stage2ALoss(nn.Module):
    """
    Combined loss for Stage 2A: Group Segmentation.
    L = bce_weight * BCE + smoothing_weight * TMSE
    """

    def __init__(self, bce_weight: float = 1.0, smoothing_weight: float = 0.15):
        super().__init__()
        self.bce_weight = bce_weight
        self.smoothing_weight = smoothing_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.tmse = TMSELoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor = None):
        """
        logits: [B, 1, T] raw logits from model
        targets: [B, T] binary labels
        mask: [B, T] valid mask

        Can also accept list of logits (multi-stage), in which case
        loss is summed over all stages.
        """
        if isinstance(logits, list):
            total_loss = torch.tensor(0.0, device=targets.device)
            for stage_logits in logits:
                total_loss = total_loss + self._single_stage_loss(
                    stage_logits, targets, mask
                )
            return total_loss / len(logits)
        else:
            return self._single_stage_loss(logits, targets, mask)

    def _single_stage_loss(self, logits, targets, mask):
        logits_flat = logits.squeeze(1)  # [B, T]

        # BCE loss
        bce_loss = self.bce(logits_flat, targets)
        if mask is not None:
            bce_loss = (bce_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            bce_loss = bce_loss.mean()

        # Smoothing loss
        probs = torch.sigmoid(logits_flat)
        smooth_loss = self.tmse(probs, mask)

        return self.bce_weight * bce_loss + self.smoothing_weight * smooth_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in onset detection.
    Most frames are non-onset, so standard MSE/BCE is dominated by negatives.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor = None):
        """
        logits: [B, 1, T]
        targets: [B, T] Gaussian peak targets in [0, 1]
        mask: [B, T]
        """
        pred = torch.sigmoid(logits.squeeze(1))  # [B, T]

        # Binary focal loss adapted for soft targets
        bce = F.binary_cross_entropy(pred, targets, reduction='none')

        # Focal weight: higher weight for hard examples
        pt = targets * pred + (1 - targets) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting: higher weight for positive (onset) frames
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        loss = alpha_t * focal_weight * bce

        if mask is not None:
            loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


class Stage2BLoss(nn.Module):
    """
    Combined loss for Stage 2B: Onset Detection.
    Options: MSE, BCE, or Focal loss on Gaussian targets.
    """

    def __init__(self, use_focal: bool = True,
                 focal_alpha: float = 0.75,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.use_focal = use_focal

        if use_focal:
            self.loss_fn = FocalLoss(focal_alpha, focal_gamma)
        else:
            self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor = None):
        """
        logits: [B, 1, T]
        targets: [B, T] Gaussian peak targets
        mask: [B, T]
        """
        if self.use_focal:
            return self.loss_fn(logits, targets, mask)
        else:
            pred = torch.sigmoid(logits.squeeze(1))
            loss = self.loss_fn(pred, targets)
            if mask is not None:
                loss = (loss * mask).sum() / (mask.sum() + 1e-8)
            else:
                loss = loss.mean()
            return loss
