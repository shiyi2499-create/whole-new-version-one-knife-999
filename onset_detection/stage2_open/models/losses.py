"""
Loss for 3-class frame-level prediction with temporal smoothing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenFrameLoss(nn.Module):
    """
    Weighted CE + temporal smoothing (MS-TCN-style TMSE).
    class_weights: [gap, keystroke, separator] to handle imbalance.
    """

    def __init__(self, class_weights=(1.0, 5.0, 3.0), smooth_weight=0.15,
                 smooth_tau=4.0):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer('weights', w)
        self.smooth_weight = smooth_weight
        self.tau = smooth_tau

    def forward(self, logits, targets, mask):
        """
        logits:  [B, 3, T]
        targets: [B, T] long
        mask:    [B, T] float
        """
        B, C, T = logits.shape

        # --- weighted cross-entropy ---
        # Reshape for F.cross_entropy: [B*T, C] vs [B*T]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)  # [B*T, C]
        targets_flat = targets.reshape(-1)                     # [B*T]
        mask_flat = mask.reshape(-1)                           # [B*T]

        ce = F.cross_entropy(logits_flat, targets_flat,
                             weight=self.weights, reduction='none')  # [B*T]
        ce = (ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        # --- temporal smoothing ---
        log_probs = F.log_softmax(logits, dim=1)  # [B, C, T]
        diff = (log_probs[:, :, 1:] - log_probs[:, :, :-1]) ** 2  # [B, C, T-1]
        diff = torch.clamp(diff.mean(dim=1), max=self.tau)  # [B, T-1]

        mask_adj = mask[:, 1:] * mask[:, :-1]
        smooth = (diff * mask_adj).sum() / (mask_adj.sum() + 1e-8)

        return ce + self.smooth_weight * smooth
