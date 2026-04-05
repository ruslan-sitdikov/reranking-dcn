"""Loss functions for DCN-v2 training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Focal loss for binary classification (Lin et al., ICCV 2017).

    Down-weights well-classified examples so the model focuses on hard
    negatives near the decision boundary -- critical at ~0.04% positive rate.

    Args:
        logits:  Raw model output before sigmoid, shape ``(B,)``.
        targets: Binary targets {0, 1}, shape ``(B,)``.
        gamma:   Focusing parameter.  ``gamma=0`` recovers standard BCE.
        alpha:   Balance factor for positives.  Negatives get ``(1 - alpha)``.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.sigmoid(logits)
    p_t = torch.where(targets > 0.5, p_t, 1.0 - p_t)
    focal_weight = (1.0 - p_t) ** gamma
    alpha_t = torch.where(targets > 0.5, alpha, 1.0 - alpha)
    return (alpha_t * focal_weight * bce).mean()
