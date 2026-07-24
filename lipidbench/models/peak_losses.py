from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Binary focal loss operating directly on logits."""

    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be none|mean|sum, got {reduction!r}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"logits and targets must have the same shape, got {tuple(logits.shape)} and {tuple(targets.shape)}"
            )
        targets = targets.to(dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_seed_classification_loss(
    loss_type: str,
    *,
    pos_weight: float | None = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Build BCE, positive-weighted BCE, or focal loss for the seed branch."""

    normalized = str(loss_type).strip().lower()
    if normalized == "bce":
        if pos_weight is not None:
            raise ValueError("pos_weight is only valid with loss_type='weighted_bce'")
        return nn.BCEWithLogitsLoss()
    if normalized == "weighted_bce":
        weight_tensor = None
        if pos_weight is not None:
            if float(pos_weight) <= 0.0:
                raise ValueError(f"pos_weight must be positive, got {pos_weight}")
            weight_tensor = torch.tensor(float(pos_weight), dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    if normalized == "focal":
        if pos_weight is not None:
            raise ValueError("pos_weight is not used by focal loss; configure focal_alpha instead")
        return BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction="mean")
    raise ValueError(f"Unsupported seed loss type {loss_type!r}; choose bce|weighted_bce|focal")
