from __future__ import annotations

import pytest
import torch

from lipidbench.models import build_seed_classification_loss


@pytest.mark.parametrize(
    ("loss_type", "pos_weight"),
    [("bce", None), ("weighted_bce", 2.5), ("focal", None)],
)
def test_seed_losses_are_scalar_finite_and_differentiable(
    loss_type: str,
    pos_weight: float | None,
) -> None:
    logits = torch.tensor([-1.2, 0.4, 1.6], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float32)
    criterion = build_seed_classification_loss(
        loss_type,
        pos_weight=pos_weight,
        focal_alpha=0.25,
        focal_gamma=2.0,
    )

    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
