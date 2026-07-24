from __future__ import annotations

import torch

from lipidbench.models import ConvNeXtTinyFPNBackbone


def test_convnext_tiny_fpn_shapes_are_finite_and_multiscale() -> None:
    torch.manual_seed(0)
    backbone = ConvNeXtTinyFPNBackbone(pretrained=False, out_channels=256).eval()
    images = torch.randn(1, 3, 480, 480)

    with torch.no_grad():
        features = backbone(images)

    assert list(features) == ["p2", "p3", "p4", "p5"]
    assert {name: tuple(feature.shape) for name, feature in features.items()} == {
        "p2": (1, 256, 120, 120),
        "p3": (1, 256, 60, 60),
        "p4": (1, 256, 30, 30),
        "p5": (1, 256, 15, 15),
    }
    assert all(torch.isfinite(feature).all() for feature in features.values())
