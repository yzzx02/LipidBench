from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import FeaturePyramidNetwork


class ConvNeXtTinyFPNBackbone(nn.Module):
    """ConvNeXt-Tiny stages C2-C5 followed by a 256-channel FPN.

    The returned ordered dictionary uses ``p2`` through ``p5`` keys. Input
    height and width are intentionally not fixed; TorchVision's detection
    transform is responsible for batching differently sized images.
    """

    stage_channels = (96, 192, 384, 768)

    def __init__(self, *, pretrained: bool = True, out_channels: int = 256) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        base = convnext_tiny(weights=weights)
        self.feature_extractor = create_feature_extractor(
            base,
            return_nodes={
                "features.1": "c2",
                "features.3": "c3",
                "features.5": "c4",
                "features.7": "c5",
            },
        )
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=list(self.stage_channels),
            out_channels=int(out_channels),
        )
        # Required by torchvision.models.detection.FasterRCNN.
        self.out_channels = int(out_channels)

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, C, H, W], got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"ConvNeXt-Tiny expects 3 input channels, got {images.shape[1]}")

        stage_features = self.feature_extractor(images)
        pyramid = self.fpn(stage_features)
        return OrderedDict(
            (
                ("p2", pyramid["c2"]),
                ("p3", pyramid["c3"]),
                ("p4", pyramid["c4"]),
                ("p5", pyramid["c5"]),
            )
        )
