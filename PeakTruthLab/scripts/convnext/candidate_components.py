"""Components for the current 16-attribute candidate-classification experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

@dataclass
class AttrScaler:
    fill: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]


class VisionBackbone(nn.Module):
    """ConvNeXt-Tiny image encoder used by the current classification runs."""

    def __init__(self, vision_backbone: str = "convnext_tiny", pretrained: bool = True) -> None:
        super().__init__()
        if vision_backbone != "convnext_tiny":
            raise ValueError("the maintained backbone is convnext_tiny")
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        base = convnext_tiny(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.out_dim = base.classifier[2].in_features
        self.vision_backbone = vision_backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.features(image)), 1)



class FusionModel(nn.Module):
    def __init__(
        self,
        attr_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        pretrained: bool = True,
        vision_backbone: str = "convnext_tiny",
        model_mode: str = "naive_concat",
    ) -> None:
        super().__init__()
        if attr_dim != 16:
            raise ValueError("candidate classification requires 16 attributes")
        mode = str(model_mode).strip().lower()
        valid_modes = {"image_only", "attr_only", "naive_concat", "gated_fusion"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported model_mode={model_mode}, choose from {sorted(valid_modes)}")
        self.model_mode = mode

        if self.model_mode != "attr_only":
            self.backbone: VisionBackbone | None = VisionBackbone(
                vision_backbone=vision_backbone,
                pretrained=pretrained,
            )
            in_dim = self.backbone.out_dim
        else:
            self.backbone = None
            in_dim = 0

        if self.model_mode != "image_only":
            self.attr_encoder: nn.Sequential | None = nn.Sequential(
                nn.Linear(attr_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 64),
                nn.GELU(),
            )
        else:
            self.attr_encoder = None

        if self.model_mode == "gated_fusion":
            self.gate: nn.Sequential | None = nn.Sequential(
                nn.Linear(in_dim + 64, 256),
                nn.GELU(),
                nn.Linear(256, in_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        if self.model_mode == "image_only":
            classifier_in = in_dim
        elif self.model_mode == "attr_only":
            classifier_in = 64
        else:
            classifier_in = in_dim + 64

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, image: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        if self.model_mode == "image_only":
            if self.backbone is None:
                raise RuntimeError("image_only mode requires backbone")
            img_feat = self.backbone(image)
            return self.classifier(img_feat)

        if self.model_mode == "attr_only":
            if self.attr_encoder is None:
                raise RuntimeError("attr_only mode requires attr_encoder")
            attr_feat = self.attr_encoder(attrs)
            return self.classifier(attr_feat)

        if self.backbone is None or self.attr_encoder is None:
            raise RuntimeError(f"mode={self.model_mode} requires both image and attr branches")

        img_feat = self.backbone(image)
        attr_feat = self.attr_encoder(attrs)

        if self.model_mode == "naive_concat":
            final_feat = torch.cat([img_feat, attr_feat], dim=1)
            return self.classifier(final_feat)

        if self.gate is None:
            raise RuntimeError("gated_fusion mode requires gate")
        fused = torch.cat([img_feat, attr_feat], dim=1)
        gate = self.gate(fused)
        gated_img_feat = img_feat * gate
        final_feat = torch.cat([gated_img_feat, attr_feat], dim=1)
        return self.classifier(final_feat)


def build_attr_scaler_from_train_csv(train_csv: Path, attr_columns: Sequence[str]) -> AttrScaler:
    df = pd.read_csv(train_csv)
    missing = [c for c in attr_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing attribute columns in {train_csv}: {missing}")

    fill: dict[str, float] = {}
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for c in attr_columns:
        s = pd.to_numeric(df[c], errors="coerce")
        med = float(s.median()) if np.isfinite(s.median()) else 0.0
        s2 = s.fillna(med)
        mu = float(s2.mean())
        sigma = float(s2.std(ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-8:
            sigma = 1.0
        fill[c] = med
        mean[c] = mu
        std[c] = sigma
    return AttrScaler(fill=fill, mean=mean, std=std)


def build_eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
