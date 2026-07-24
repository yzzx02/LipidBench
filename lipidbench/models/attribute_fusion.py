from __future__ import annotations

import torch
import torch.nn as nn


VALID_FUSION_MODES = {"image_only", "attr_only", "naive_concat", "gated_fusion"}


class AttributeEncoder(nn.Module):
    """Encode 13- or 15-dimensional seed attributes into a compact vector."""

    def __init__(self, attr_dim: int, *, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        if attr_dim <= 0:
            raise ValueError(f"attr_dim must be positive, got {attr_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.attr_dim = int(attr_dim)
        self.out_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.attr_dim, self.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim, self.out_dim),
            nn.GELU(),
        )

    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        if attributes.ndim != 2 or attributes.shape[1] != self.attr_dim:
            raise ValueError(
                f"attributes must have shape [B, {self.attr_dim}], got {tuple(attributes.shape)}"
            )
        return self.network(attributes)


class GatedSeedFusionHead(nn.Module):
    """Fuse one seed RoI per image with its candidate-specific attributes."""

    def __init__(
        self,
        *,
        roi_channels: int = 256,
        roi_output_size: int = 7,
        attr_dim: int = 13,
        image_embedding_dim: int = 256,
        attr_embedding_dim: int = 64,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
        mode: str = "gated_fusion",
    ) -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in VALID_FUSION_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_FUSION_MODES)}, got {mode!r}")
        for name, value in {
            "roi_channels": roi_channels,
            "roi_output_size": roi_output_size,
            "image_embedding_dim": image_embedding_dim,
            "attr_embedding_dim": attr_embedding_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.mode = mode
        self.roi_channels = int(roi_channels)
        self.roi_output_size = int(roi_output_size)
        self.attr_dim = int(attr_dim)
        self.image_embedding_dim = int(image_embedding_dim)
        self.attr_embedding_dim = int(attr_embedding_dim)

        roi_flat_dim = self.roi_channels * self.roi_output_size * self.roi_output_size
        self.image_encoder = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(roi_flat_dim, self.image_embedding_dim),
            nn.GELU(),
        )
        self.attribute_encoder = AttributeEncoder(
            self.attr_dim,
            hidden_dim=self.attr_embedding_dim,
            dropout=dropout,
        )

        if self.mode == "gated_fusion":
            self.gate: nn.Module | None = nn.Sequential(
                nn.Linear(self.image_embedding_dim + self.attr_embedding_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Linear(fusion_hidden_dim, self.image_embedding_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        if self.mode == "image_only":
            classifier_in = self.image_embedding_dim
        elif self.mode == "attr_only":
            classifier_in = self.attr_embedding_dim
        else:
            classifier_in = self.image_embedding_dim + self.attr_embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, seed_roi_features: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        expected_roi_shape = (
            self.roi_channels,
            self.roi_output_size,
            self.roi_output_size,
        )
        if seed_roi_features.ndim != 4 or tuple(seed_roi_features.shape[1:]) != expected_roi_shape:
            raise ValueError(
                "seed_roi_features must have shape "
                f"[B, {expected_roi_shape[0]}, {expected_roi_shape[1]}, {expected_roi_shape[2]}], "
                f"got {tuple(seed_roi_features.shape)}"
            )
        if attributes.ndim != 2 or attributes.shape != (seed_roi_features.shape[0], self.attr_dim):
            raise ValueError(
                f"attributes must have shape [{seed_roi_features.shape[0]}, {self.attr_dim}], "
                f"got {tuple(attributes.shape)}"
            )

        image_embedding = self.image_encoder(seed_roi_features)
        attr_embedding = self.attribute_encoder(attributes)

        if self.mode == "image_only":
            final_embedding = image_embedding
        elif self.mode == "attr_only":
            final_embedding = attr_embedding
        elif self.mode == "naive_concat":
            final_embedding = torch.cat((image_embedding, attr_embedding), dim=1)
        else:
            if self.gate is None:
                raise RuntimeError("gated_fusion mode requires a gate module")
            combined = torch.cat((image_embedding, attr_embedding), dim=1)
            gated_image = image_embedding * self.gate(combined)
            final_embedding = torch.cat((gated_image, attr_embedding), dim=1)

        return self.classifier(final_embedding).squeeze(1)
