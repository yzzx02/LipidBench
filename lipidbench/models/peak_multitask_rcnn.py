from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .attribute_fusion import GatedSeedFusionHead
from .convnext_fpn import ConvNeXtTinyFPNBackbone
from .peak_losses import build_seed_classification_loss


def resize_seed_boxes(
    boxes: torch.Tensor,
    original_size: tuple[int, int],
    new_size: tuple[int, int],
) -> torch.Tensor:
    """Scale ``[x1, y1, x2, y2]`` boxes between image sizes.

    Sizes are ``(height, width)``. The function is intentionally independent
    of any fixed image resolution and is used after GeneralizedRCNNTransform.
    """

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must have shape [N, 4], got {tuple(boxes.shape)}")
    if not boxes.is_floating_point():
        raise ValueError("boxes must use a floating-point dtype")
    old_h, old_w = int(original_size[0]), int(original_size[1])
    new_h, new_w = int(new_size[0]), int(new_size[1])
    if min(old_h, old_w, new_h, new_w) <= 0:
        raise ValueError(f"image sizes must be positive, got {original_size} -> {new_size}")

    scales = boxes.new_tensor(
        (new_w / old_w, new_h / old_h, new_w / old_w, new_h / old_h)
    )
    return boxes * scales


class PeakMultiTaskRCNN(nn.Module):
    """Shared ConvNeXt-FPN detector with candidate-specific seed validation.

    The full-window Faster R-CNN branch detects only ``True_Peak`` foreground
    objects (class 1; class 0 is implicit background). The seed branch pools
    one externally supplied candidate box per image from the *same* FPN
    features, then optionally fuses an arbitrary positive-dimensional
    attribute vector. Current experiments compare 13- and 15-dimensional
    variants.
    """

    def __init__(
        self,
        *,
        anchor_sizes: Sequence[Sequence[int]],
        anchor_aspect_ratios: Sequence[Sequence[float]],
        attr_dim: int = 13,
        num_classes: int = 2,
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        image_min_size: int = 480,
        image_max_size: int = 480,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        rpn_nms_thresh: float = 0.7,
        rpn_pre_nms_top_n_train: int = 1000,
        rpn_pre_nms_top_n_test: int = 500,
        rpn_post_nms_top_n_train: int = 500,
        rpn_post_nms_top_n_test: int = 250,
        rpn_batch_size_per_image: int = 256,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.6,
        box_detections_per_img: int = 100,
        box_batch_size_per_image: int = 256,
        seed_roi_output_size: int = 7,
        seed_roi_sampling_ratio: int = 2,
        seed_image_embedding_dim: int = 256,
        attr_embedding_dim: int = 64,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.2,
        fusion_mode: str = "gated_fusion",
        seed_loss_type: str = "weighted_bce",
        seed_loss_weight: float = 1.0,
        seed_pos_weight: float | None = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError(
                "PeakMultiTaskRCNN fixes num_classes=2: background plus True_Peak foreground"
            )
        if attr_dim <= 0:
            raise ValueError(f"attr_dim must be a positive integer, got {attr_dim}")
        if seed_loss_weight < 0.0:
            raise ValueError(f"seed_loss_weight must be non-negative, got {seed_loss_weight}")
        if len(anchor_sizes) != 4 or len(anchor_aspect_ratios) != 4:
            raise ValueError("anchor_sizes and anchor_aspect_ratios must each define P2-P5 (4 levels)")

        sizes = tuple(tuple(int(v) for v in level) for level in anchor_sizes)
        ratios = tuple(tuple(float(v) for v in level) for level in anchor_aspect_ratios)
        if any(not level or any(v <= 0 for v in level) for level in sizes):
            raise ValueError("every anchor size level must contain positive values")
        if any(not level or any(v <= 0 for v in level) for level in ratios):
            raise ValueError("every anchor aspect-ratio level must contain positive values")

        backbone = ConvNeXtTinyFPNBackbone(
            pretrained=pretrained,
            out_channels=fpn_out_channels,
        )

        # TorchVision defines aspect ratio as height / width. Internally it
        # computes h=sqrt(ratio)*scale and w=scale/sqrt(ratio), so ratios above
        # 1.0 create tall/narrow anchors suitable for chromatographic boxes.
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
        detection_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=7,
            sampling_ratio=2,
        )
        self.detector = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            min_size=int(image_min_size),
            max_size=int(image_max_size),
            image_mean=list(float(v) for v in image_mean),
            image_std=list(float(v) for v in image_std),
            rpn_anchor_generator=anchor_generator,
            rpn_pre_nms_top_n_train=int(rpn_pre_nms_top_n_train),
            rpn_pre_nms_top_n_test=int(rpn_pre_nms_top_n_test),
            rpn_post_nms_top_n_train=int(rpn_post_nms_top_n_train),
            rpn_post_nms_top_n_test=int(rpn_post_nms_top_n_test),
            rpn_nms_thresh=float(rpn_nms_thresh),
            rpn_batch_size_per_image=int(rpn_batch_size_per_image),
            box_roi_pool=detection_roi_pool,
            box_score_thresh=float(box_score_thresh),
            box_nms_thresh=float(box_nms_thresh),
            box_detections_per_img=int(box_detections_per_img),
            box_batch_size_per_image=int(box_batch_size_per_image),
        )

        self.seed_roi_pool = MultiScaleRoIAlign(
            featmap_names=["p2", "p3", "p4", "p5"],
            output_size=int(seed_roi_output_size),
            sampling_ratio=int(seed_roi_sampling_ratio),
        )
        self.seed_fusion_head = GatedSeedFusionHead(
            roi_channels=fpn_out_channels,
            roi_output_size=seed_roi_output_size,
            attr_dim=attr_dim,
            image_embedding_dim=seed_image_embedding_dim,
            attr_embedding_dim=attr_embedding_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            mode=fusion_mode,
        )
        self.seed_loss_fn = build_seed_classification_loss(
            seed_loss_type,
            pos_weight=seed_pos_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        self.attr_dim = int(attr_dim)
        self.seed_loss_weight = float(seed_loss_weight)

    @property
    def backbone(self) -> nn.Module:
        """Expose the single shared backbone without registering it twice."""

        return self.detector.backbone

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        pretrained: bool | None = None,
    ) -> "PeakMultiTaskRCNN":
        model_cfg = dict(config.get("model", {}))
        loss_cfg = dict(config.get("loss", {}))
        if "anchor_sizes" not in model_cfg or "anchor_aspect_ratios" not in model_cfg:
            raise ValueError("config.model must define anchor_sizes and anchor_aspect_ratios")
        configured_pretrained = bool(model_cfg.get("pretrained", True))
        return cls(
            anchor_sizes=model_cfg["anchor_sizes"],
            anchor_aspect_ratios=model_cfg["anchor_aspect_ratios"],
            attr_dim=int(model_cfg.get("attr_dim", 13)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            pretrained=configured_pretrained if pretrained is None else bool(pretrained),
            fpn_out_channels=int(model_cfg.get("fpn_out_channels", 256)),
            image_min_size=int(model_cfg.get("image_min_size", 480)),
            image_max_size=int(model_cfg.get("image_max_size", 480)),
            image_mean=model_cfg.get("image_mean", (0.485, 0.456, 0.406)),
            image_std=model_cfg.get("image_std", (0.229, 0.224, 0.225)),
            rpn_nms_thresh=float(model_cfg.get("rpn_nms_thresh", 0.7)),
            rpn_pre_nms_top_n_train=int(model_cfg.get("rpn_pre_nms_top_n_train", 1000)),
            rpn_pre_nms_top_n_test=int(model_cfg.get("rpn_pre_nms_top_n_test", 500)),
            rpn_post_nms_top_n_train=int(model_cfg.get("rpn_post_nms_top_n_train", 500)),
            rpn_post_nms_top_n_test=int(model_cfg.get("rpn_post_nms_top_n_test", 250)),
            rpn_batch_size_per_image=int(model_cfg.get("rpn_batch_size_per_image", 256)),
            box_score_thresh=float(model_cfg.get("box_score_thresh", 0.05)),
            box_nms_thresh=float(model_cfg.get("box_nms_thresh", 0.6)),
            box_detections_per_img=int(model_cfg.get("box_detections_per_img", 100)),
            box_batch_size_per_image=int(model_cfg.get("box_batch_size_per_image", 256)),
            seed_roi_output_size=int(model_cfg.get("seed_roi_output_size", 7)),
            seed_roi_sampling_ratio=int(model_cfg.get("seed_roi_sampling_ratio", 2)),
            seed_image_embedding_dim=int(model_cfg.get("seed_image_embedding_dim", 256)),
            attr_embedding_dim=int(model_cfg.get("attr_embedding_dim", 64)),
            fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256)),
            fusion_dropout=float(model_cfg.get("fusion_dropout", 0.2)),
            fusion_mode=str(model_cfg.get("fusion_mode", "gated_fusion")),
            seed_loss_type=str(loss_cfg.get("seed_loss_type", "weighted_bce")),
            seed_loss_weight=float(loss_cfg.get("seed_loss_weight", 1.0)),
            seed_pos_weight=loss_cfg.get("seed_pos_weight"),
            focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
            focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        )

    def _validate_inputs(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
        seed_boxes: list[torch.Tensor] | None,
        attributes: torch.Tensor | None,
        seed_labels: torch.Tensor | None,
    ) -> None:
        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("images must be a non-empty list[Tensor[C,H,W]]")
        batch_size = len(images)
        for index, image in enumerate(images):
            if not isinstance(image, torch.Tensor) or image.ndim != 3:
                raise ValueError(f"images[{index}] must be Tensor[C,H,W], got {type(image)}")
            if image.shape[0] != 3:
                raise ValueError(f"images[{index}] must have 3 channels, got {tuple(image.shape)}")
            if not image.is_floating_point():
                raise ValueError(f"images[{index}] must be a floating-point tensor")
            if not torch.isfinite(image).all():
                raise ValueError(f"images[{index}] contains NaN or Inf")

        if seed_boxes is None or len(seed_boxes) != batch_size:
            raise ValueError(f"seed_boxes must contain one Tensor[1,4] per image (B={batch_size})")
        for index, (box, image) in enumerate(zip(seed_boxes, images, strict=True)):
            if not isinstance(box, torch.Tensor) or box.shape != (1, 4):
                shape = tuple(box.shape) if isinstance(box, torch.Tensor) else None
                raise ValueError(f"seed_boxes[{index}] must have shape [1,4], got {shape}")
            if not box.is_floating_point() or not torch.isfinite(box).all():
                raise ValueError(f"seed_boxes[{index}] must be finite floating-point coordinates")
            self._validate_box_bounds(box, image.shape[-2:], f"seed_boxes[{index}]")

        if not isinstance(attributes, torch.Tensor) or attributes.shape != (batch_size, self.attr_dim):
            shape = tuple(attributes.shape) if isinstance(attributes, torch.Tensor) else None
            raise ValueError(f"attributes must have shape [{batch_size}, {self.attr_dim}], got {shape}")
        if not attributes.is_floating_point() or not torch.isfinite(attributes).all():
            raise ValueError("attributes must be a finite floating-point tensor")

        if self.training:
            if targets is None or len(targets) != batch_size:
                raise ValueError(f"training requires one target dict per image (B={batch_size})")
            for index, (target, image) in enumerate(zip(targets, images, strict=True)):
                if not isinstance(target, dict) or "boxes" not in target or "labels" not in target:
                    raise ValueError(f"targets[{index}] must contain boxes and labels")
                boxes, labels = target["boxes"], target["labels"]
                if not isinstance(boxes, torch.Tensor) or boxes.ndim != 2 or boxes.shape[1] != 4:
                    raise ValueError(f"targets[{index}]['boxes'] must have shape [N,4]")
                if not boxes.is_floating_point() or not torch.isfinite(boxes).all():
                    raise ValueError(f"targets[{index}]['boxes'] must be finite floating-point coordinates")
                if not isinstance(labels, torch.Tensor) or labels.shape != (boxes.shape[0],):
                    raise ValueError(f"targets[{index}]['labels'] must have shape [{boxes.shape[0]}]")
                if labels.dtype != torch.int64:
                    raise ValueError(f"targets[{index}]['labels'] must use torch.int64")
                if labels.numel() and not torch.all(labels == 1):
                    raise ValueError("all non-background detection target labels must equal 1 (True_Peak)")
                self._validate_box_bounds(boxes, image.shape[-2:], f"targets[{index}]['boxes']")

            if not isinstance(seed_labels, torch.Tensor) or seed_labels.shape != (batch_size,):
                shape = tuple(seed_labels.shape) if isinstance(seed_labels, torch.Tensor) else None
                raise ValueError(f"seed_labels must have shape [{batch_size}] during training, got {shape}")
            if not torch.isfinite(seed_labels).all() or not torch.all((seed_labels == 0) | (seed_labels == 1)):
                raise ValueError("seed_labels must contain only binary values 0 or 1")

    @staticmethod
    def _validate_box_bounds(boxes: torch.Tensor, image_size: Sequence[int], name: str) -> None:
        if boxes.numel() == 0:
            return
        height, width = int(image_size[0]), int(image_size[1])
        if torch.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError(f"{name} contains a degenerate box with non-positive width or height")
        if torch.any(boxes[:, 0] < 0) or torch.any(boxes[:, 1] < 0):
            raise ValueError(f"{name} contains negative coordinates")
        if torch.any(boxes[:, 2] > width) or torch.any(boxes[:, 3] > height):
            raise ValueError(f"{name} exceeds image bounds (height={height}, width={width})")

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
        seed_boxes: list[torch.Tensor] | None = None,
        attributes: torch.Tensor | None = None,
        seed_labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        self._validate_inputs(images, targets, seed_boxes, attributes, seed_labels)
        assert seed_boxes is not None
        assert attributes is not None

        original_image_sizes = [tuple(int(v) for v in image.shape[-2:]) for image in images]
        image_list, transformed_targets = self.detector.transform(images, targets)

        # Exactly one shared ConvNeXt-Tiny + FPN forward for both tasks.
        features = self.detector.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("p2", features)])

        proposals, proposal_losses = self.detector.rpn(
            image_list,
            features,
            transformed_targets,
        )
        detections, detector_losses = self.detector.roi_heads(
            features,
            proposals,
            image_list.image_sizes,
            transformed_targets,
        )

        feature_device = next(iter(features.values())).device
        scaled_seed_boxes = [
            resize_seed_boxes(
                box.to(device=feature_device),
                original_size=old_size,
                new_size=new_size,
            )
            for box, old_size, new_size in zip(
                seed_boxes,
                original_image_sizes,
                image_list.image_sizes,
                strict=True,
            )
        ]
        seed_roi_features = self.seed_roi_pool(
            features,
            scaled_seed_boxes,
            image_list.image_sizes,
        )
        seed_logits = self.seed_fusion_head(
            seed_roi_features,
            attributes.to(device=feature_device, dtype=seed_roi_features.dtype),
        )

        if self.training:
            assert seed_labels is not None
            seed_targets = seed_labels.to(device=seed_logits.device, dtype=seed_logits.dtype)
            loss_seed_cls = self.seed_loss_fn(seed_logits, seed_targets) * self.seed_loss_weight
            losses: dict[str, torch.Tensor] = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_seed_cls"] = loss_seed_cls
            return losses

        detections = self.detector.transform.postprocess(
            detections,
            image_list.image_sizes,
            original_image_sizes,
        )
        return {
            "detections": detections,
            "seed_logits": seed_logits,
            "seed_probabilities": torch.sigmoid(seed_logits),
        }
