from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.models import PeakMultiTaskRCNN


def _target(boxes: list[list[float]]) -> dict[str, torch.Tensor]:
    if boxes:
        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
    else:
        box_tensor = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    return {"boxes": box_tensor, "labels": labels}


def _build_random_batch(
    image_size: int,
    attr_dim: int,
) -> tuple[
    list[torch.Tensor],
    list[dict[str, torch.Tensor]],
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    s = float(image_size)
    images = [torch.rand(3, image_size, image_size) for _ in range(4)]
    targets = [
        _target([]),
        _target([[0.35 * s, 0.10 * s, 0.55 * s, 0.90 * s]]),
        _target(
            [
                [0.10 * s, 0.08 * s, 0.28 * s, 0.92 * s],
                [0.62 * s, 0.12 * s, 0.82 * s, 0.88 * s],
            ]
        ),
        # A real peak is far from this false seed candidate.
        _target([[0.68 * s, 0.10 * s, 0.88 * s, 0.90 * s]]),
    ]
    seed_boxes = [
        torch.tensor([[0.40 * s, 0.15 * s, 0.55 * s, 0.85 * s]], dtype=torch.float32),
        torch.tensor([[0.35 * s, 0.10 * s, 0.55 * s, 0.90 * s]], dtype=torch.float32),
        torch.tensor([[0.10 * s, 0.08 * s, 0.28 * s, 0.92 * s]], dtype=torch.float32),
        torch.tensor([[0.08 * s, 0.15 * s, 0.24 * s, 0.85 * s]], dtype=torch.float32),
    ]
    attributes = torch.randn(4, attr_dim)
    seed_labels = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    return images, targets, seed_boxes, attributes, seed_labels


def _select_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available")
    return torch.device(raw)


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = _select_device(args.device)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    model_cfg = config["model"]
    model_cfg["attr_dim"] = int(args.attr_dim)
    model_cfg["image_min_size"] = int(args.image_size)
    model_cfg["image_max_size"] = int(args.image_size)
    # Keep the CPU smoke test fast while retaining the configured anchors and heads.
    model_cfg["rpn_pre_nms_top_n_train"] = min(int(model_cfg["rpn_pre_nms_top_n_train"]), 64)
    model_cfg["rpn_pre_nms_top_n_test"] = min(int(model_cfg["rpn_pre_nms_top_n_test"]), 32)
    model_cfg["rpn_post_nms_top_n_train"] = min(int(model_cfg["rpn_post_nms_top_n_train"]), 32)
    model_cfg["rpn_post_nms_top_n_test"] = min(int(model_cfg["rpn_post_nms_top_n_test"]), 16)
    model_cfg["rpn_batch_size_per_image"] = 64
    model_cfg["box_batch_size_per_image"] = 32
    model_cfg["box_detections_per_img"] = 10
    model_cfg["box_score_thresh"] = 0.0

    # Explicit False prevents any pretrained-weight download in this smoke test.
    model = PeakMultiTaskRCNN.from_config(config, pretrained=False).to(device)

    model.eval()
    with torch.no_grad():
        probe = torch.randn(2, 3, 256, 256, device=device)
        stages = model.backbone.feature_extractor(probe)
        raw_pyramid = model.backbone.fpn(stages)
    print(f"device={device}")
    print("ConvNeXt-Tiny stage shapes:")
    for name, feature in stages.items():
        print(f"  {name}: {tuple(feature.shape)}")
    print("FPN shapes:")
    pyramid_names = {"c2": "p2", "c3": "p3", "c4": "p4", "c5": "p5"}
    if tuple(raw_pyramid) != tuple(pyramid_names):
        raise RuntimeError(f"unexpected FPN outputs: {tuple(raw_pyramid)}")
    for name, feature in raw_pyramid.items():
        print(f"  {pyramid_names[name]}: {tuple(feature.shape)}")
    del probe, stages, raw_pyramid

    images, targets, seed_boxes, attributes, seed_labels = _build_random_batch(
        args.image_size,
        args.attr_dim,
    )
    images = [image.to(device) for image in images]
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    seed_boxes = [box.to(device) for box in seed_boxes]
    attributes = attributes.to(device)
    seed_labels = seed_labels.to(device)

    model.train()
    losses = model(
        images,
        targets=targets,
        seed_boxes=seed_boxes,
        attributes=attributes,
        seed_labels=seed_labels,
    )
    total_loss = sum(losses.values())
    total_loss.backward()
    print("Training losses:")
    for name, value in losses.items():
        print(f"  {name}: {float(value.detach().cpu()):.6f}")
    print(f"  total: {float(total_loss.detach().cpu()):.6f}")
    print("backward: ok")

    model.eval()
    with torch.no_grad():
        output = model(
            images,
            seed_boxes=seed_boxes,
            attributes=attributes,
        )
    print(f"seed_probabilities: {output['seed_probabilities'].detach().cpu().tolist()}")
    print("Detection output shapes:")
    for index, detection in enumerate(output["detections"]):
        print(
            f"  image[{index}]: boxes={tuple(detection['boxes'].shape)}, "
            f"scores={tuple(detection['scores'].shape)}, labels={tuple(detection['labels'].shape)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Random-data smoke test for PeakMultiTaskRCNN")
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            PROJECT_ROOT / "PeakTruthLab" / "configs" / "peak_multitask.yaml"
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--attr-dim", type=int, choices=[13, 15], default=13)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.image_size < 64:
        raise ValueError("--image-size must be at least 64 for the ConvNeXt feature hierarchy")
    return args


if __name__ == "__main__":
    run(parse_args())
