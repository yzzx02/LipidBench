from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.data import (  # noqa: E402
    AttributePreprocessor,
    PeakMultiTaskDataset,
    collate_peak_multitask_batch,
    load_manifest_jsonl,
)
from lipidbench.models import PeakMultiTaskRCNN  # noqa: E402

from train_peak_multitask_smoke import (  # noqa: E402
    _balanced_smoke_subset,
    _match_detection_counts,
    _move_batch,
)


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm device unavailable; set HSA_ENABLE_DXG_DETECTION=1. "
            "Refusing to silently evaluate on CPU."
        )
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("fusion_mode") != "image_only":
        raise ValueError("this evaluation expects an image_only checkpoint")
    config = copy.deepcopy(checkpoint["config"])
    if args.box_nms_thresh is not None:
        config["model"]["box_nms_thresh"] = args.box_nms_thresh
    model = PeakMultiTaskRCNN.from_config(
        config,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    preprocessor = AttributePreprocessor.load_json(args.preprocessing)
    records = _balanced_smoke_subset(
        load_manifest_jsonl(args.val_manifest),
        args.max_val_samples,
        seed=args.seed + 1,
    )
    dataset = PeakMultiTaskDataset(records, image_root=args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_peak_multitask_batch,
        pin_memory=device.type == "cuda",
    )

    predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    seed_probabilities: list[float] = []
    seed_labels: list[int] = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        attributes = preprocessor.transform(
            batch["attributes"],
            batch["attribute_masks"],
        )
        output = model(
            batch["images"],
            seed_boxes=batch["seed_boxes"],
            attributes=attributes,
        )
        seed_probabilities.extend(
            float(value)
            for value in output["seed_probabilities"].detach().cpu()
        )
        seed_labels.extend(
            int(value) for value in batch["seed_labels"].detach().cpu()
        )
        predictions.extend(
            (
                detection["boxes"].detach().cpu(),
                detection["scores"].detach().cpu(),
                target["boxes"].detach().cpu(),
            )
            for detection, target in zip(
                output["detections"],
                batch["targets"],
                strict=True,
            )
        )

    threshold_results: dict[str, Any] = {}
    for threshold in args.thresholds:
        true_positive = false_positive = false_negative = 0
        for predicted_boxes, predicted_scores, target_boxes in predictions:
            tp, fp, fn = _match_detection_counts(
                predicted_boxes,
                predicted_scores,
                target_boxes,
                score_threshold=threshold,
            )
            true_positive += tp
            false_positive += fp
            false_negative += fn
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        threshold_results[f"{threshold:g}"] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": _safe_divide(
                2 * true_positive,
                2 * true_positive + false_positive + false_negative,
            ),
            "predictions_per_image": _safe_divide(
                true_positive + false_positive,
                len(predictions),
            ),
        }

    seed_predictions = [
        int(probability >= 0.5) for probability in seed_probabilities
    ]
    seed_tp = sum(
        prediction == 1 and label == 1
        for prediction, label in zip(seed_predictions, seed_labels, strict=True)
    )
    seed_fp = sum(
        prediction == 1 and label == 0
        for prediction, label in zip(seed_predictions, seed_labels, strict=True)
    )
    seed_fn = sum(
        prediction == 0 and label == 1
        for prediction, label in zip(seed_predictions, seed_labels, strict=True)
    )
    seed_tn = sum(
        prediction == 0 and label == 0
        for prediction, label in zip(seed_predictions, seed_labels, strict=True)
    )
    result = {
        "status": "ok",
        "checkpoint": str(args.checkpoint.resolve()),
        "fusion_mode": "image_only",
        "box_nms_thresh": float(config["model"]["box_nms_thresh"]),
        "samples": len(records),
        "detection_iou_threshold": 0.5,
        "detection_thresholds": threshold_results,
        "seed_at_0_5": {
            "true_positive": seed_tp,
            "false_positive": seed_fp,
            "false_negative": seed_fn,
            "true_negative": seed_tn,
            "accuracy": _safe_divide(seed_tp + seed_tn, len(seed_labels)),
            "precision": _safe_divide(seed_tp, seed_tp + seed_fp),
            "recall": _safe_divide(seed_tp, seed_tp + seed_fn),
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluate one image-only PeakMultiTaskRCNN checkpoint at several thresholds"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--box-nms-thresh", type=float)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.25, 0.5, 0.75),
    )
    args = parser.parse_args()
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.thresholds):
        raise ValueError("all --thresholds must be in [0, 1]")
    if args.box_nms_thresh is not None and not 0.0 <= args.box_nms_thresh <= 1.0:
        raise ValueError("--box-nms-thresh must be in [0, 1]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    run(parse_args())
