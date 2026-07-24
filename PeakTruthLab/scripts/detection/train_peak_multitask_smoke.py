from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.data import (  # noqa: E402
    AttributePreprocessor,
    PeakManifestRecord,
    PeakMultiTaskDataset,
    collate_peak_multitask_batch,
    load_manifest_jsonl,
)
from lipidbench.models import PeakMultiTaskRCNN  # noqa: E402


def _find_default_manifest(filename: str) -> Path | None:
    candidates = sorted(
        (PROJECT_ROOT / "PeakTruthLab" / "datasets").glob(f"*/训练清单/{filename}")
    )
    return candidates[0] if len(candidates) == 1 else None


def _balanced_smoke_subset(
    records: Iterable[PeakManifestRecord],
    limit: int,
    *,
    seed: int,
) -> list[PeakManifestRecord]:
    materialised = list(records)
    if limit <= 0 or limit >= len(materialised):
        return materialised

    rng = random.Random(seed)
    shuffled = materialised[:]
    rng.shuffle(shuffled)
    selected: list[PeakManifestRecord] = []
    selected_ids: set[str] = set()
    predicates = (
        lambda record: record.seed_label == 0,
        lambda record: record.seed_label == 1,
        lambda record: len(record.boxes) == 0,
        lambda record: len(record.boxes) == 1,
        lambda record: len(record.boxes) > 1,
    )
    for predicate in predicates:
        match = next(
            (
                record
                for record in shuffled
                if record.sample_id not in selected_ids and predicate(record)
            ),
            None,
        )
        if match is not None and len(selected) < limit:
            selected.append(match)
            selected_ids.add(match.sample_id)
    for record in shuffled:
        if len(selected) >= limit:
            break
        if record.sample_id not in selected_ids:
            selected.append(record)
            selected_ids.add(record.sample_id)
    return selected


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        **batch,
        "images": [image.to(device) for image in batch["images"]],
        "targets": [
            {name: value.to(device) for name, value in target.items()}
            for target in batch["targets"]
        ],
        "seed_boxes": [box.to(device) for box in batch["seed_boxes"]],
        "attributes": batch["attributes"].to(device),
        "attribute_masks": batch["attribute_masks"].to(device),
        "seed_labels": batch["seed_labels"].to(device),
    }


def _selected_loss(
    losses: dict[str, torch.Tensor],
    task: str,
) -> torch.Tensor:
    if task == "joint":
        selected = list(losses.values())
    elif task == "detection":
        selected = [
            value for name, value in losses.items() if name != "loss_seed_cls"
        ]
    elif task == "seed":
        selected = [losses["loss_seed_cls"]]
    else:
        raise ValueError(f"unsupported task: {task}")
    if not selected:
        raise RuntimeError(f"task {task!r} selected no losses")
    return torch.stack(selected).sum()


def _train_one_epoch(
    model: PeakMultiTaskRCNN,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    preprocessor: AttributePreprocessor,
    device: torch.device,
    *,
    task: str,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        attributes = preprocessor.transform(
            batch["attributes"],
            batch["attribute_masks"],
        )
        optimizer.zero_grad(set_to_none=True)
        losses = model(
            batch["images"],
            targets=batch["targets"],
            seed_boxes=batch["seed_boxes"],
            attributes=attributes,
            seed_labels=batch["seed_labels"],
        )
        loss = _selected_loss(losses, task)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss: {losses}")
        loss.backward()
        optimizer.step()
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu())
        totals["selected_total"] += float(loss.detach().cpu())
        batches += 1
    if batches == 0:
        raise RuntimeError("training loader produced no batches")
    return {name: value / batches for name, value in sorted(totals.items())}


def _match_detection_counts(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    keep = predicted_scores >= score_threshold
    predicted_boxes = predicted_boxes[keep]
    if predicted_boxes.numel() == 0:
        return 0, 0, int(target_boxes.shape[0])
    if target_boxes.numel() == 0:
        return 0, int(predicted_boxes.shape[0]), 0

    overlaps = box_iou(predicted_boxes, target_boxes)
    matched_targets: set[int] = set()
    true_positive = 0
    for prediction_index in range(predicted_boxes.shape[0]):
        values = overlaps[prediction_index]
        order = torch.argsort(values, descending=True)
        for raw_target_index in order:
            target_index = int(raw_target_index.item())
            if target_index in matched_targets:
                continue
            if float(values[target_index].item()) < iou_threshold:
                break
            matched_targets.add(target_index)
            true_positive += 1
            break
    false_positive = int(predicted_boxes.shape[0]) - true_positive
    false_negative = int(target_boxes.shape[0]) - true_positive
    return true_positive, false_positive, false_negative


@torch.no_grad()
def _evaluate(
    model: PeakMultiTaskRCNN,
    loader: DataLoader[dict[str, Any]],
    preprocessor: AttributePreprocessor,
    device: torch.device,
    *,
    score_threshold: float,
) -> dict[str, Any]:
    model.eval()
    seed_probabilities: list[float] = []
    seed_labels: list[int] = []
    detection_tp = detection_fp = detection_fn = 0
    images = 0
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
        probabilities = output["seed_probabilities"].detach().cpu()
        labels = batch["seed_labels"].detach().cpu()
        seed_probabilities.extend(float(value) for value in probabilities)
        seed_labels.extend(int(value) for value in labels)
        for detection, target in zip(
            output["detections"],
            batch["targets"],
            strict=True,
        ):
            tp, fp, fn = _match_detection_counts(
                detection["boxes"].detach().cpu(),
                detection["scores"].detach().cpu(),
                target["boxes"].detach().cpu(),
                score_threshold=score_threshold,
            )
            detection_tp += tp
            detection_fp += fp
            detection_fn += fn
            images += 1

    seed_predictions = [int(value >= 0.5) for value in seed_probabilities]
    seed_correct = sum(
        prediction == label
        for prediction, label in zip(seed_predictions, seed_labels, strict=True)
    )
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

    def safe_divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "samples": images,
        "seed": {
            "accuracy_at_0_5": safe_divide(seed_correct, len(seed_labels)),
            "precision_at_0_5": safe_divide(seed_tp, seed_tp + seed_fp),
            "recall_at_0_5": safe_divide(seed_tp, seed_tp + seed_fn),
            "positive_labels": sum(seed_labels),
            "negative_labels": len(seed_labels) - sum(seed_labels),
            "probability_min": min(seed_probabilities),
            "probability_max": max(seed_probabilities),
        },
        "detection": {
            "score_threshold": score_threshold,
            "iou_threshold": 0.5,
            "true_positive": detection_tp,
            "false_positive": detection_fp,
            "false_negative": detection_fn,
            "precision": safe_divide(
                detection_tp,
                detection_tp + detection_fp,
            ),
            "recall": safe_divide(
                detection_tp,
                detection_tp + detection_fn,
            ),
        },
    }


def _dataset_summary(records: list[PeakManifestRecord]) -> dict[str, int]:
    return {
        "samples": len(records),
        "seed_positive": sum(record.seed_label == 1 for record in records),
        "seed_negative": sum(record.seed_label == 0 for record in records),
        "detection_empty": sum(not record.boxes for record in records),
        "detection_single": sum(len(record.boxes) == 1 for record in records),
        "detection_multi": sum(len(record.boxes) > 1 for record in records),
        "detection_boxes": sum(len(record.boxes) for record in records),
    }


def _select_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm/CUDA device was requested but unavailable. In WSL set "
            "HSA_ENABLE_DXG_DETECTION=1; refusing to silently use CPU."
        )
    return torch.device(raw)


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = _select_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_records_full = load_manifest_jsonl(args.train_manifest)
    val_records_full = load_manifest_jsonl(args.val_manifest)
    preprocessor = AttributePreprocessor.fit(train_records_full)
    train_records = _balanced_smoke_subset(
        train_records_full,
        args.max_train_samples,
        seed=args.seed,
    )
    val_records = _balanced_smoke_subset(
        val_records_full,
        args.max_val_samples,
        seed=args.seed + 1,
    )

    train_dataset = PeakMultiTaskDataset(
        train_records,
        image_root=args.image_root,
    )
    val_dataset = PeakMultiTaskDataset(
        val_records,
        image_root=args.image_root,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=collate_peak_multitask_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_peak_multitask_batch,
        pin_memory=device.type == "cuda",
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    model_cfg = config["model"]
    model_cfg["fusion_mode"] = "image_only"
    model_cfg["image_min_size"] = args.image_size
    model_cfg["image_max_size"] = args.image_size
    model_cfg["rpn_pre_nms_top_n_train"] = min(
        int(model_cfg["rpn_pre_nms_top_n_train"]),
        args.rpn_train_proposals,
    )
    model_cfg["rpn_pre_nms_top_n_test"] = min(
        int(model_cfg["rpn_pre_nms_top_n_test"]),
        args.rpn_test_proposals,
    )
    model_cfg["rpn_post_nms_top_n_train"] = min(
        int(model_cfg["rpn_post_nms_top_n_train"]),
        args.rpn_train_proposals,
    )
    model_cfg["rpn_post_nms_top_n_test"] = min(
        int(model_cfg["rpn_post_nms_top_n_test"]),
        args.rpn_test_proposals,
    )
    model_cfg["rpn_batch_size_per_image"] = min(
        int(model_cfg["rpn_batch_size_per_image"]),
        128,
    )
    model_cfg["box_batch_size_per_image"] = min(
        int(model_cfg["box_batch_size_per_image"]),
        64,
    )
    model_cfg["box_detections_per_img"] = min(
        int(model_cfg["box_detections_per_img"]),
        50,
    )
    if config["loss"].get("seed_pos_weight") is None:
        positives = sum(record.seed_label == 1 for record in train_records_full)
        negatives = len(train_records_full) - positives
        config["loss"]["seed_pos_weight"] = negatives / positives

    model = PeakMultiTaskRCNN.from_config(
        config,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{summary_path} already exists; choose a new output directory "
            "or pass --overwrite"
        )
    preprocessor.save_json(args.output_dir / "attribute_preprocessing.json")

    started = time.perf_counter()
    epoch_history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        train_losses = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            preprocessor,
            device,
            task=args.task,
        )
        epoch_history.append({"epoch": epoch + 1, "train_losses": train_losses})
        print(
            json.dumps(
                epoch_history[-1],
                ensure_ascii=False,
                allow_nan=False,
            ),
            flush=True,
        )

    validation = _evaluate(
        model,
        val_loader,
        preprocessor,
        device,
        score_threshold=args.score_threshold,
    )
    checkpoint_path = args.output_dir / "last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": args.epochs,
            "task": args.task,
            "fusion_mode": "image_only",
        },
        checkpoint_path,
    )
    summary: dict[str, Any] = {
        "status": "ok",
        "purpose": "real-data small-sample pipeline smoke test",
        "device": {
            "type": str(device),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else "CPU"
            ),
        },
        "task": args.task,
        "fusion_mode": "image_only",
        "attribute_fusion_enabled": False,
        "attribute_preprocessing": preprocessor.to_mapping(),
        "data": {
            "train_manifest": str(args.train_manifest.resolve()),
            "val_manifest": str(args.val_manifest.resolve()),
            "image_root": str(args.image_root.resolve()),
            "train_full_samples_used_to_fit_preprocessing": len(train_records_full),
            "train_smoke": _dataset_summary(train_records),
            "val_smoke": _dataset_summary(val_records),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "learning_rate": args.learning_rate,
            "history": epoch_history,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "validation": validation,
        "outputs": {
            "checkpoint": str(checkpoint_path.resolve()),
            "attribute_preprocessing": str(
                (args.output_dir / "attribute_preprocessing.json").resolve()
            ),
        },
    }
    if device.type == "cuda":
        summary["device"]["max_memory_allocated_bytes"] = (
            torch.cuda.max_memory_allocated()
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


def parse_args() -> argparse.Namespace:
    default_train = _find_default_manifest("train.jsonl")
    default_val = _find_default_manifest("val.jsonl")
    parser = argparse.ArgumentParser(
        "Real-data, image-only PeakMultiTaskRCNN small-sample test"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "PeakTruthLab" / "configs" / "peak_multitask.yaml",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=default_train,
        required=default_train is None,
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=default_val,
        required=default_val is None,
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "PeakTruthLab"
            / "datasets"
            / "eic_images_flat"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "PeakTruthLab"
            / "results"
            / "smoke_image_only"
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--task",
        choices=["joint", "detection", "seed"],
        default="joint",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=24)
    parser.add_argument("--max-val-samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--rpn-train-proposals", type=int, default=128)
    parser.add_argument("--rpn-test-proposals", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.image_size < 64:
        raise ValueError("--image-size must be at least 64")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    return args


if __name__ == "__main__":
    run(parse_args())
