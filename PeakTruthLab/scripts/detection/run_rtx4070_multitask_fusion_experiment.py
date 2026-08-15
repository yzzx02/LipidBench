from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

try:
    import fcntl
except ImportError:  # pragma: no cover - the training target is WSL/Linux
    fcntl = None

PROJECT_ROOT = Path(
    os.environ.get("LIPIDBENCH_PROJECT_ROOT", r"D:\CODE\LipidBench")
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DETECTION_SCRIPT_DIR = PROJECT_ROOT / "PeakTruthLab" / "scripts" / "detection"
if str(DETECTION_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DETECTION_SCRIPT_DIR))

from lipidbench.data import (  # noqa: E402
    AttributePreprocessor,
    PeakManifestRecord,
    PeakMultiTaskDataset,
    collate_peak_multitask_batch,
    load_manifest_jsonl,
)
from lipidbench.models import PeakMultiTaskRCNN  # noqa: E402

from train_peak_multitask_smoke import (  # noqa: E402
    _balanced_smoke_subset,
    _move_batch,
)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_ranking_metrics(
    probabilities: list[float],
    labels: list[int],
) -> tuple[float, float]:
    ranked = sorted(
        zip(probabilities, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0, 0.0
    true_positives = false_positives = 0
    previous_tpr = previous_fpr = 0.0
    auroc = average_precision = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        group_positive = group_negative = 0
        while index < len(ranked) and ranked[index][0] == score:
            if ranked[index][1] == 1:
                group_positive += 1
            else:
                group_negative += 1
            index += 1
        true_positives += group_positive
        false_positives += group_negative
        tpr = true_positives / positives
        fpr = false_positives / negatives
        auroc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        if group_positive:
            precision = true_positives / (true_positives + false_positives)
            average_precision += precision * (tpr - previous_tpr)
        previous_tpr = tpr
        previous_fpr = fpr
    return float(auroc), float(average_precision)


def _expected_calibration_error(
    probabilities: list[float],
    labels: list[int],
    *,
    bins: int = 10,
) -> float:
    error = 0.0
    total = len(labels)
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        indices = [
            index
            for index, probability in enumerate(probabilities)
            if probability >= lower
            and (
                probability < upper
                or (bin_index == bins - 1 and probability <= upper)
            )
        ]
        if not indices:
            continue
        confidence = sum(probabilities[index] for index in indices) / len(indices)
        accuracy = sum(labels[index] for index in indices) / len(indices)
        error += len(indices) / total * abs(accuracy - confidence)
    return float(error)


def _accelerator_backend(device: torch.device) -> str:
    if device.type != "cuda":
        return device.type
    return "rocm" if torch.version.hip is not None else "cuda"


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


def _selected_loss(
    losses: dict[str, torch.Tensor],
    *,
    training_task: str,
) -> torch.Tensor:
    expected = {
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_classifier",
        "loss_box_reg",
        "loss_seed_cls",
    }
    missing = sorted(expected.difference(losses))
    if missing:
        raise RuntimeError(f"model did not return expected losses: {missing}")
    if training_task == "joint":
        selected = sorted(expected)
    elif training_task == "detection":
        selected = sorted(expected - {"loss_seed_cls"})
    elif training_task == "seed":
        selected = ["loss_seed_cls"]
    else:
        raise ValueError(f"unsupported training task: {training_task!r}")
    return torch.stack([losses[name] for name in selected]).sum()


def _train_one_epoch(
    model: PeakMultiTaskRCNN,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    preprocessor: AttributePreprocessor,
    device: torch.device,
    *,
    amp_enabled: bool,
    gradient_accumulation_steps: int,
    scaler: torch.amp.GradScaler,
    training_task: str,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    loader_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for batch_index, raw_batch in enumerate(loader):
        attributes = preprocessor.transform(
            raw_batch["attributes"],
            raw_batch["attribute_masks"],
        ).to(device)
        batch = _move_batch(raw_batch, device)
        group_start = (
            batch_index // gradient_accumulation_steps
        ) * gradient_accumulation_steps
        group_size = min(
            gradient_accumulation_steps,
            loader_batches - group_start,
        )
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            losses = model(
                batch["images"],
                targets=batch["targets"],
                seed_boxes=batch["seed_boxes"],
                attributes=attributes,
                seed_labels=batch["seed_labels"],
            )
            total_loss = _selected_loss(
                losses,
                training_task=training_task,
            )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite training loss: {losses}")
        scaler.scale(total_loss / group_size).backward()
        group_complete = (
            (batch_index + 1) % gradient_accumulation_steps == 0
            or batch_index + 1 == loader_batches
        )
        if group_complete:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        for name, value in losses.items():
            totals[name] += float(value.detach().cpu())
        totals["loss_total"] += float(total_loss.detach().cpu())
        batches += 1
    if not batches:
        raise RuntimeError("training loader produced no batches")
    return {name: value / batches for name, value in sorted(totals.items())}


def _match_predictions(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int, list[float], list[float]]:
    keep = predicted_scores >= score_threshold
    predicted_boxes = predicted_boxes[keep]
    if predicted_boxes.numel() == 0:
        return 0, 0, int(target_boxes.shape[0]), [], []
    if target_boxes.numel() == 0:
        return 0, int(predicted_boxes.shape[0]), 0, [], []

    overlaps = box_iou(predicted_boxes, target_boxes)
    matched_targets: set[int] = set()
    matched_ious: list[float] = []
    matched_coordinate_mae: list[float] = []
    for prediction_index in range(predicted_boxes.shape[0]):
        order = torch.argsort(overlaps[prediction_index], descending=True)
        for raw_target_index in order:
            target_index = int(raw_target_index.item())
            if target_index in matched_targets:
                continue
            overlap = float(overlaps[prediction_index, target_index].item())
            if overlap < iou_threshold:
                break
            matched_targets.add(target_index)
            matched_ious.append(overlap)
            matched_coordinate_mae.append(
                float(
                    torch.abs(
                        predicted_boxes[prediction_index]
                        - target_boxes[target_index]
                    )
                    .mean()
                    .item()
                )
            )
            break
    true_positive = len(matched_targets)
    return (
        true_positive,
        int(predicted_boxes.shape[0]) - true_positive,
        int(target_boxes.shape[0]) - true_positive,
        matched_ious,
        matched_coordinate_mae,
    )


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _draw_overlay(
    sample: dict[str, Any],
    *,
    score_threshold: float,
) -> Image.Image:
    image = _tensor_to_pil(sample["image"])
    draw = ImageDraw.Draw(image)
    for box in sample["target_boxes"]:
        draw.rectangle(tuple(float(value) for value in box), outline="red", width=3)
    seed_box = tuple(float(value) for value in sample["seed_box"][0])
    draw.rectangle(seed_box, outline="blue", width=2)
    for box, score in zip(
        sample["predicted_boxes"],
        sample["predicted_scores"],
        strict=True,
    ):
        if float(score) < score_threshold:
            continue
        coordinates = tuple(float(value) for value in box)
        draw.rectangle(coordinates, outline="lime", width=3)
        draw.text(
            (coordinates[0] + 2, max(0.0, coordinates[1] - 12)),
            f"{float(score):.2f}",
            fill="lime",
        )
    title = (
        f"{sample['sample_id']}  seed y={sample['seed_label']} "
        f"p={sample['seed_probability']:.3f}  "
        "GT=red Pred=green Seed=blue"
    )
    draw.rectangle((0, 0, image.width, 18), fill="black")
    draw.text((4, 3), title, fill="white")
    return image


def _save_overlay_grid(
    samples: list[dict[str, Any]],
    path: Path,
    *,
    score_threshold: float,
    columns: int = 2,
) -> None:
    if not samples:
        return
    panels = [
        _draw_overlay(sample, score_threshold=score_threshold)
        for sample in samples
    ]
    panel_width = max(panel.width for panel in panels)
    panel_height = max(panel.height for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (panel_width * columns, panel_height * rows),
        color="white",
    )
    for index, panel in enumerate(panels):
        x = (index % columns) * panel_width
        y = (index // columns) * panel_height
        canvas.paste(panel, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def _evaluate(
    model: PeakMultiTaskRCNN,
    loader: DataLoader[dict[str, Any]],
    preprocessor: AttributePreprocessor,
    device: torch.device,
    *,
    score_threshold: float,
    iou_threshold: float,
    overlay_count: int,
    amp_enabled: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    detection_tp = detection_fp = detection_fn = 0
    matched_ious: list[float] = []
    matched_coordinate_mae: list[float] = []
    seed_probabilities: list[float] = []
    seed_labels: list[int] = []
    overlay_samples: list[dict[str, Any]] = []

    for raw_batch in loader:
        attributes = preprocessor.transform(
            raw_batch["attributes"],
            raw_batch["attribute_masks"],
        ).to(device)
        batch = _move_batch(raw_batch, device)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            output = model(
                batch["images"],
                seed_boxes=batch["seed_boxes"],
                attributes=attributes,
            )
        probabilities = output["seed_probabilities"].detach().cpu()
        labels = batch["seed_labels"].detach().cpu()
        seed_probabilities.extend(float(value) for value in probabilities)
        seed_labels.extend(int(value) for value in labels)
        for index, (detection, target) in enumerate(
            zip(output["detections"], batch["targets"], strict=True)
        ):
            predicted_boxes = detection["boxes"].detach().cpu()
            predicted_scores = detection["scores"].detach().cpu()
            target_boxes = target["boxes"].detach().cpu()
            tp, fp, fn, ious, coordinate_errors = _match_predictions(
                predicted_boxes,
                predicted_scores,
                target_boxes,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
            )
            detection_tp += tp
            detection_fp += fp
            detection_fn += fn
            matched_ious.extend(ious)
            matched_coordinate_mae.extend(coordinate_errors)
            if len(overlay_samples) < overlay_count:
                overlay_samples.append(
                    {
                        "image": batch["images"][index].detach().cpu(),
                        "target_boxes": target_boxes,
                        "seed_box": batch["seed_boxes"][index].detach().cpu(),
                        "predicted_boxes": predicted_boxes,
                        "predicted_scores": predicted_scores,
                        "seed_probability": float(probabilities[index]),
                        "seed_label": int(labels[index]),
                        "sample_id": str(
                            batch["metadata"][index]["sample_id"]
                        ),
                    }
                )

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
    seed_sensitivity = _safe_divide(seed_tp, seed_tp + seed_fn)
    seed_specificity = _safe_divide(seed_tn, seed_tn + seed_fp)
    clamped_probabilities = torch.tensor(seed_probabilities).clamp(1e-7, 1 - 1e-7)
    label_tensor = torch.tensor(seed_labels, dtype=torch.float32)
    seed_bce = float(
        torch.nn.functional.binary_cross_entropy(
            clamped_probabilities,
            label_tensor,
        ).item()
    )
    seed_auroc, seed_average_precision = _binary_ranking_metrics(
        seed_probabilities,
        seed_labels,
    )
    seed_brier = float(
        torch.mean((clamped_probabilities - label_tensor) ** 2).item()
    )
    seed_ece = _expected_calibration_error(seed_probabilities, seed_labels)
    detection_precision = _safe_divide(
        detection_tp,
        detection_tp + detection_fp,
    )
    detection_recall = _safe_divide(
        detection_tp,
        detection_tp + detection_fn,
    )
    metrics = {
        "samples": len(seed_labels),
        "detection": {
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "true_positive": detection_tp,
            "false_positive": detection_fp,
            "false_negative": detection_fn,
            "precision": detection_precision,
            "recall": detection_recall,
            "f1": _safe_divide(
                2 * detection_tp,
                2 * detection_tp + detection_fp + detection_fn,
            ),
            "matched_mean_iou": (
                sum(matched_ious) / len(matched_ious)
                if matched_ious
                else 0.0
            ),
            "matched_median_iou": (
                float(torch.tensor(matched_ious).median().item())
                if matched_ious
                else 0.0
            ),
            "matched_coordinate_mae_px": (
                sum(matched_coordinate_mae) / len(matched_coordinate_mae)
                if matched_coordinate_mae
                else 0.0
            ),
            "predictions_per_image": _safe_divide(
                detection_tp + detection_fp,
                len(seed_labels),
            ),
        },
        "seed": {
            "true_positive": seed_tp,
            "false_positive": seed_fp,
            "false_negative": seed_fn,
            "true_negative": seed_tn,
            "accuracy": _safe_divide(seed_tp + seed_tn, len(seed_labels)),
            "precision": _safe_divide(seed_tp, seed_tp + seed_fp),
            "recall": seed_sensitivity,
            "f1": _safe_divide(2 * seed_tp, 2 * seed_tp + seed_fp + seed_fn),
            "specificity": seed_specificity,
            "balanced_accuracy": (seed_sensitivity + seed_specificity) / 2.0,
            "auroc": seed_auroc,
            "average_precision": seed_average_precision,
            "bce": seed_bce,
            "brier": seed_brier,
            "ece_10_bin": seed_ece,
        },
    }
    return metrics, overlay_samples


def _save_checkpoint(
    path: Path,
    *,
    model: PeakMultiTaskRCNN,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
    mode: str,
    fusion_mode: str,
    training_task: str,
    scaler: torch.amp.GradScaler | None = None,
    data_loader_generator: torch.Generator | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
            "mode": mode,
            "fusion_mode": fusion_mode,
            "attribute_fusion_enabled": fusion_mode != "image_only",
            "training_task": training_task,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "python_random_state": random.getstate(),
            "data_loader_generator_state": (
                data_loader_generator.get_state()
                if data_loader_generator is not None
                else None
            ),
        },
        path,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_with_process_lock(args: argparse.Namespace) -> dict[str, Any]:
    if fcntl is None:
        return run(args)

    lock_path = PROJECT_ROOT / ".peak_fusion_training.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.seek(0)
            holder = lock_file.read().strip() or "unknown process"
            raise RuntimeError(
                "another peak fusion training process is still active; "
                f"lock holder: {holder}"
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "output_dir": str(args.output_dir),
                    "started_at_unix": time.time(),
                },
                ensure_ascii=False,
            )
        )
        lock_file.flush()
        return run(args)
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def _build_model_config(
    args: argparse.Namespace,
    train_records_full: list[PeakManifestRecord],
) -> dict[str, Any]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    model_cfg = config["model"]
    attribute_dimensions = {len(record.attributes) for record in train_records_full}
    if len(attribute_dimensions) != 1:
        raise ValueError(
            f"training manifest has inconsistent attribute dimensions: {attribute_dimensions}"
        )
    model_cfg["attr_dim"] = attribute_dimensions.pop()
    model_cfg["fusion_mode"] = args.fusion_mode
    model_cfg["image_min_size"] = args.image_size
    model_cfg["image_max_size"] = args.image_size
    model_cfg["rpn_pre_nms_top_n_train"] = args.rpn_train_proposals
    model_cfg["rpn_post_nms_top_n_train"] = args.rpn_train_proposals
    model_cfg["rpn_pre_nms_top_n_test"] = args.rpn_test_proposals
    model_cfg["rpn_post_nms_top_n_test"] = args.rpn_test_proposals
    model_cfg["box_nms_thresh"] = args.box_nms_thresh
    positives = sum(record.seed_label == 1 for record in train_records_full)
    negatives = len(train_records_full) - positives
    if positives == 0:
        raise ValueError("training manifest contains no Seed-positive samples")
    config["loss"]["seed_pos_weight"] = negatives / positives
    return config


def _overfit_passed(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    detection = metrics["detection"]
    seed = metrics["seed"]
    return (
        seed["accuracy"] >= args.overfit_seed_accuracy
        and detection["precision"] >= args.overfit_detection_precision
        and detection["recall"] >= args.overfit_detection_recall
        and detection["matched_mean_iou"] >= args.overfit_mean_iou
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-compatible accelerator unavailable. Install the matching "
            "ROCm or NVIDIA CUDA PyTorch build; refusing to silently train on CPU."
        )
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    train_records_full = load_manifest_jsonl(args.train_manifest)
    if args.mode == "overfit":
        train_records = _balanced_smoke_subset(
            train_records_full,
            args.train_limit,
            seed=args.seed,
        )
        validation_records = train_records
    else:
        if args.val_manifest is None:
            raise ValueError("--val-manifest is required in pilot mode")
        train_records = _balanced_smoke_subset(
            train_records_full,
            args.train_limit,
            seed=args.seed,
        )
        validation_records = _balanced_smoke_subset(
            load_manifest_jsonl(args.val_manifest),
            args.val_limit,
            seed=args.seed + 1,
        )

    preprocessor = AttributePreprocessor.fit(train_records_full)
    train_dataset = PeakMultiTaskDataset(
        train_records,
        image_root=args.image_root,
    )
    validation_dataset = PeakMultiTaskDataset(
        validation_records,
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
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_peak_multitask_batch,
        pin_memory=device.type == "cuda",
    )

    config = _build_model_config(args, train_records_full)
    model = PeakMultiTaskRCNN.from_config(
        config,
        pretrained=args.pretrained,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "summary.json").exists() and args.resume:
        completed = json.loads(
            (args.output_dir / "summary.json").read_text(encoding="utf-8")
        )
        if int(completed.get("epochs_completed", 0)) >= args.epochs:
            print(json.dumps(completed, ensure_ascii=False, indent=2), flush=True)
            return completed
    if (args.output_dir / "summary.json").exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"{args.output_dir / 'summary.json'} exists; pass --overwrite "
            "or use a new output directory"
        )
    existing_files = list(args.output_dir.iterdir())
    if existing_files and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"{args.output_dir} is not empty; pass --resume, --overwrite, or use a new directory"
        )
    preprocessor_path = args.output_dir / "attribute_preprocessing.json"
    if args.resume and preprocessor_path.exists():
        saved_preprocessor = AttributePreprocessor.load_json(preprocessor_path)
        if saved_preprocessor.to_mapping() != preprocessor.to_mapping():
            raise ValueError("resume refused: train-fitted attribute preprocessing changed")
    else:
        preprocessor.save_json(preprocessor_path)
    run_config = {
        "mode": args.mode,
        "fusion_mode": args.fusion_mode,
        "attribute_fusion_enabled": args.fusion_mode != "image_only",
        "training_task": args.training_task,
        "data_augmentation_enabled": False,
        "test_manifest_used": False,
        "pretrained_backbone": bool(args.pretrained),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "train": _dataset_summary(train_records),
        "validation": _dataset_summary(validation_records),
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": _sha256(args.train_manifest),
        "val_manifest": str(args.val_manifest.resolve()) if args.val_manifest else None,
        "val_manifest_sha256": _sha256(args.val_manifest) if args.val_manifest else None,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "automatic_mixed_precision": amp_enabled,
        "accelerator_backend": _accelerator_backend(device),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": torch.version.hip,
        "max_epochs": args.epochs,
        "attribute_dimension": int(config["model"]["attr_dim"]),
        "resume_enabled": bool(args.resume),
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_detection_score_threshold": args.score_threshold,
        "validation_detection_iou_threshold": args.iou_threshold,
        "rpn_train_proposals": args.rpn_train_proposals,
        "rpn_test_proposals": args.rpn_test_proposals,
        "box_nms_threshold": args.box_nms_thresh,
        "minimum_overfit_epochs": args.min_overfit_epochs,
        "overfit_pass_criteria": {
            "required_consecutive_epochs": args.overfit_pass_streak,
            "seed_accuracy": args.overfit_seed_accuracy,
            "detection_precision": args.overfit_detection_precision,
            "detection_recall": args.overfit_detection_recall,
            "matched_mean_iou": args.overfit_mean_iou,
        },
    }
    run_config_path = args.output_dir / "run_config.json"
    if args.resume and run_config_path.exists():
        previous_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        immutable_keys = (
            "fusion_mode",
            "training_task",
            "image_size",
            "batch_size",
            "gradient_accumulation_steps",
            "automatic_mixed_precision",
            "attribute_dimension",
            "train_manifest_sha256",
            "val_manifest_sha256",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "validation_detection_score_threshold",
            "validation_detection_iou_threshold",
            "rpn_train_proposals",
            "rpn_test_proposals",
            "box_nms_threshold",
        )
        changed = {
            key: (previous_config.get(key), run_config.get(key))
            for key in immutable_keys
            if previous_config.get(key) != run_config.get(key)
        }
        if changed:
            raise ValueError(f"resume refused: immutable configuration changed: {changed}")
    _write_json(run_config_path, run_config)

    history: list[dict[str, Any]] = []
    best_detection_score = -1.0
    best_seed_score = -1.0
    best_detection_epoch: int | None = None
    best_seed_epoch: int | None = None
    pass_streak = 0
    passed_epoch: int | None = None
    started = time.perf_counter()
    start_epoch = 1

    if args.resume:
        checkpoint_path = args.output_dir / "last.pt"
        history_path = args.output_dir / "history.json"
        if not checkpoint_path.exists() or not history_path.exists():
            raise FileNotFoundError(
                "--resume requires both last.pt and history.json in output-dir"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("fusion_mode") != args.fusion_mode:
            raise ValueError("resume refused: fusion mode differs from last checkpoint")
        if checkpoint.get("training_task", "joint") != args.training_task:
            raise ValueError("resume refused: training task differs from last checkpoint")
        if checkpoint.get("config") != config:
            raise ValueError("resume refused: model configuration differs from last checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if checkpoint.get("python_random_state") is not None:
            random.setstate(checkpoint["python_random_state"])
        if checkpoint.get("data_loader_generator_state") is not None:
            generator.set_state(checkpoint["data_loader_generator_state"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        checkpoint_epoch = int(checkpoint["epoch"])
        if not history or int(history[-1]["epoch"]) != checkpoint_epoch:
            raise ValueError("resume refused: history and last checkpoint epoch differ")
        start_epoch = checkpoint_epoch + 1
        best_detection_score = max(
            float(item["validation"]["detection"]["f1"]) for item in history
        )
        best_seed_score = max(
            float(item["validation"]["seed"]["balanced_accuracy"])
            for item in history
        )
        best_detection_epoch = next(
            int(item["epoch"])
            for item in history
            if float(item["validation"]["detection"]["f1"]) == best_detection_score
        )
        best_seed_epoch = next(
            int(item["epoch"])
            for item in history
            if float(item["validation"]["seed"]["balanced_accuracy"])
            == best_seed_score
        )
        print(f"resuming from epoch {checkpoint_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_losses = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            preprocessor,
            device,
            amp_enabled=amp_enabled,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scaler=scaler,
            training_task=args.training_task,
        )
        metrics, overlay_samples = _evaluate(
            model,
            validation_loader,
            preprocessor,
            device,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
            overlay_count=args.overlay_count,
            amp_enabled=amp_enabled,
        )
        _save_overlay_grid(
            overlay_samples,
            args.output_dir / "overlays" / f"epoch_{epoch:03d}.png",
            score_threshold=args.score_threshold,
        )

        epoch_result = {
            "epoch": epoch,
            "train_losses": train_losses,
            "validation": metrics,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_result)
        detection_score = float(metrics["detection"]["f1"])
        seed_score = float(metrics["seed"]["balanced_accuracy"])
        if detection_score > best_detection_score:
            best_detection_score = detection_score
            best_detection_epoch = epoch
            _save_checkpoint(
                args.output_dir / "best_detection.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                metrics=metrics,
                mode=args.mode,
                fusion_mode=args.fusion_mode,
                training_task=args.training_task,
                scaler=scaler,
                data_loader_generator=generator,
            )
        if seed_score > best_seed_score:
            best_seed_score = seed_score
            best_seed_epoch = epoch
            _save_checkpoint(
                args.output_dir / "best_seed.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                metrics=metrics,
                mode=args.mode,
                fusion_mode=args.fusion_mode,
                training_task=args.training_task,
                scaler=scaler,
                data_loader_generator=generator,
            )
        _save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            metrics=metrics,
            mode=args.mode,
            fusion_mode=args.fusion_mode,
            training_task=args.training_task,
            scaler=scaler,
            data_loader_generator=generator,
        )
        _write_json(args.output_dir / "history.json", history)
        print(json.dumps(epoch_result, ensure_ascii=False, allow_nan=False), flush=True)

        if args.mode == "overfit" and epoch >= args.min_overfit_epochs:
            if _overfit_passed(metrics, args):
                pass_streak += 1
            else:
                pass_streak = 0
            if pass_streak >= args.overfit_pass_streak:
                passed_epoch = epoch
                break

    summary = {
        "status": "ok",
        "mode": args.mode,
        "fusion_mode": args.fusion_mode,
        "attribute_fusion_enabled": args.fusion_mode != "image_only",
        "training_task": args.training_task,
        "data_augmentation_enabled": False,
        "test_manifest_used": False,
        "pretrained_backbone": bool(args.pretrained),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "device": {
            "type": str(device),
            "backend": _accelerator_backend(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "name": (
                torch.cuda.get_device_name(0)
                if device.type == "cuda"
                else "CPU"
            ),
            "max_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else 0
            ),
            "max_memory_reserved_bytes": (
                torch.cuda.max_memory_reserved()
                if device.type == "cuda"
                else 0
            ),
        },
        "data": {
            "train_manifest": str(args.train_manifest.resolve()),
            "val_manifest": (
                str(args.val_manifest.resolve())
                if args.val_manifest is not None
                else None
            ),
            "train": _dataset_summary(train_records),
            "validation": _dataset_summary(validation_records),
        },
        "epochs_completed": int(history[-1]["epoch"]),
        "elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in history),
        "best_detection_epoch": best_detection_epoch,
        "best_detection_f1": best_detection_score,
        "best_seed_epoch": best_seed_epoch,
        "best_seed_balanced_accuracy": best_seed_score,
        "overfit_check": {
            "required": args.mode == "overfit",
            "passed": passed_epoch is not None,
            "passed_epoch": passed_epoch,
            "final_pass_streak": pass_streak,
            "criteria": run_config["overfit_pass_criteria"],
        },
        "final": history[-1],
        "outputs": {
            "best_detection": str(
                (args.output_dir / "best_detection.pt").resolve()
            ),
            "best_seed": str((args.output_dir / "best_seed.pt").resolve()),
            "last": str((args.output_dir / "last.pt").resolve()),
            "history": str((args.output_dir / "history.json").resolve()),
            "overlays": str((args.output_dir / "overlays").resolve()),
        },
    }
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "PeakMultiTaskRCNN fusion and ablation training"
    )
    parser.add_argument("--mode", choices=["overfit", "pilot"], required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "PeakTruthLab" / "configs" / "peak_multitask.yaml",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--fusion-mode",
        choices=["image_only", "attr_only", "naive_concat", "gated_fusion"],
        required=True,
    )
    parser.add_argument(
        "--training-task",
        choices=["joint", "detection", "seed"],
        default="joint",
    )
    parser.add_argument("--train-limit", type=int, required=True)
    parser.add_argument("--val-limit", type=int, default=500)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--min-overfit-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="optimizer update interval; effective batch is batch-size times this value",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="use float16 autocast and gradient scaling on CUDA/ROCm",
    )
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--rpn-train-proposals", type=int, default=256)
    parser.add_argument("--rpn-test-proposals", type=int, default=128)
    parser.add_argument("--box-nms-thresh", type=float, default=0.6)
    parser.add_argument("--overlay-count", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume exactly from output-dir/last.pt and history.json",
    )
    parser.add_argument("--overfit-pass-streak", type=int, default=2)
    parser.add_argument("--overfit-seed-accuracy", type=float, default=0.95)
    parser.add_argument("--overfit-detection-precision", type=float, default=0.80)
    parser.add_argument("--overfit-detection-recall", type=float, default=0.80)
    parser.add_argument("--overfit-mean-iou", type=float, default=0.65)
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.mode == "pilot" and args.val_manifest is None:
        parser.error("--val-manifest is required when --mode pilot")
    if args.image_size != 480:
        raise ValueError("this staged protocol requires --image-size 480")
    if (
        args.epochs <= 0
        or args.train_limit <= 0
        or args.batch_size <= 0
        or args.gradient_accumulation_steps <= 0
    ):
        raise ValueError(
            "epochs, train-limit, batch-size, and "
            "gradient-accumulation-steps must be positive"
        )
    if args.batch_size > 4 and not args.amp:
        raise ValueError(
            "batch sizes above 4 require --amp on the supported GPU training path"
        )
    if args.batch_size >= 4 and (
        args.rpn_train_proposals > 128 or args.rpn_test_proposals > 64
    ):
        raise ValueError(
            "batch sizes of 4 or more require RPN proposal limits at or below "
            "128 train / 64 test for this staged protocol"
        )
    if args.mode == "overfit" and not 20 <= args.epochs <= 30:
        raise ValueError("overfit mode requires 20 to 30 epochs")
    return args


if __name__ == "__main__":
    _run_with_process_lock(parse_args())
