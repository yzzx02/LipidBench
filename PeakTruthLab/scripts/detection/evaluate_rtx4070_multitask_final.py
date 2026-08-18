from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou


PROJECT_ROOT = Path(
    os.environ.get("LIPIDBENCH_PROJECT_ROOT", r"D:\CODE\LipidBench")
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.data import (  # noqa: E402
    AttributePreprocessor,
    PeakMultiTaskDataset,
    collate_peak_multitask_batch,
    load_manifest_jsonl,
)
from lipidbench.models import PeakMultiTaskRCNN  # noqa: E402


EXPERIMENTS = (
    ("A_image_only", "image_only"),
    ("B_attr_only", "attr_only"),
    ("C_naive_concat", "naive_concat"),
    ("D_gated_fusion", "gated_fusion"),
)
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tensor_boxes(value: torch.Tensor) -> list[list[float]]:
    return [[float(x) for x in row] for row in value.detach().cpu().tolist()]


def sample_class(true_peak_count: int) -> str:
    if true_peak_count == 0:
        return "empty"
    if true_peak_count == 1:
        return "single"
    return "multi"


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PeakMultiTaskRCNN, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    info = {
        "checkpoint": str(checkpoint_path.resolve()),
        "epoch": int(checkpoint["epoch"]),
        "fusion_mode": str(checkpoint["fusion_mode"]),
        "training_task": str(checkpoint.get("training_task", "joint")),
        "validation_metrics_at_save": checkpoint["metrics"],
    }
    state_dict = checkpoint["model_state_dict"]
    model = PeakMultiTaskRCNN.from_config(config, pretrained=False)
    model.load_state_dict(state_dict)
    del checkpoint, state_dict
    model.to(device)
    model.eval()
    return model, info


@torch.inference_mode()
def infer_manifest(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    image_root: Path,
    preprocessor_path: Path,
    device: torch.device,
    batch_size: int,
    amp: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = load_manifest_jsonl(manifest_path)
    dataset = PeakMultiTaskDataset(records, image_root=image_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_peak_multitask_batch,
    )
    preprocessor = AttributePreprocessor.load_json(preprocessor_path)
    model, checkpoint_info = load_model(checkpoint_path, device)
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for raw_batch in loader:
        attributes = preprocessor.transform(
            raw_batch["attributes"],
            raw_batch["attribute_masks"],
        ).to(device)
        images = [image.to(device) for image in raw_batch["images"]]
        seed_boxes = [box.to(device) for box in raw_batch["seed_boxes"]]
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(
                images,
                seed_boxes=seed_boxes,
                attributes=attributes,
            )
        probabilities = output["seed_probabilities"].detach().cpu().tolist()
        for index, detection in enumerate(output["detections"]):
            labels = detection["labels"].detach().cpu()
            keep = labels == 1
            metadata = raw_batch["metadata"][index]
            target_boxes = tensor_boxes(raw_batch["targets"][index]["boxes"])
            samples.append(
                {
                    "sample_id": str(metadata["sample_id"]),
                    "source_file": str(metadata["source_file"]),
                    "subsets": list(metadata.get("subsets") or ()),
                    "true_peak_count_class": sample_class(len(target_boxes)),
                    "target_boxes": target_boxes,
                    "seed_label": int(raw_batch["seed_labels"][index].item()),
                    "seed_probability": float(probabilities[index]),
                    "predicted_boxes": tensor_boxes(detection["boxes"][keep]),
                    "predicted_scores": [
                        float(value)
                        for value in detection["scores"][keep]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                }
            )
    checkpoint_info["inference_samples"] = len(samples)
    checkpoint_info["inference_seconds"] = time.perf_counter() - started
    del model, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return samples, checkpoint_info


def match_one_sample(
    sample: dict[str, Any],
    *,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_boxes = torch.tensor(sample["target_boxes"], dtype=torch.float32)
    predicted_boxes = torch.tensor(
        sample["predicted_boxes"], dtype=torch.float32
    ).reshape(-1, 4)
    predicted_scores = sample["predicted_scores"]
    kept_indices = [
        index
        for index, score in enumerate(predicted_scores)
        if score >= score_threshold
    ]
    kept_indices.sort(key=lambda index: predicted_scores[index], reverse=True)
    matched_targets: set[int] = set()
    rows: list[dict[str, Any]] = []
    matched_ious: list[float] = []
    left_errors: list[float] = []
    right_errors: list[float] = []
    width_errors: list[float] = []
    center_errors: list[float] = []
    coordinate_maes: list[float] = []
    for prediction_index in kept_indices:
        box = predicted_boxes[prediction_index]
        matched_target: int | None = None
        matched_iou = 0.0
        if target_boxes.numel():
            overlaps = box_iou(box.reshape(1, 4), target_boxes)[0]
            for raw_target_index in torch.argsort(overlaps, descending=True):
                target_index = int(raw_target_index.item())
                if target_index in matched_targets:
                    continue
                overlap = float(overlaps[target_index].item())
                if overlap >= iou_threshold:
                    matched_target = target_index
                    matched_iou = overlap
                break
        if matched_target is not None:
            matched_targets.add(matched_target)
            target = target_boxes[matched_target]
            matched_ious.append(matched_iou)
            left_errors.append(float(abs(box[0] - target[0]).item()))
            right_errors.append(float(abs(box[2] - target[2]).item()))
            width_errors.append(
                float(abs((box[2] - box[0]) - (target[2] - target[0])).item())
            )
            center_errors.append(
                float(
                    abs(
                        (box[0] + box[2]) / 2.0
                        - (target[0] + target[2]) / 2.0
                    ).item()
                )
            )
            coordinate_maes.append(float(torch.mean(abs(box - target)).item()))
        raw_box = sample["predicted_boxes"][prediction_index]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "source_file": sample["source_file"],
                "true_peak_count_class": sample["true_peak_count_class"],
                "prediction_index": prediction_index,
                "score": predicted_scores[prediction_index],
                "pred_x1": raw_box[0],
                "pred_y1": raw_box[1],
                "pred_x2": raw_box[2],
                "pred_y2": raw_box[3],
                "matched": matched_target is not None,
                "matched_target_index": (
                    matched_target if matched_target is not None else ""
                ),
                "matched_iou": matched_iou,
                "target_box_json": (
                    json.dumps(sample["target_boxes"][matched_target])
                    if matched_target is not None
                    else ""
                ),
            }
        )
    true_positive = len(matched_targets)
    stats = {
        "true_positive": true_positive,
        "false_positive": len(kept_indices) - true_positive,
        "false_negative": len(sample["target_boxes"]) - true_positive,
        "target_count": len(sample["target_boxes"]),
        "prediction_count": len(kept_indices),
        "exact_count": len(sample["target_boxes"]) == len(kept_indices),
        "count_absolute_error": abs(
            len(sample["target_boxes"]) - len(kept_indices)
        ),
        "matched_ious": matched_ious,
        "left_errors": left_errors,
        "right_errors": right_errors,
        "width_errors": width_errors,
        "center_errors": center_errors,
        "coordinate_maes": coordinate_maes,
    }
    return rows, stats


def mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def detection_threshold_metrics(
    samples: list[dict[str, Any]],
    *,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    tp = fp = fn = 0
    matched_ious: list[float] = []
    left_errors: list[float] = []
    right_errors: list[float] = []
    width_errors: list[float] = []
    center_errors: list[float] = []
    coordinate_maes: list[float] = []
    for sample in samples:
        rows, stats = match_one_sample(
            sample,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )
        prediction_rows.extend(rows)
        tp += stats["true_positive"]
        fp += stats["false_positive"]
        fn += stats["false_negative"]
        matched_ious.extend(stats["matched_ious"])
        left_errors.extend(stats["left_errors"])
        right_errors.extend(stats["right_errors"])
        width_errors.extend(stats["width_errors"])
        center_errors.extend(stats["center_errors"])
        coordinate_maes.extend(stats["coordinate_maes"])
        sample_rows.append(
            {
                "sample_id": sample["sample_id"],
                "source_file": sample["source_file"],
                "true_peak_count_class": sample["true_peak_count_class"],
                **{
                    key: stats[key]
                    for key in (
                        "true_positive",
                        "false_positive",
                        "false_negative",
                        "target_count",
                        "prediction_count",
                        "exact_count",
                        "count_absolute_error",
                    )
                },
                "mean_matched_iou": mean_or_zero(stats["matched_ious"]),
            }
        )
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    metrics = {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "matched_mean_iou": mean_or_zero(matched_ious),
        "matched_median_iou": median_or_zero(matched_ious),
        "left_boundary_mae_px": mean_or_zero(left_errors),
        "right_boundary_mae_px": mean_or_zero(right_errors),
        "width_mae_px": mean_or_zero(width_errors),
        "center_mae_px": mean_or_zero(center_errors),
        "box_coordinate_mae_px": mean_or_zero(coordinate_maes),
        "exact_peak_count_accuracy": mean_or_zero(
            [float(row["exact_count"]) for row in sample_rows]
        ),
        "peak_count_mae": mean_or_zero(
            [float(row["count_absolute_error"]) for row in sample_rows]
        ),
        "predictions_per_image": safe_divide(tp + fp, len(samples)),
    }
    return metrics, prediction_rows, sample_rows


def detection_threshold_sweep(
    samples: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for index in range(1, 20):
        threshold = round(index * 0.05, 2)
        metrics, _, _ = detection_threshold_metrics(
            samples,
            score_threshold=threshold,
            iou_threshold=0.5,
        )
        rows.append(metrics)
    best = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["recall"],
            row["precision"],
            -abs(row["score_threshold"] - 0.5),
        ),
    )
    return float(best["score_threshold"]), rows


def detection_ap_curve(
    samples: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[float, list[dict[str, Any]]]:
    ranked_predictions: list[tuple[float, int]] = []
    total_targets = sum(len(sample["target_boxes"]) for sample in samples)
    for sample in samples:
        target_boxes = torch.tensor(sample["target_boxes"], dtype=torch.float32)
        predicted_boxes = torch.tensor(
            sample["predicted_boxes"], dtype=torch.float32
        ).reshape(-1, 4)
        order = sorted(
            range(len(sample["predicted_scores"])),
            key=lambda index: sample["predicted_scores"][index],
            reverse=True,
        )
        matched_targets: set[int] = set()
        for prediction_index in order:
            is_true_positive = 0
            if target_boxes.numel():
                overlaps = box_iou(
                    predicted_boxes[prediction_index].reshape(1, 4),
                    target_boxes,
                )[0]
                for raw_target_index in torch.argsort(overlaps, descending=True):
                    target_index = int(raw_target_index.item())
                    if target_index in matched_targets:
                        continue
                    if float(overlaps[target_index].item()) >= iou_threshold:
                        matched_targets.add(target_index)
                        is_true_positive = 1
                    break
            ranked_predictions.append(
                (sample["predicted_scores"][prediction_index], is_true_positive)
            )
    ranked_predictions.sort(key=lambda value: value[0], reverse=True)
    rows: list[dict[str, Any]] = []
    true_positive = false_positive = 0
    for rank, (score, label) in enumerate(ranked_predictions, start=1):
        true_positive += label
        false_positive += 1 - label
        rows.append(
            {
                "iou_threshold": iou_threshold,
                "rank": rank,
                "score": score,
                "precision": safe_divide(
                    true_positive,
                    true_positive + false_positive,
                ),
                "recall": safe_divide(true_positive, total_targets),
            }
        )
    interpolated: list[float] = []
    for recall_index in range(101):
        recall_level = recall_index / 100.0
        interpolated.append(
            max(
                (
                    row["precision"]
                    for row in rows
                    if row["recall"] >= recall_level
                ),
                default=0.0,
            )
        )
    return mean_or_zero(interpolated), rows


def detection_ap_metrics(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for threshold in IOU_THRESHOLDS:
        average_precision, rows = detection_ap_curve(
            samples,
            iou_threshold=threshold,
        )
        summary_rows.append(
            {"iou_threshold": threshold, "average_precision": average_precision}
        )
        curve_rows.extend(rows)
    by_threshold = {
        row["iou_threshold"]: row["average_precision"] for row in summary_rows
    }
    metrics = {
        "ap50": by_threshold[0.5],
        "ap75": by_threshold[0.75],
        "map_50_95": mean_or_zero(list(by_threshold.values())),
    }
    return metrics, summary_rows, curve_rows


def binary_ranking_curves(
    probabilities: list[float],
    labels: list[int],
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    ranked = sorted(
        zip(probabilities, labels, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(labels)
    negatives = len(labels) - positives
    tp = fp = 0
    previous_tpr = previous_fpr = 0.0
    auroc = average_precision = 0.0
    roc_rows = [
        {
            "threshold": 1.0,
            "false_positive_rate": 0.0,
            "true_positive_rate": 0.0,
        }
    ]
    pr_rows = [
        {
            "threshold": 1.0,
            "recall": 0.0,
            "precision": 1.0,
        }
    ]
    index = 0
    while index < len(ranked):
        threshold = ranked[index][0]
        group_positive = group_negative = 0
        while index < len(ranked) and ranked[index][0] == threshold:
            if ranked[index][1] == 1:
                group_positive += 1
            else:
                group_negative += 1
            index += 1
        tp += group_positive
        fp += group_negative
        tpr = safe_divide(tp, positives)
        fpr = safe_divide(fp, negatives)
        precision = safe_divide(tp, tp + fp)
        auroc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        if group_positive:
            average_precision += precision * (tpr - previous_tpr)
        roc_rows.append(
            {
                "threshold": threshold,
                "false_positive_rate": fpr,
                "true_positive_rate": tpr,
            }
        )
        pr_rows.append(
            {
                "threshold": threshold,
                "recall": tpr,
                "precision": precision,
            }
        )
        previous_tpr = tpr
        previous_fpr = fpr
    return (
        {"auroc": float(auroc), "average_precision": float(average_precision)},
        roc_rows,
        pr_rows,
    )


def seed_threshold_metrics(
    samples: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    probabilities = [sample["seed_probability"] for sample in samples]
    labels = [sample["seed_label"] for sample in samples]
    predictions = [int(value >= threshold) for value in probabilities]
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels, strict=True))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels, strict=True))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels, strict=True))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels, strict=True))
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    clamped = [min(max(value, 1e-7), 1 - 1e-7) for value in probabilities]
    bce = -mean_or_zero(
        [
            label * math.log(probability)
            + (1 - label) * math.log(1 - probability)
            for probability, label in zip(clamped, labels, strict=True)
        ]
    )
    brier = mean_or_zero(
        [
            (probability - label) ** 2
            for probability, label in zip(probabilities, labels, strict=True)
        ]
    )
    ranking, _, _ = binary_ranking_curves(probabilities, labels)
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "accuracy": safe_divide(tp + tn, len(labels)),
        "precision": safe_divide(tp, tp + fp),
        "recall": sensitivity,
        "specificity": specificity,
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "auroc": ranking["auroc"],
        "average_precision": ranking["average_precision"],
        "bce": bce,
        "brier": brier,
    }


def seed_threshold_sweep(
    samples: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    rows = [
        seed_threshold_metrics(samples, threshold=index / 1000.0)
        for index in range(1001)
    ]
    best = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            row["f1"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return float(best["threshold"]), rows


def seed_calibration_rows(
    samples: list[dict[str, Any]],
    *,
    bins: int = 10,
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    total = len(samples)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            sample
            for sample in samples
            if sample["seed_probability"] >= lower
            and (
                sample["seed_probability"] < upper
                or (index == bins - 1 and sample["seed_probability"] <= upper)
            )
        ]
        confidence = mean_or_zero(
            [sample["seed_probability"] for sample in selected]
        )
        observed = mean_or_zero(
            [float(sample["seed_label"]) for sample in selected]
        )
        contribution = len(selected) / total * abs(observed - confidence)
        ece += contribution
        rows.append(
            {
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "n": len(selected),
                "mean_probability": confidence,
                "observed_positive_fraction": observed,
                "ece_contribution": contribution,
            }
        )
    return rows, float(ece)


def detection_subgroup_rows(
    samples: list[dict[str, Any]],
    *,
    score_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[("peak_count_class", sample["true_peak_count_class"])].append(sample)
        groups[("source_file", sample["source_file"])].append(sample)
    for (group_type, group_value), selected in sorted(groups.items()):
        metrics, _, _ = detection_threshold_metrics(
            selected,
            score_threshold=score_threshold,
            iou_threshold=0.5,
        )
        rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "samples": len(selected),
                **metrics,
            }
        )
    return rows


def seed_subgroup_rows(
    samples: list[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[("peak_count_class", sample["true_peak_count_class"])].append(sample)
        groups[("source_file", sample["source_file"])].append(sample)
    for (group_type, group_value), selected in sorted(groups.items()):
        metrics = seed_threshold_metrics(selected, threshold=threshold)
        rows.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "samples": len(selected),
                **metrics,
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    result_root = args.result_root.resolve()
    dataset_root = result_root / "dataset_release" / "PeakTruthLab-dataset-v2"
    experiments_root = result_root / "experiments"
    output_root = result_root / "test_evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    provenance_code = result_root / "00_provenance" / "code"
    provenance_code.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), provenance_code)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for final evaluation")
    torch.manual_seed(20260725)
    aggregate_rows: list[dict[str, Any]] = []
    for experiment_id, expected_mode in EXPERIMENTS:
        print(f"[START] {experiment_id}", flush=True)
        experiment_dir = experiments_root / experiment_id
        summary = json.loads(
            (experiment_dir / "summary.json").read_text(encoding="utf-8")
        )
        if summary["fusion_mode"] != expected_mode:
            raise ValueError(f"{experiment_id}: unexpected fusion mode")
        evaluation_dir = output_root / experiment_id
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        preprocessor_path = experiment_dir / "attribute_preprocessing.json"

        val_detection, val_detection_checkpoint = infer_manifest(
            checkpoint_path=experiment_dir / "best_detection.pt",
            manifest_path=dataset_root / "manifests" / "val.jsonl",
            image_root=dataset_root / "eic_images_flat",
            preprocessor_path=preprocessor_path,
            device=device,
            batch_size=args.batch_size,
            amp=True,
        )
        detection_threshold, detection_sweep = detection_threshold_sweep(
            val_detection
        )
        write_csv(
            evaluation_dir / "val_detection_threshold_sweep.csv",
            detection_sweep,
            list(detection_sweep[0]),
        )
        del val_detection
        gc.collect()

        val_seed, val_seed_checkpoint = infer_manifest(
            checkpoint_path=experiment_dir / "best_seed.pt",
            manifest_path=dataset_root / "manifests" / "val.jsonl",
            image_root=dataset_root / "eic_images_flat",
            preprocessor_path=preprocessor_path,
            device=device,
            batch_size=args.batch_size,
            amp=True,
        )
        seed_threshold, seed_sweep = seed_threshold_sweep(val_seed)
        write_csv(
            evaluation_dir / "val_seed_threshold_sweep.csv",
            seed_sweep,
            list(seed_sweep[0]),
        )
        del val_seed
        gc.collect()

        selection = {
            "experiment_id": experiment_id,
            "fusion_mode": expected_mode,
            "test_data_inspected_during_selection": False,
            "detection": {
                "checkpoint": val_detection_checkpoint,
                "selection_metric": "Val detection F1 at IoU=0.50",
                "selected_score_threshold": detection_threshold,
            },
            "seed": {
                "checkpoint": val_seed_checkpoint,
                "selection_metric": "Val Seed balanced accuracy",
                "selected_probability_threshold": seed_threshold,
            },
        }
        write_json(evaluation_dir / "selection_before_test.json", selection)

        test_detection, test_detection_checkpoint = infer_manifest(
            checkpoint_path=experiment_dir / "best_detection.pt",
            manifest_path=dataset_root / "manifests" / "test.jsonl",
            image_root=dataset_root / "eic_images_flat",
            preprocessor_path=preprocessor_path,
            device=device,
            batch_size=args.batch_size,
            amp=True,
        )
        detection_metrics, detection_predictions, detection_sample_rows = (
            detection_threshold_metrics(
                test_detection,
                score_threshold=detection_threshold,
                iou_threshold=0.5,
            )
        )
        ap_metrics, ap_summary_rows, ap_curve_rows = detection_ap_metrics(
            test_detection
        )
        detection_metrics.update(ap_metrics)
        detection_subgroups = detection_subgroup_rows(
            test_detection,
            score_threshold=detection_threshold,
        )
        write_csv(
            evaluation_dir / "test_detection_predictions.csv",
            detection_predictions,
            list(detection_predictions[0]) if detection_predictions else [
                "sample_id",
                "source_file",
                "true_peak_count_class",
                "prediction_index",
                "score",
                "pred_x1",
                "pred_y1",
                "pred_x2",
                "pred_y2",
                "matched",
                "matched_target_index",
                "matched_iou",
                "target_box_json",
            ],
        )
        write_csv(
            evaluation_dir / "test_detection_image_metrics.csv",
            detection_sample_rows,
            list(detection_sample_rows[0]),
        )
        write_csv(
            evaluation_dir / "test_detection_ap_summary.csv",
            ap_summary_rows,
            list(ap_summary_rows[0]),
        )
        write_csv(
            evaluation_dir / "test_detection_pr_curves.csv",
            ap_curve_rows,
            list(ap_curve_rows[0]),
        )
        write_csv(
            evaluation_dir / "test_detection_subgroups.csv",
            detection_subgroups,
            list(detection_subgroups[0]),
        )
        del test_detection
        gc.collect()

        test_seed, test_seed_checkpoint = infer_manifest(
            checkpoint_path=experiment_dir / "best_seed.pt",
            manifest_path=dataset_root / "manifests" / "test.jsonl",
            image_root=dataset_root / "eic_images_flat",
            preprocessor_path=preprocessor_path,
            device=device,
            batch_size=args.batch_size,
            amp=True,
        )
        seed_metrics = seed_threshold_metrics(
            test_seed,
            threshold=seed_threshold,
        )
        ranking_metrics, roc_rows, pr_rows = binary_ranking_curves(
            [sample["seed_probability"] for sample in test_seed],
            [sample["seed_label"] for sample in test_seed],
        )
        seed_metrics.update(ranking_metrics)
        calibration_rows, ece = seed_calibration_rows(test_seed)
        seed_metrics["ece_10_bin"] = ece
        seed_subgroups = seed_subgroup_rows(test_seed, threshold=seed_threshold)
        seed_prediction_rows = [
            {
                "sample_id": sample["sample_id"],
                "source_file": sample["source_file"],
                "true_peak_count_class": sample["true_peak_count_class"],
                "seed_label": sample["seed_label"],
                "seed_probability": sample["seed_probability"],
                "seed_prediction": int(
                    sample["seed_probability"] >= seed_threshold
                ),
                "correct": int(
                    sample["seed_probability"] >= seed_threshold
                )
                == sample["seed_label"],
            }
            for sample in test_seed
        ]
        write_csv(
            evaluation_dir / "test_seed_predictions.csv",
            seed_prediction_rows,
            list(seed_prediction_rows[0]),
        )
        write_csv(
            evaluation_dir / "test_seed_roc_curve.csv",
            roc_rows,
            list(roc_rows[0]),
        )
        write_csv(
            evaluation_dir / "test_seed_pr_curve.csv",
            pr_rows,
            list(pr_rows[0]),
        )
        write_csv(
            evaluation_dir / "test_seed_calibration.csv",
            calibration_rows,
            list(calibration_rows[0]),
        )
        write_csv(
            evaluation_dir / "test_seed_subgroups.csv",
            seed_subgroups,
            list(seed_subgroups[0]),
        )
        del test_seed
        gc.collect()

        final_metrics = {
            "experiment_id": experiment_id,
            "fusion_mode": expected_mode,
            "selection": selection,
            "test_was_used_only_after_val_selection": True,
            "detection_checkpoint": test_detection_checkpoint,
            "seed_checkpoint": test_seed_checkpoint,
            "test_detection": detection_metrics,
            "test_seed": seed_metrics,
        }
        write_json(evaluation_dir / "test_metrics.json", final_metrics)
        aggregate_rows.append(
            {
                "experiment_id": experiment_id,
                "fusion_mode": expected_mode,
                "parameter_count": summary["parameter_count"],
                "train_seconds": summary["elapsed_seconds"],
                "peak_gpu_memory_bytes": summary["device"][
                    "max_memory_allocated_bytes"
                ],
                "best_detection_epoch": summary["best_detection_epoch"],
                "val_best_detection_f1_at_0_5": summary["best_detection_f1"],
                "selected_detection_score_threshold": detection_threshold,
                "test_detection_precision": detection_metrics["precision"],
                "test_detection_recall": detection_metrics["recall"],
                "test_detection_f1": detection_metrics["f1"],
                "test_detection_mean_iou": detection_metrics[
                    "matched_mean_iou"
                ],
                "test_detection_ap50": detection_metrics["ap50"],
                "test_detection_ap75": detection_metrics["ap75"],
                "test_detection_map_50_95": detection_metrics["map_50_95"],
                "test_left_boundary_mae_px": detection_metrics[
                    "left_boundary_mae_px"
                ],
                "test_right_boundary_mae_px": detection_metrics[
                    "right_boundary_mae_px"
                ],
                "test_peak_count_mae": detection_metrics["peak_count_mae"],
                "best_seed_epoch": summary["best_seed_epoch"],
                "val_best_seed_balanced_accuracy_at_0_5": summary[
                    "best_seed_balanced_accuracy"
                ],
                "selected_seed_threshold": seed_threshold,
                "test_seed_balanced_accuracy": seed_metrics[
                    "balanced_accuracy"
                ],
                "test_seed_auroc": seed_metrics["auroc"],
                "test_seed_average_precision": seed_metrics[
                    "average_precision"
                ],
                "test_seed_f1": seed_metrics["f1"],
                "test_seed_precision": seed_metrics["precision"],
                "test_seed_recall": seed_metrics["recall"],
                "test_seed_specificity": seed_metrics["specificity"],
                "test_seed_brier": seed_metrics["brier"],
                "test_seed_ece_10_bin": seed_metrics["ece_10_bin"],
            }
        )
        write_csv(
            output_root / "ablation_test_metrics.csv",
            aggregate_rows,
            list(aggregate_rows[0]),
        )
        write_json(
            output_root / "ablation_test_metrics.json",
            aggregate_rows,
        )
        print(f"[DONE] {experiment_id}", flush=True)
    write_json(
        output_root / "evaluation_complete.json",
        {
            "status": "completed",
            "experiments": [value[0] for value in EXPERIMENTS],
            "test_manifest": str(
                (dataset_root / "manifests" / "test.jsonl").resolve()
            ),
            "selection_policy": (
                "All checkpoints and thresholds selected on Val before Test "
                "inference; Test used only for final reporting."
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
