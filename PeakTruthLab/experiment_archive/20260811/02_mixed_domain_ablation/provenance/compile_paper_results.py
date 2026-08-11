from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPERIMENTS = (
    ("A_image_only", "Image only"),
    ("B_attr_only", "Attributes only (Seed head)"),
    ("C_naive_concat", "Naive concatenation"),
    ("D_gated_fusion", "Gated fusion"),
)
BOOTSTRAP_METRICS = (
    "test_detection_precision",
    "test_detection_recall",
    "test_detection_f1",
    "test_detection_mean_iou",
    "test_seed_balanced_accuracy",
    "test_seed_auroc",
    "test_seed_average_precision",
    "test_seed_f1",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def binary_ranking_metrics(
    probabilities: list[float], labels: list[int]
) -> tuple[float, float]:
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
    index = 0
    while index < len(ranked):
        score = ranked[index][0]
        group_positive = group_negative = 0
        while index < len(ranked) and ranked[index][0] == score:
            if ranked[index][1]:
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
        previous_tpr = tpr
        previous_fpr = fpr
    return float(auroc), float(average_precision)


def detection_from_rows(
    rows: list[dict[str, str]], indices: list[int]
) -> dict[str, float]:
    tp = sum(int(float(rows[index]["true_positive"])) for index in indices)
    fp = sum(int(float(rows[index]["false_positive"])) for index in indices)
    fn = sum(int(float(rows[index]["false_negative"])) for index in indices)
    matched_iou_sum = sum(
        float(rows[index]["mean_matched_iou"])
        * int(float(rows[index]["true_positive"]))
        for index in indices
    )
    exact = sum(
        str(rows[index]["exact_count"]).strip().lower() in {"1", "true"}
        for index in indices
    )
    count_error = sum(
        float(rows[index]["count_absolute_error"]) for index in indices
    )
    return {
        "test_detection_precision": safe_divide(tp, tp + fp),
        "test_detection_recall": safe_divide(tp, tp + fn),
        "test_detection_f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "test_detection_mean_iou": safe_divide(matched_iou_sum, tp),
        "test_exact_peak_count_accuracy": safe_divide(exact, len(indices)),
        "test_peak_count_mae": safe_divide(count_error, len(indices)),
    }


def seed_from_rows(
    rows: list[dict[str, str]], indices: list[int]
) -> dict[str, float]:
    labels = [int(rows[index]["seed_label"]) for index in indices]
    probabilities = [float(rows[index]["seed_probability"]) for index in indices]
    predictions = [int(rows[index]["seed_prediction"]) for index in indices]
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels, strict=True))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels, strict=True))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels, strict=True))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels, strict=True))
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    auroc, average_precision = binary_ranking_metrics(probabilities, labels)
    return {
        "test_seed_balanced_accuracy": (sensitivity + specificity) / 2.0,
        "test_seed_auroc": auroc,
        "test_seed_average_precision": average_precision,
        "test_seed_f1": safe_divide(2 * tp, 2 * tp + fp + fn),
    }


def bootstrap_experiment(
    detection_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]],
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, tuple[float, float]]:
    if [row["sample_id"] for row in detection_rows] != [
        row["sample_id"] for row in seed_rows
    ]:
        detection_rows = sorted(detection_rows, key=lambda row: row["sample_id"])
        seed_rows = sorted(seed_rows, key=lambda row: row["sample_id"])
    if [row["sample_id"] for row in detection_rows] != [
        row["sample_id"] for row in seed_rows
    ]:
        raise ValueError("Detection and Seed rows do not align by sample_id")
    rng = random.Random(random_seed)
    n = len(detection_rows)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        metrics = detection_from_rows(detection_rows, indices)
        metrics.update(seed_from_rows(seed_rows, indices))
        for name, value in metrics.items():
            distributions[name].append(value)
    return {
        name: (
            percentile(values, 0.025),
            percentile(values, 0.975),
        )
        for name, values in distributions.items()
    }


def paired_bootstrap_differences(
    baseline_detection: list[dict[str, str]],
    baseline_seed: list[dict[str, str]],
    comparison_detection: list[dict[str, str]],
    comparison_seed: list[dict[str, str]],
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, tuple[float, float, float]]:
    def ordered(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        return sorted(rows, key=lambda row: row["sample_id"])

    bd, bs = ordered(baseline_detection), ordered(baseline_seed)
    cd, cs = ordered(comparison_detection), ordered(comparison_seed)
    sample_ids = [row["sample_id"] for row in bd]
    if not (
        sample_ids
        == [row["sample_id"] for row in bs]
        == [row["sample_id"] for row in cd]
        == [row["sample_id"] for row in cs]
    ):
        raise ValueError("Paired bootstrap sample ids are not aligned")
    rng = random.Random(random_seed)
    n = len(sample_ids)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        indices = [rng.randrange(n) for _ in range(n)]
        baseline = detection_from_rows(bd, indices)
        baseline.update(seed_from_rows(bs, indices))
        comparison = detection_from_rows(cd, indices)
        comparison.update(seed_from_rows(cs, indices))
        for name in BOOTSTRAP_METRICS:
            distributions[name].append(comparison[name] - baseline[name])
    output: dict[str, tuple[float, float, float]] = {}
    for name, values in distributions.items():
        lower = percentile(values, 0.025)
        upper = percentile(values, 0.975)
        probability_nonpositive = sum(value <= 0 for value in values) / len(values)
        probability_nonnegative = sum(value >= 0 for value in values) / len(values)
        approximate_p = min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))
        output[name] = (lower, upper, approximate_p)
    return output


def parse_command_flags(command: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        name = token[2:]
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            output[name] = command[index + 1]
            index += 2
        else:
            output[name] = True
            index += 1
    return output


def build_report(
    *,
    quality: dict[str, Any],
    aggregate_rows: list[dict[str, Any]],
    ci_rows: dict[str, dict[str, tuple[float, float]]],
    output_path: Path,
    bootstrap_replicates: int,
) -> None:
    split_map = {row["split"]: row for row in quality["split_summary"]}
    overall = split_map["all"]
    train = split_map["train"]
    validation = split_map["val"]
    test = split_map["test"]
    correction = quality["correction_summary"]
    by_id = {row["experiment_id"]: row for row in aggregate_rows}

    def percent(value: float) -> str:
        return f"{100.0 * float(value):.2f}%"

    def metric_ci(experiment: str, name: str) -> str:
        row = by_id[experiment]
        lower, upper = ci_rows[experiment][name]
        return f"{float(row[name]):.4f} ({lower:.4f} to {upper:.4f})"

    detection_best = max(
        aggregate_rows,
        key=lambda row: float(row["test_detection_f1"]),
    )
    seed_best = max(
        aggregate_rows,
        key=lambda row: float(row["test_seed_balanced_accuracy"]),
    )

    lines = [
        "# PeakTruthLab v2: manuscript-ready Methods, Data and Results",
        "",
        "## Dataset and annotation quality",
        "",
        (
            f"The finalized dataset contains {overall['images']:,} EIC images and "
            f"{overall['true_peaks']:,} manually annotated true-peak boxes "
            f"({overall['mean_peaks_per_image']:.3f} boxes/image). "
            f"Images with no peak, one peak and multiple peaks account for "
            f"{percent(overall['empty_image_fraction'])}, "
            f"{percent(overall['single_peak_fraction'])} and "
            f"{percent(overall['multi_peak_fraction'])}, respectively."
        ),
        "",
        (
            f"Data were grouped by source file before splitting into "
            f"Train ({train['images']:,} images; {train['true_peaks']:,} boxes), "
            f"Validation ({validation['images']:,}; {validation['true_peaks']:,}) "
            f"and Test ({test['images']:,}; {test['true_peaks']:,}). "
            "No source file occurs in more than one split. The Test split was "
            "not used for training, checkpoint selection or threshold selection."
        ),
        "",
        (
            f"A model-assisted review queue of {correction['reviewed_samples']} "
            f"Validation images was inspected manually in LabelMe. "
            f"{correction['manually_changed_samples']} annotations were changed; "
            f"{correction['added_peak_boxes']} boxes were added and "
            f"{correction['removed_peak_boxes']} were removed "
            f"(net change {correction['net_true_peak_delta']:+d}). "
            f"Seed labels changed for {correction['seed_label_changes']} images "
            f"({correction['seed_0_to_1']} negative-to-positive and "
            f"{correction['seed_1_to_0']} positive-to-negative). Original files "
            "were preserved and the corrected labels were released as a separate v2 dataset."
        ),
        "",
        (
            "For manually changed samples, Seed labels were recalculated from the "
            "overlap between the unchanged candidate Seed box and corrected true-peak "
            "boxes. A new match required two-dimensional IoU >=0.05. For an originally "
            "positive Seed, the prior linked target was retained when its overlap with "
            "a corrected box was >=0.30; otherwise the new IoU rule was applied. "
            "The 13 numeric attributes (SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, "
            "DENT, DM, ENT and JAG) describe the unchanged Seed candidate and were "
            "therefore preserved, not recomputed."
        ),
        "",
        "## Model and training",
        "",
        (
            "A shared ConvNeXt-Tiny feature pyramid supplied a Faster R-CNN "
            "true-peak detector and a candidate-specific Seed classifier. The detector "
            "used four FPN levels, anchors of 16/32/64/128 pixels, and tall-anchor "
            "aspect ratios of 1, 2, 4, 8 and 16. The Seed branch pooled a 7 x 7 RoI "
            "feature for the supplied candidate box. The image embedding was 256-D; "
            "the standardized 13-attribute vector was encoded to 64-D by a two-layer "
            "MLP with GELU activations and dropout 0.2."
        ),
        "",
        (
            "Four joint-training conditions were compared: (i) image-only Seed "
            "classification, (ii) attribute-only Seed classification, (iii) direct "
            "concatenation of image and attribute embeddings, and (iv) gated fusion, "
            "where a learned sigmoid gate conditioned on both modalities modulated the "
            "image embedding before concatenation. The detection branch remained "
            "image-based in every condition; the ablation changes only the Seed head "
            "and its auxiliary gradients through the shared backbone."
        ),
        "",
        (
            "All models used 480 x 480 inputs, batch size 8, automatic mixed precision "
            "(FP16), AdamW (learning rate 1e-4; weight decay 1e-4), a "
            "class-weighted binary cross-entropy Seed loss, and 10 epochs without data "
            "augmentation or a learning-rate scheduler. ImageNet-pretrained backbone "
            "weights and random seed 20260725 were used. Validation was performed after "
            "every epoch. The best detection checkpoint maximized Validation F1 at "
            "IoU=0.50; the best Seed checkpoint maximized Validation balanced accuracy."
        ),
        "",
        "## Final evaluation and statistics",
        "",
        (
            "After checkpoint selection, the detection confidence threshold was selected "
            "on Validation data by maximizing F1 over thresholds 0.05-0.95. The Seed "
            "probability threshold was selected on Validation data by maximizing balanced "
            "accuracy over thresholds 0.000-1.000. Test data were then evaluated once. "
            "Detection endpoints include precision, recall, F1, mean matched IoU, AP50, "
            "AP75, mAP50:95, boundary error and peak-count error. Seed endpoints include "
            "balanced accuracy, AUROC, average precision, F1, sensitivity, specificity, "
            "Brier score and 10-bin expected calibration error. Ninety-five percent "
            f"confidence intervals were estimated with {bootstrap_replicates:,} "
            "nonparametric image-level bootstrap replicates."
        ),
        "",
        "## Core Test results",
        "",
        "| Condition | Detection F1 (95% CI) | Mean IoU (95% CI) | AP50 | mAP50:95 | Seed BA (95% CI) | Seed AUROC (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment_id, label in EXPERIMENTS:
        row = by_id[experiment_id]
        lines.append(
            f"| {label} | {metric_ci(experiment_id, 'test_detection_f1')} | "
            f"{metric_ci(experiment_id, 'test_detection_mean_iou')} | "
            f"{float(row['test_detection_ap50']):.4f} | "
            f"{float(row['test_detection_map_50_95']):.4f} | "
            f"{metric_ci(experiment_id, 'test_seed_balanced_accuracy')} | "
            f"{metric_ci(experiment_id, 'test_seed_auroc')} |"
        )
    lines.extend(
        [
            "",
            (
                f"The highest Test detection F1 was obtained by "
                f"{dict(EXPERIMENTS)[detection_best['experiment_id']]} "
                f"({float(detection_best['test_detection_f1']):.4f}). "
                f"The highest Test Seed balanced accuracy was obtained by "
                f"{dict(EXPERIMENTS)[seed_best['experiment_id']]} "
                f"({float(seed_best['test_seed_balanced_accuracy']):.4f})."
            ),
            "",
            "## Reproducibility and interpretation notes",
            "",
            "- All four experiments are single-run, fixed-seed comparisons; bootstrap intervals quantify Test-sample uncertainty, not between-training-run variability.",
            "- Model-assisted review was restricted to a manually inspected Validation subset; Test labels and Train labels were unchanged.",
            "- `Attributes only` refers to the Seed classifier. The object detector always consumes EIC images.",
            "- AP is reported from the detector's retained predictions (internal minimum score 0.05); operating-point precision/recall/F1 use the Validation-selected threshold.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    result_root = args.result_root.resolve()
    evaluation_root = result_root / "test_evaluation"
    completion = read_json(evaluation_root / "evaluation_complete.json")
    if completion.get("status") != "completed":
        raise RuntimeError("Final Test evaluation has not completed")
    output_root = result_root / "paper_tables_figures"
    output_root.mkdir(parents=True, exist_ok=True)

    quality = read_json(result_root / "data_quality" / "data_quality_report.json")
    pipeline = read_json(result_root / "experiments" / "pipeline_status.json")
    aggregate_rows = read_json(evaluation_root / "ablation_test_metrics.json")
    aggregate_by_id = {row["experiment_id"]: row for row in aggregate_rows}

    clean_split_rows: list[dict[str, Any]] = []
    for row in quality["split_summary"]:
        clean_split_rows.append(
            {
                "split": row["split"],
                "images": row["images"],
                "true_peak_boxes": row["true_peaks"],
                "mean_peaks_per_image": row["mean_peaks_per_image"],
                "median_peaks_per_image": row["median_peaks_per_image"],
                "max_peaks_per_image": row["max_peaks_per_image"],
                "empty_images": row["empty_images"],
                "empty_image_fraction": row["empty_image_fraction"],
                "single_peak_images": row["single_peak_images"],
                "single_peak_fraction": row["single_peak_fraction"],
                "multi_peak_images": row["multi_peak_images"],
                "multi_peak_fraction": row["multi_peak_fraction"],
                "seed_positive": row["seed_positive"],
                "seed_negative": row["seed_negative"],
                "seed_positive_fraction": row["seed_positive_fraction"],
                "unique_source_files": row["unique_sources"],
            }
        )
    write_csv(
        output_root / "dataset_split_summary.csv",
        clean_split_rows,
        list(clean_split_rows[0]),
    )

    correction = quality["correction_summary"]
    review_rows = [
        {"metric": "Reviewed samples", "value": correction["reviewed_samples"]},
        {
            "metric": "Manually changed samples",
            "value": correction["manually_changed_samples"],
        },
        {
            "metric": "Unchanged after review",
            "value": correction["unchanged_after_review"],
        },
        {
            "metric": "Added true-peak boxes",
            "value": correction["added_peak_boxes"],
        },
        {
            "metric": "Removed true-peak boxes",
            "value": correction["removed_peak_boxes"],
        },
        {
            "metric": "Net true-peak box change",
            "value": correction["net_true_peak_delta"],
        },
        {
            "metric": "Seed-label changes",
            "value": correction["seed_label_changes"],
        },
        {
            "metric": "Seed 0->1",
            "value": correction["seed_0_to_1"],
        },
        {
            "metric": "Seed 1->0",
            "value": correction["seed_1_to_0"],
        },
    ]
    write_csv(output_root / "review_summary.csv", review_rows, ["metric", "value"])

    first_experiment = pipeline["experiments"]["A_image_only"]
    flags = parse_command_flags(first_experiment["command"])
    run_config = read_json(
        result_root / "experiments" / "A_image_only" / "run_config.json"
    )
    training_positive = run_config["train"]["seed_positive"]
    training_negative = run_config["train"]["seed_negative"]
    hyperparameter_rows = [
        {"section": "Data", "parameter": "Train images", "value": 10525},
        {"section": "Data", "parameter": "Validation images", "value": 2507},
        {"section": "Data", "parameter": "Test images", "value": 2285},
        {"section": "Input", "parameter": "Image size", "value": "480 x 480 px"},
        {"section": "Optimization", "parameter": "Epochs", "value": 10},
        {"section": "Optimization", "parameter": "Batch size", "value": 8},
        {
            "section": "Optimization",
            "parameter": "Gradient accumulation",
            "value": 1,
        },
        {"section": "Optimization", "parameter": "Optimizer", "value": "AdamW"},
        {
            "section": "Optimization",
            "parameter": "Learning rate",
            "value": float(flags["learning-rate"]),
        },
        {
            "section": "Optimization",
            "parameter": "Weight decay",
            "value": float(flags["weight-decay"]),
        },
        {"section": "Optimization", "parameter": "LR scheduler", "value": "None"},
        {
            "section": "Optimization",
            "parameter": "Mixed precision",
            "value": "FP16 AMP",
        },
        {
            "section": "Optimization",
            "parameter": "Data augmentation",
            "value": "None",
        },
        {
            "section": "Loss",
            "parameter": "Seed loss",
            "value": "Weighted BCE",
        },
        {
            "section": "Loss",
            "parameter": "Seed positive weight",
            "value": training_negative / training_positive,
        },
        {
            "section": "Model",
            "parameter": "Backbone",
            "value": "ConvNeXt-Tiny + FPN",
        },
        {
            "section": "Model",
            "parameter": "Pretrained",
            "value": "ImageNet weights",
        },
        {
            "section": "Model",
            "parameter": "Detector",
            "value": "Faster R-CNN (background + True_Peak)",
        },
        {
            "section": "Model",
            "parameter": "Anchor sizes",
            "value": "16, 32, 64, 128 px",
        },
        {
            "section": "Model",
            "parameter": "Anchor aspect ratios (H/W)",
            "value": "1, 2, 4, 8, 16",
        },
        {
            "section": "Model",
            "parameter": "RPN proposals train/test",
            "value": f"{flags['rpn-train-proposals']}/{flags['rpn-test-proposals']}",
        },
        {
            "section": "Model",
            "parameter": "Box NMS threshold",
            "value": float(flags["box-nms-thresh"]),
        },
        {"section": "Model", "parameter": "Seed RoI output", "value": "7 x 7"},
        {
            "section": "Model",
            "parameter": "Image/attribute embedding",
            "value": "256-D / 64-D",
        },
        {
            "section": "Model",
            "parameter": "Fusion dropout",
            "value": 0.2,
        },
        {
            "section": "Reproducibility",
            "parameter": "Random seed",
            "value": int(flags["seed"]),
        },
        {
            "section": "Reproducibility",
            "parameter": "PyTorch / CUDA",
            "value": f"{run_config['torch_version']} / {run_config['cuda_runtime']}",
        },
        {
            "section": "Reproducibility",
            "parameter": "GPU",
            "value": "NVIDIA GeForce RTX 4070 SUPER",
        },
    ]
    write_csv(
        output_root / "hyperparameters.csv",
        hyperparameter_rows,
        ["section", "parameter", "value"],
    )

    history_rows: list[dict[str, Any]] = []
    val_best_rows: list[dict[str, Any]] = []
    for experiment_id, label in EXPERIMENTS:
        experiment_dir = result_root / "experiments" / experiment_id
        summary = read_json(experiment_dir / "summary.json")
        for item in read_json(experiment_dir / "history.json"):
            detection = item["validation"]["detection"]
            seed = item["validation"]["seed"]
            history_rows.append(
                {
                    "experiment_id": experiment_id,
                    "condition": label,
                    "fusion_mode": summary["fusion_mode"],
                    "epoch": item["epoch"],
                    "loss_total": item["train_losses"]["loss_total"],
                    "loss_detection_box_reg": item["train_losses"]["loss_box_reg"],
                    "loss_detection_classifier": item["train_losses"][
                        "loss_classifier"
                    ],
                    "loss_rpn_objectness": item["train_losses"]["loss_objectness"],
                    "loss_rpn_box_reg": item["train_losses"]["loss_rpn_box_reg"],
                    "loss_seed": item["train_losses"]["loss_seed_cls"],
                    "val_detection_precision": detection["precision"],
                    "val_detection_recall": detection["recall"],
                    "val_detection_f1": detection["f1"],
                    "val_detection_mean_iou": detection["matched_mean_iou"],
                    "val_seed_balanced_accuracy": seed["balanced_accuracy"],
                    "val_seed_auroc": seed["auroc"],
                    "val_seed_average_precision": seed["average_precision"],
                    "val_seed_f1": seed["f1"],
                    "val_seed_brier": seed["brier"],
                    "val_seed_ece_10_bin": seed["ece_10_bin"],
                    "epoch_seconds": item["elapsed_seconds"],
                }
            )
        val_best_rows.append(
            {
                "experiment_id": experiment_id,
                "condition": label,
                "fusion_mode": summary["fusion_mode"],
                "parameter_count": summary["parameter_count"],
                "training_seconds": summary["elapsed_seconds"],
                "peak_gpu_memory_allocated_bytes": summary["device"][
                    "max_memory_allocated_bytes"
                ],
                "peak_gpu_memory_reserved_bytes": summary["device"][
                    "max_memory_reserved_bytes"
                ],
                "best_detection_epoch": summary["best_detection_epoch"],
                "best_val_detection_f1_at_0_5": summary["best_detection_f1"],
                "best_seed_epoch": summary["best_seed_epoch"],
                "best_val_seed_balanced_accuracy_at_0_5": summary[
                    "best_seed_balanced_accuracy"
                ],
            }
        )
    write_csv(
        output_root / "validation_training_curves.csv",
        history_rows,
        list(history_rows[0]),
    )
    write_csv(
        output_root / "validation_best_checkpoints.csv",
        val_best_rows,
        list(val_best_rows[0]),
    )

    per_experiment_rows: dict[
        str, tuple[list[dict[str, str]], list[dict[str, str]]]
    ] = {}
    ci_by_experiment: dict[str, dict[str, tuple[float, float]]] = {}
    for experiment_index, (experiment_id, _) in enumerate(EXPERIMENTS):
        evaluation_dir = evaluation_root / experiment_id
        detection_rows = read_csv(
            evaluation_dir / "test_detection_image_metrics.csv"
        )
        seed_rows = read_csv(evaluation_dir / "test_seed_predictions.csv")
        per_experiment_rows[experiment_id] = (detection_rows, seed_rows)
        ci_by_experiment[experiment_id] = bootstrap_experiment(
            detection_rows,
            seed_rows,
            replicates=args.bootstrap_replicates,
            random_seed=args.random_seed + experiment_index,
        )

    core_rows: list[dict[str, Any]] = []
    for experiment_id, label in EXPERIMENTS:
        row = dict(aggregate_by_id[experiment_id])
        row["condition"] = label
        for metric, (lower, upper) in ci_by_experiment[experiment_id].items():
            row[f"{metric}_ci95_low"] = lower
            row[f"{metric}_ci95_high"] = upper
        core_rows.append(row)
    core_fields = (
        ["experiment_id", "condition", "fusion_mode"]
        + [
            field
            for field in core_rows[0]
            if field not in {"experiment_id", "condition", "fusion_mode"}
        ]
    )
    write_csv(output_root / "ablation_test_core.csv", core_rows, core_fields)

    baseline_detection, baseline_seed = per_experiment_rows["A_image_only"]
    delta_rows: list[dict[str, Any]] = []
    for experiment_index, (experiment_id, label) in enumerate(EXPERIMENTS[1:], start=1):
        comparison_detection, comparison_seed = per_experiment_rows[experiment_id]
        intervals = paired_bootstrap_differences(
            baseline_detection,
            baseline_seed,
            comparison_detection,
            comparison_seed,
            replicates=args.bootstrap_replicates,
            random_seed=args.random_seed + 100 + experiment_index,
        )
        comparison = aggregate_by_id[experiment_id]
        baseline = aggregate_by_id["A_image_only"]
        for metric in BOOTSTRAP_METRICS:
            lower, upper, approximate_p = intervals[metric]
            delta_rows.append(
                {
                    "experiment_id": experiment_id,
                    "condition": label,
                    "baseline": "A_image_only",
                    "metric": metric,
                    "point_difference": float(comparison[metric])
                    - float(baseline[metric]),
                    "paired_bootstrap_ci95_low": lower,
                    "paired_bootstrap_ci95_high": upper,
                    "approximate_two_sided_bootstrap_p": approximate_p,
                    "bootstrap_replicates": args.bootstrap_replicates,
                }
            )
    write_csv(
        output_root / "ablation_delta_vs_image_only.csv",
        delta_rows,
        list(delta_rows[0]),
    )

    calibration_rows: list[dict[str, Any]] = []
    detection_subgroup_rows: list[dict[str, Any]] = []
    seed_subgroup_rows: list[dict[str, Any]] = []
    for experiment_id, label in EXPERIMENTS:
        for row in read_csv(
            evaluation_root / experiment_id / "test_seed_calibration.csv"
        ):
            calibration_rows.append(
                {"experiment_id": experiment_id, "condition": label, **row}
            )
        for row in read_csv(
            evaluation_root / experiment_id / "test_detection_subgroups.csv"
        ):
            detection_subgroup_rows.append(
                {"experiment_id": experiment_id, "condition": label, **row}
            )
        for row in read_csv(
            evaluation_root / experiment_id / "test_seed_subgroups.csv"
        ):
            seed_subgroup_rows.append(
                {"experiment_id": experiment_id, "condition": label, **row}
            )
    write_csv(
        output_root / "test_seed_calibration_all.csv",
        calibration_rows,
        list(calibration_rows[0]),
    )
    write_csv(
        output_root / "test_detection_subgroups_all.csv",
        detection_subgroup_rows,
        list(detection_subgroup_rows[0]),
    )
    write_csv(
        output_root / "test_seed_subgroups_all.csv",
        seed_subgroup_rows,
        list(seed_subgroup_rows[0]),
    )

    for source_name in (
        "peak_count_distribution.csv",
        "attribute_summary.csv",
        "box_geometry_summary.csv",
        "source_distribution.csv",
        "review_corrections.csv",
    ):
        shutil.copy2(
            result_root / "data_quality" / source_name,
            output_root / source_name,
        )

    completion_record = {
        "status": "completed",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_random_seed": args.random_seed,
        "test_selection_policy": completion["selection_policy"],
        "dataset_archive_sha256": quality["dataset_archive"]["sha256"],
        "source_file_leakage": quality["dataset_info"]["validation"][
            "source_file_leakage"
        ],
        "outputs": sorted(path.name for path in output_root.iterdir()),
    }
    write_json(output_root / "paper_results_manifest.json", completion_record)
    build_report(
        quality=quality,
        aggregate_rows=aggregate_rows,
        ci_rows=ci_by_experiment,
        output_path=output_root / "paper_methods_data_results.md",
        bootstrap_replicates=args.bootstrap_replicates,
    )
    write_json(
        output_root / "core_results.json",
        {
            "dataset": clean_split_rows,
            "review": review_rows,
            "validation_best": val_best_rows,
            "test_ablation": core_rows,
            "paired_differences": delta_rows,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260728)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
