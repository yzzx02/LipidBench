from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from evaluate_ablation_final import (  # noqa: E402
    binary_ranking_curves,
    detection_ap_metrics,
    detection_subgroup_rows,
    detection_threshold_metrics,
    detection_threshold_sweep,
    infer_manifest,
    seed_calibration_rows,
    seed_subgroup_rows,
    seed_threshold_metrics,
    seed_threshold_sweep,
    write_csv,
    write_json,
)


DOMAINS = (
    ("External A", "external_A.jsonl", "ST003127"),
    ("External B", "external_B.jsonl", "ST003941"),
    ("External C", "external_C.jsonl", "ST003514"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected.upper():
        raise RuntimeError(f"Frozen file hash mismatch: {path}: {actual}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def seed_prediction_rows(samples: list[dict[str, Any]], threshold: float):
    return [
        {
            "sample_id": sample["sample_id"],
            "source_file": sample["source_file"],
            "true_peak_count_class": sample["true_peak_count_class"],
            "seed_label": sample["seed_label"],
            "seed_probability": sample["seed_probability"],
            "seed_prediction": int(sample["seed_probability"] >= threshold),
            "correct": int(sample["seed_probability"] >= threshold) == sample["seed_label"],
        }
        for sample in samples
    ]


def evaluate_domain(
    *,
    label: str,
    manifest: Path,
    experiment_dir: Path,
    image_root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    detection_threshold: float,
    seed_threshold: float,
    directory_name: str | None = None,
) -> dict[str, Any]:
    domain_dir = output_dir / (directory_name or label.replace(" ", "_"))
    domain_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = experiment_dir / "attribute_preprocessing.json"

    detection_samples, detection_checkpoint = infer_manifest(
        checkpoint_path=experiment_dir / "best_detection.pt",
        manifest_path=manifest,
        image_root=image_root,
        preprocessor_path=preprocessor,
        device=device,
        batch_size=batch_size,
        amp=True,
    )
    detection_metrics, detection_predictions, detection_image_rows = detection_threshold_metrics(
        detection_samples,
        score_threshold=detection_threshold,
        iou_threshold=0.5,
    )
    ap_metrics, ap_summary_rows, ap_curve_rows = detection_ap_metrics(detection_samples)
    detection_metrics.update(ap_metrics)
    detection_subgroups = detection_subgroup_rows(
        detection_samples,
        score_threshold=detection_threshold,
    )
    write_csv(
        domain_dir / "detection_predictions.csv",
        detection_predictions,
        list(detection_predictions[0]) if detection_predictions else ["sample_id"],
    )
    write_csv(domain_dir / "detection_image_metrics.csv", detection_image_rows, list(detection_image_rows[0]))
    write_csv(domain_dir / "detection_ap_summary.csv", ap_summary_rows, list(ap_summary_rows[0]))
    write_csv(domain_dir / "detection_pr_curves.csv", ap_curve_rows, list(ap_curve_rows[0]))
    write_csv(domain_dir / "detection_subgroups.csv", detection_subgroups, list(detection_subgroups[0]))
    sample_count = len(detection_samples)
    del detection_samples
    gc.collect()

    seed_samples, seed_checkpoint = infer_manifest(
        checkpoint_path=experiment_dir / "best_seed.pt",
        manifest_path=manifest,
        image_root=image_root,
        preprocessor_path=preprocessor,
        device=device,
        batch_size=batch_size,
        amp=True,
    )
    seed_metrics = seed_threshold_metrics(seed_samples, threshold=seed_threshold)
    ranking, roc_rows, pr_rows = binary_ranking_curves(
        [sample["seed_probability"] for sample in seed_samples],
        [sample["seed_label"] for sample in seed_samples],
    )
    seed_metrics.update(ranking)
    calibration_rows, ece = seed_calibration_rows(seed_samples)
    seed_metrics["ece_10_bin"] = ece
    seed_subgroups = seed_subgroup_rows(seed_samples, threshold=seed_threshold)
    prediction_rows = seed_prediction_rows(seed_samples, seed_threshold)
    write_csv(domain_dir / "seed_predictions.csv", prediction_rows, list(prediction_rows[0]))
    write_csv(domain_dir / "seed_roc_curve.csv", roc_rows, list(roc_rows[0]))
    write_csv(domain_dir / "seed_pr_curve.csv", pr_rows, list(pr_rows[0]))
    write_csv(domain_dir / "seed_calibration.csv", calibration_rows, list(calibration_rows[0]))
    write_csv(domain_dir / "seed_subgroups.csv", seed_subgroups, list(seed_subgroups[0]))
    del seed_samples
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = {
        "condition": label,
        "manifest": str(manifest.resolve()),
        "samples": sample_count,
        "detection_checkpoint": detection_checkpoint,
        "seed_checkpoint": seed_checkpoint,
        "detection": detection_metrics,
        "seed": seed_metrics,
        "thresholds": {
            "detection_score": detection_threshold,
            "seed_probability": seed_threshold,
            "matching_iou": 0.5,
        },
    }
    write_json(domain_dir / "metrics.json", result)
    return result


def flat_row(result: dict[str, Any]) -> dict[str, Any]:
    detection = result["detection"]
    seed = result["seed"]
    return {
        "condition": result["condition"],
        "samples": result["samples"],
        "detection_precision": detection["precision"],
        "detection_recall": detection["recall"],
        "detection_f1": detection["f1"],
        "detection_mean_iou": detection["matched_mean_iou"],
        "detection_median_iou": detection["matched_median_iou"],
        "detection_ap50": detection["ap50"],
        "detection_ap75": detection["ap75"],
        "detection_map_50_95": detection["map_50_95"],
        "seed_balanced_accuracy": seed["balanced_accuracy"],
        "seed_auroc": seed["auroc"],
        "seed_auprc": seed["average_precision"],
        "seed_f1": seed["f1"],
        "seed_precision": seed["precision"],
        "seed_recall": seed["recall"],
        "seed_specificity": seed["specificity"],
        "seed_brier": seed["brier"],
    }


def run(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or protocol.get("fusion_mode") != "naive_concat":
        raise RuntimeError("Cross-domain protocol is not frozen for Naive concat")
    if protocol["external_domains"] != {label: study for label, _, study in DOMAINS}:
        raise RuntimeError("External-domain identities differ from the confirmed protocol")

    verify_hash(args.split_root / "train.jsonl", protocol["split_sha256"]["train.jsonl"])
    verify_hash(args.split_root / "val.jsonl", protocol["split_sha256"]["val.jsonl"])
    summary = json.loads((args.experiment_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["fusion_mode"] != "naive_concat" or summary["epochs_completed"] != 15:
        raise RuntimeError("Training output is not the frozen 15-epoch Naive concat run")
    if summary.get("test_manifest_used") is not False:
        raise RuntimeError("Training summary indicates test data use")

    args.threshold_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(int(protocol["random_seed"]))
    preprocessor = args.experiment_dir / "attribute_preprocessing.json"

    val_detection, detection_checkpoint = infer_manifest(
        checkpoint_path=args.experiment_dir / "best_detection.pt",
        manifest_path=args.split_root / "val.jsonl",
        image_root=args.image_root,
        preprocessor_path=preprocessor,
        device=device,
        batch_size=args.batch_size,
        amp=True,
    )
    detection_threshold, detection_sweep = detection_threshold_sweep(val_detection)
    write_csv(args.threshold_dir / "val_detection_threshold_sweep.csv", detection_sweep, list(detection_sweep[0]))
    del val_detection
    gc.collect()

    val_seed, seed_checkpoint = infer_manifest(
        checkpoint_path=args.experiment_dir / "best_seed.pt",
        manifest_path=args.split_root / "val.jsonl",
        image_root=args.image_root,
        preprocessor_path=preprocessor,
        device=device,
        batch_size=args.batch_size,
        amp=True,
    )
    seed_threshold, seed_sweep = seed_threshold_sweep(val_seed)
    write_csv(args.threshold_dir / "val_seed_threshold_sweep.csv", seed_sweep, list(seed_sweep[0]))
    del val_seed
    gc.collect()

    selection = {
        "status": "locked_before_external_inference",
        "external_data_read_during_checkpoint_or_threshold_selection": False,
        "protocol_lock": str(args.protocol_lock.resolve()),
        "fusion_mode": "naive_concat",
        "random_seed": int(protocol["random_seed"]),
        "validation_manifest": str((args.split_root / "val.jsonl").resolve()),
        "detection": {
            "checkpoint": detection_checkpoint,
            "selection_metric": "Val Detection F1 at IoU=0.50",
            "selected_score_threshold": detection_threshold,
        },
        "seed": {
            "checkpoint": seed_checkpoint,
            "selection_metric": "Val Seed balanced accuracy",
            "selected_probability_threshold": seed_threshold,
        },
    }
    selection_path = args.threshold_dir / "selection_before_external.json"
    write_json(selection_path, selection)
    if not selection_path.is_file():
        raise RuntimeError("Selection lock was not persisted")

    for _, filename, _ in DOMAINS:
        verify_hash(args.split_root / filename, protocol["split_sha256"][filename])

    domain_results = []
    for label, filename, study in DOMAINS:
        print(f"[EXTERNAL START] {label} {study}", flush=True)
        result = evaluate_domain(
            label=label,
            manifest=args.split_root / filename,
            experiment_dir=args.experiment_dir,
            image_root=args.image_root,
            output_dir=args.metrics_dir,
            device=device,
            batch_size=args.batch_size,
            detection_threshold=detection_threshold,
            seed_threshold=seed_threshold,
        )
        domain_results.append(result)
        print(f"[EXTERNAL DONE] {label}", flush=True)

    raw_rows = [flat_row(result) for result in domain_results]
    metric_fields = [key for key in raw_rows[0] if key not in {"condition", "samples"}]
    macro_row = {
        "condition": "External macro average",
        "samples": sum(row["samples"] for row in raw_rows),
        **{field: fmean(float(row[field]) for row in raw_rows) for field in metric_fields},
    }
    summary_rows = raw_rows + [macro_row]
    write_csv(args.summary_dir / "external_domain_raw_metrics.csv", raw_rows, list(raw_rows[0]))
    write_csv(args.summary_dir / "external_domain_summary.csv", summary_rows, list(summary_rows[0]))

    baseline_rows = [
        row for row in read_csv_rows(args.mixed_baseline_raw)
        if row["experiment_id"] == "C_naive_concat"
    ]
    baseline_map = {
        "detection_f1": "detection_f1",
        "detection_mean_iou": "detection_mean_iou",
        "seed_balanced_accuracy": "seed_balanced_accuracy",
        "seed_auroc": "seed_auroc",
        "seed_auprc": "seed_average_precision",
    }
    baseline = {
        output: fmean(float(row[source]) for row in baseline_rows)
        for output, source in baseline_map.items()
    }
    baseline_sd = {
        output: stdev(float(row[source]) for row in baseline_rows)
        for output, source in baseline_map.items()
    }
    comparison = [{
        "condition": "Mixed-domain baseline (3-seed mean)",
        "samples": "",
        **baseline,
        **{f"{key}_sd": value for key, value in baseline_sd.items()},
    }]
    for row in summary_rows:
        comparison.append({
            "condition": row["condition"],
            "samples": row["samples"],
            **{key: row[key] for key in baseline},
            **{f"{key}_sd": "" for key in baseline},
        })
    write_csv(args.summary_dir / "cross_domain_comparison.csv", comparison, list(comparison[0]))

    gap_rows = []
    for row in summary_rows:
        gap_rows.append({
            "condition": row["condition"],
            **{f"delta_{key}": baseline[key] - float(row[key]) for key in baseline},
        })
    write_csv(args.summary_dir / "domain_gap.csv", gap_rows, list(gap_rows[0]))

    report = [
        "# Cross-domain generalization: single-seed result",
        "",
        "- Model: Naive concat",
        f"- Random seed: {protocol['random_seed']}",
        f"- Best detection epoch: {summary['best_detection_epoch']}",
        f"- Best Seed epoch: {summary['best_seed_epoch']}",
        f"- Val-selected detection threshold: {detection_threshold}",
        f"- Val-selected Seed threshold: {seed_threshold}",
        "- External data were evaluated only after both checkpoints and thresholds were locked.",
        "",
        "| Condition | Detection F1 | Mean IoU | Seed BA | Seed AUROC | Seed AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['condition']} | {row['detection_f1']:.4f} | {row['detection_mean_iou']:.4f} | "
            f"{row['seed_balanced_accuracy']:.4f} | {row['seed_auroc']:.4f} | {row['seed_auprc']:.4f} |"
        )
    report.extend([
        "",
        f"Training time: {summary['elapsed_seconds'] / 60:.1f} min",
        f"Peak CUDA allocated memory: {summary['device']['max_memory_allocated_bytes'] / 1024**3:.2f} GiB",
        "",
        "External macro average is the unweighted arithmetic mean of the three separately evaluated domains. No pooled-only External metric is used.",
    ])
    (args.summary_dir / "cross_domain_results_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(args.summary_dir / "evaluation_complete.json", {
        "status": "completed",
        "selection_lock": str(selection_path.resolve()),
        "domains": [label for label, _, _ in DOMAINS],
        "macro_average": macro_row,
        "external_used_only_after_val_selection": True,
    })
    args.provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), args.provenance_dir)
    shutil.copy2(WORK_ROOT / "evaluate_ablation_final.py", args.provenance_dir)
    print(json.dumps({"status": "completed", "macro": macro_row}, ensure_ascii=True, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--threshold-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--mixed-baseline-raw", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
