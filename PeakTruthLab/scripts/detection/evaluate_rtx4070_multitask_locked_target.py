from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from evaluate_rtx4070_multitask_final import (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_fields() -> list[str]:
    return [
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
    ]


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to evaluate on CPU")
    experiment_dir = args.experiment_dir.resolve()
    val_manifest = args.val_manifest.resolve()
    target_manifest = args.target_manifest.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    complete_path = output_dir / "evaluation_complete.json"
    if complete_path.exists():
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
            return
        raise RuntimeError(f"invalid existing completion marker: {complete_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"{output_dir} is non-empty; pass --resume only after auditing partial output"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = experiment_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        raise ValueError("training summary is not complete")
    if summary.get("fusion_mode") != "naive_concat":
        raise ValueError("locked target evaluator is restricted to Naive concat")
    if summary.get("test_manifest_used") is not False:
        raise ValueError("training summary does not prove Test/heldout was untouched")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    preprocessor = experiment_dir / "attribute_preprocessing.json"

    # Phase 1: checkpoint/threshold selection uses Val only.  The target
    # manifest is deliberately not opened or hashed before this phase ends.
    val_detection, detection_checkpoint = infer_manifest(
        checkpoint_path=experiment_dir / "best_detection.pt",
        manifest_path=val_manifest,
        image_root=image_root,
        preprocessor_path=preprocessor,
        device=device,
        batch_size=args.batch_size,
        amp=True,
    )
    detection_threshold, detection_sweep = detection_threshold_sweep(val_detection)
    write_csv(
        output_dir / "val_detection_threshold_sweep.csv",
        detection_sweep,
        list(detection_sweep[0]),
    )
    del val_detection
    gc.collect()
    torch.cuda.empty_cache()

    val_seed, seed_checkpoint = infer_manifest(
        checkpoint_path=experiment_dir / "best_seed.pt",
        manifest_path=val_manifest,
        image_root=image_root,
        preprocessor_path=preprocessor,
        device=device,
        batch_size=args.batch_size,
        amp=True,
    )
    seed_threshold, seed_sweep = seed_threshold_sweep(val_seed)
    write_csv(
        output_dir / "val_seed_threshold_sweep.csv",
        seed_sweep,
        list(seed_sweep[0]),
    )
    del val_seed
    gc.collect()
    torch.cuda.empty_cache()

    selection: dict[str, Any] = {
        "status": "locked_before_target_access",
        "model": "naive_concat",
        "seed": args.seed,
        "val_manifest": str(val_manifest),
        "val_manifest_sha256": sha256(val_manifest),
        "target_name": args.target_name,
        "target_data_inspected_during_selection": False,
        "detection": {
            "checkpoint": detection_checkpoint,
            "selection_metric": "Val detection F1 at IoU=0.50",
            "selected_score_threshold": detection_threshold,
        },
        "seed_classification": {
            "checkpoint": seed_checkpoint,
            "selection_metric": "Val Seed balanced accuracy",
            "selected_probability_threshold": seed_threshold,
        },
    }
    write_json(output_dir / "selection_before_target_access.json", selection)

    # Phase 2: single locked evaluation on Test or one held-out domain.
    target_hash = sha256(target_manifest)
    target_detection, target_detection_checkpoint = infer_manifest(
        checkpoint_path=experiment_dir / "best_detection.pt",
        manifest_path=target_manifest,
        image_root=image_root,
        preprocessor_path=preprocessor,
        device=device,
        batch_size=args.batch_size,
        amp=True,
    )
    detection_metrics, detection_predictions, detection_image_rows = (
        detection_threshold_metrics(
            target_detection,
            score_threshold=detection_threshold,
            iou_threshold=0.5,
        )
    )
    ap_metrics, ap_summary, ap_curves = detection_ap_metrics(target_detection)
    detection_metrics.update(ap_metrics)
    detection_subgroups = detection_subgroup_rows(
        target_detection,
        score_threshold=detection_threshold,
    )
    write_csv(
        output_dir / "target_detection_predictions.csv",
        detection_predictions,
        list(detection_predictions[0]) if detection_predictions else prediction_fields(),
    )
    write_csv(
        output_dir / "target_detection_image_metrics.csv",
        detection_image_rows,
        list(detection_image_rows[0]),
    )
    write_csv(
        output_dir / "target_detection_ap_summary.csv",
        ap_summary,
        list(ap_summary[0]),
    )
    write_csv(
        output_dir / "target_detection_pr_curves.csv",
        ap_curves,
        list(ap_curves[0]),
    )
    write_csv(
        output_dir / "target_detection_subgroups.csv",
        detection_subgroups,
        list(detection_subgroups[0]),
    )
    del target_detection
    gc.collect()
    torch.cuda.empty_cache()

    target_seed, target_seed_checkpoint = infer_manifest(
        checkpoint_path=experiment_dir / "best_seed.pt",
        manifest_path=target_manifest,
        image_root=image_root,
        preprocessor_path=preprocessor,
        device=device,
        batch_size=args.batch_size,
        amp=True,
    )
    seed_metrics = seed_threshold_metrics(target_seed, threshold=seed_threshold)
    ranking_metrics, roc_rows, pr_rows = binary_ranking_curves(
        [sample["seed_probability"] for sample in target_seed],
        [sample["seed_label"] for sample in target_seed],
    )
    seed_metrics.update(ranking_metrics)
    calibration_rows, ece = seed_calibration_rows(target_seed)
    seed_metrics["ece_10_bin"] = ece
    seed_subgroups = seed_subgroup_rows(target_seed, threshold=seed_threshold)
    seed_rows = [
        {
            "sample_id": sample["sample_id"],
            "source_file": sample["source_file"],
            "true_peak_count_class": sample["true_peak_count_class"],
            "seed_label": sample["seed_label"],
            "seed_probability": sample["seed_probability"],
            "seed_prediction": int(sample["seed_probability"] >= seed_threshold),
            "correct": int(sample["seed_probability"] >= seed_threshold)
            == sample["seed_label"],
        }
        for sample in target_seed
    ]
    write_csv(output_dir / "target_seed_predictions.csv", seed_rows, list(seed_rows[0]))
    write_csv(output_dir / "target_seed_roc_curve.csv", roc_rows, list(roc_rows[0]))
    write_csv(output_dir / "target_seed_pr_curve.csv", pr_rows, list(pr_rows[0]))
    write_csv(
        output_dir / "target_seed_calibration.csv",
        calibration_rows,
        list(calibration_rows[0]),
    )
    write_csv(
        output_dir / "target_seed_subgroups.csv",
        seed_subgroups,
        list(seed_subgroups[0]),
    )
    del target_seed
    gc.collect()
    torch.cuda.empty_cache()

    metrics = {
        "status": "complete",
        "target_name": args.target_name,
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": target_hash,
        "selection_was_locked_before_target_access": True,
        "selection": selection,
        "detection_checkpoint": target_detection_checkpoint,
        "seed_checkpoint": target_seed_checkpoint,
        "target_detection": detection_metrics,
        "target_seed": seed_metrics,
    }
    write_json(output_dir / "target_metrics.json", metrics)
    complete = {
        "status": "complete",
        "target_name": args.target_name,
        "target_manifest_sha256": target_hash,
        "detection_f1": detection_metrics["f1"],
        "detection_precision": detection_metrics["precision"],
        "detection_recall": detection_metrics["recall"],
        "detection_mean_iou": detection_metrics["matched_mean_iou"],
        "seed_balanced_accuracy": seed_metrics["balanced_accuracy"],
        "seed_auroc": seed_metrics["auroc"],
    }
    write_json(complete_path, complete)
    print(json.dumps(complete, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock thresholds on Val, then evaluate one untouched target once."
    )
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
