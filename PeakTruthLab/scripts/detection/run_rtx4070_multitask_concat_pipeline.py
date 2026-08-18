from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in seen:
                fieldnames.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_and_log(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\nCOMMAND: " + subprocess.list2cmdline(command) + "\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def training_command(
    *,
    args: argparse.Namespace,
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.train_script.resolve()),
        "--mode",
        "pilot",
        "--train-manifest",
        str(train_manifest),
        "--val-manifest",
        str(val_manifest),
        "--image-root",
        str(args.image_root.resolve()),
        "--output-dir",
        str(output_dir),
        "--fusion-mode",
        "naive_concat",
        "--training-task",
        "joint",
        "--train-limit",
        "999999",
        "--val-limit",
        "999999",
        "--epochs",
        "30",
        "--batch-size",
        "16",
        "--gradient-accumulation-steps",
        "1",
        "--amp",
        "--image-size",
        "480",
        "--learning-rate",
        "0.0001",
        "--weight-decay",
        "0.0001",
        "--score-threshold",
        "0.5",
        "--iou-threshold",
        "0.5",
        "--rpn-train-proposals",
        "128",
        "--rpn-test-proposals",
        "64",
        "--box-nms-thresh",
        "0.6",
        "--overlay-count",
        "8",
        "--num-workers",
        "0",
        "--seed",
        "20260814",
        "--pretrained",
    ]
    if (output_dir / "last.pt").exists():
        command.append("--resume")
    return command


def evaluation_command(
    *,
    args: argparse.Namespace,
    experiment_dir: Path,
    val_manifest: Path,
    target_manifest: Path,
    target_name: str,
    output_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(args.eval_script.resolve()),
        "--experiment-dir",
        str(experiment_dir),
        "--val-manifest",
        str(val_manifest),
        "--target-manifest",
        str(target_manifest),
        "--target-name",
        target_name,
        "--image-root",
        str(args.image_root.resolve()),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        "16",
        "--seed",
        "20260814",
    ]
    if output_dir.exists() and any(output_dir.iterdir()):
        command.append("--resume")
    return command


def ensure_training(
    *,
    args: argparse.Namespace,
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("status") != "ok" or summary.get("fusion_mode") != "naive_concat":
            raise RuntimeError(f"invalid training summary: {summary_path}")
        if int(summary.get("epochs_completed", 0)) != 30:
            raise RuntimeError(f"training summary did not complete 30 epochs: {summary_path}")
        print(json.dumps({"event": "training_already_complete", "output": str(output_dir)}), flush=True)
        return summary
    print(json.dumps({"event": "training_start", "output": str(output_dir)}), flush=True)
    run_and_log(
        training_command(
            args=args,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            output_dir=output_dir,
        ),
        log_path,
    )
    return read_json(summary_path)


def ensure_evaluation(
    *,
    args: argparse.Namespace,
    experiment_dir: Path,
    val_manifest: Path,
    target_manifest: Path,
    target_name: str,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    metrics_path = output_dir / "target_metrics.json"
    complete_path = output_dir / "evaluation_complete.json"
    if metrics_path.exists() and complete_path.exists():
        metrics = read_json(metrics_path)
        if metrics.get("status") != "complete" or metrics.get("target_name") != target_name:
            raise RuntimeError(f"invalid target evaluation: {metrics_path}")
        print(json.dumps({"event": "evaluation_already_complete", "target": target_name}), flush=True)
        return metrics
    print(json.dumps({"event": "locked_target_evaluation_start", "target": target_name}), flush=True)
    run_and_log(
        evaluation_command(
            args=args,
            experiment_dir=experiment_dir,
            val_manifest=val_manifest,
            target_manifest=target_manifest,
            target_name=target_name,
            output_dir=output_dir,
        ),
        log_path,
    )
    return read_json(metrics_path)


def metric_row(scope: str, metrics: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    detection = metrics["target_detection"]
    seed = metrics["target_seed"]
    return {
        "scope": scope,
        "detection_precision": detection["precision"],
        "detection_recall": detection["recall"],
        "detection_f1": detection["f1"],
        "detection_mean_iou": detection["matched_mean_iou"],
        "detection_ap50": detection["ap50"],
        "detection_ap75": detection["ap75"],
        "detection_map_50_95": detection["map_50_95"],
        "left_boundary_mae_px": detection["left_boundary_mae_px"],
        "right_boundary_mae_px": detection["right_boundary_mae_px"],
        "peak_count_mae": detection["peak_count_mae"],
        "seed_balanced_accuracy": seed["balanced_accuracy"],
        "seed_auroc": seed["auroc"],
        "seed_average_precision": seed["average_precision"],
        "seed_f1": seed["f1"],
        "best_detection_epoch": summary["best_detection_epoch"],
        "best_seed_epoch": summary["best_seed_epoch"],
        "train_seconds": summary["elapsed_seconds"],
        "peak_gpu_memory_bytes": summary["device"]["max_memory_allocated_bytes"],
    }


def main(args: argparse.Namespace) -> None:
    manifest_root = args.manifest_root.resolve()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    state_path = result_root / "PIPELINE_STATE.json"
    state: dict[str, Any] = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model": "PeakMultiTaskRCNN Naive concat",
            "task": "joint detection + Seed classification",
            "seed": 20260814,
            "epochs": 30,
            "batch_size": 16,
            "image_size": 480,
            "fp16": True,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "augmentation": False,
            "threshold_selection": "Val only",
        },
        "test_used_during_training": False,
    }
    if state_path.exists():
        previous = read_json(state_path)
        state["started_utc"] = previous.get("started_utc", state["started_utc"])
    write_json(state_path, state)

    rows: list[dict[str, Any]] = []
    main_root = result_root / "main_split"
    main_training = main_root / "training"
    summary = ensure_training(
        args=args,
        train_manifest=manifest_root / "main" / "train.jsonl",
        val_manifest=manifest_root / "main" / "val.jsonl",
        output_dir=main_training,
        log_path=main_root / "training_console.log",
    )
    main_metrics = ensure_evaluation(
        args=args,
        experiment_dir=main_training,
        val_manifest=manifest_root / "main" / "val.jsonl",
        target_manifest=manifest_root / "main" / "test.jsonl",
        target_name="Main Test",
        output_dir=main_root / "locked_test_evaluation",
        log_path=main_root / "locked_test_evaluation_console.log",
    )
    rows.append(metric_row("Main Test", main_metrics, summary))
    state.update({"main_split_complete": True, "last_update_utc": datetime.now(timezone.utc).isoformat()})
    write_json(state_path, state)

    lodo_rows: list[dict[str, Any]] = []
    for fold_dir in sorted((manifest_root / "lodo").glob("fold_*")):
        heldout_domain = fold_dir.name.split("_", 2)[2]
        fold_result = result_root / "lodo" / fold_dir.name
        training_dir = fold_result / "training"
        summary = ensure_training(
            args=args,
            train_manifest=fold_dir / "train.jsonl",
            val_manifest=fold_dir / "val.jsonl",
            output_dir=training_dir,
            log_path=fold_result / "training_console.log",
        )
        metrics = ensure_evaluation(
            args=args,
            experiment_dir=training_dir,
            val_manifest=fold_dir / "val.jsonl",
            target_manifest=fold_dir / "heldout.jsonl",
            target_name=heldout_domain,
            output_dir=fold_result / "locked_heldout_evaluation",
            log_path=fold_result / "locked_heldout_evaluation_console.log",
        )
        row = metric_row(heldout_domain, metrics, summary)
        rows.append(row)
        lodo_rows.append(row)
        write_csv(result_root / "multitask_concat_results_in_progress.csv", rows)
        state.update(
            {
                "lodo_folds_completed": len(lodo_rows),
                "current_or_last_domain": heldout_domain,
                "last_update_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(state_path, state)

    metric_names = [
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "detection_mean_iou",
        "detection_ap50",
        "detection_ap75",
        "detection_map_50_95",
        "left_boundary_mae_px",
        "right_boundary_mae_px",
        "peak_count_mae",
        "seed_balanced_accuracy",
        "seed_auroc",
        "seed_average_precision",
        "seed_f1",
    ]
    macro = {"scope": "LODO Macro average"}
    for name in metric_names:
        values = [float(row[name]) for row in lodo_rows]
        macro[name] = statistics.fmean(values)
        macro[f"{name}_sample_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    final_rows = [rows[0], *lodo_rows, macro]
    write_csv(result_root / "multitask_concat_main_and_lodo_results.csv", final_rows)
    write_json(result_root / "multitask_concat_main_and_lodo_results.json", final_rows)
    state.update(
        {
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "lodo_folds_completed": len(lodo_rows),
            "result_csv": str(result_root / "multitask_concat_main_and_lodo_results.csv"),
        }
    )
    write_json(state_path, state)
    print(
        json.dumps({"event": "pipeline_complete", "lodo_folds": len(lodo_rows)}),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814\detection_manifests_16attrs_20260815"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814"),
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\results\rtx4070_final_merged_20260814\multitask_concat_detection_seed_20260816_bs16_ep30"),
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\scripts\detection\run_rtx4070_multitask_fusion_experiment.py"),
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(r"D:\CODE\LipidBench\PeakTruthLab\scripts\detection\evaluate_rtx4070_multitask_locked_target.py"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
