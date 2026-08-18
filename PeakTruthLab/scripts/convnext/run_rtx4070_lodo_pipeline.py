from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def main(args: argparse.Namespace) -> None:
    split_root = Path(args.split_root).resolve()
    result_root = Path(args.result_root).resolve()
    image_root = Path(args.image_root).resolve()
    train_script = Path(args.train_script).resolve()
    eval_script = Path(args.eval_script).resolve()
    manifest_path = split_root / "LODO_FOLD_MANIFEST.csv"
    global_audit = read_json(split_root / "LODO_SPLIT_GLOBAL_AUDIT.json")
    if global_audit.get("status") != "ok" or global_audit.get("folds") != 11:
        raise RuntimeError("Global LODO split audit failed")
    result_root.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        folds = list(csv.DictReader(handle))
    if len(folds) != 11:
        raise RuntimeError(f"Expected 11 folds, found {len(folds)}")

    state_path = result_root / "PIPELINE_STATE.json"
    state = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "seed": 20260814,
        "model_mode": "naive_concat",
        "folds_total": len(folds),
        "folds": [],
    }
    if state_path.exists():
        previous = read_json(state_path)
        state["started_utc"] = previous.get("started_utc", state["started_utc"])
    write_json(state_path, state)

    for fold in folds:
        fold_id = fold["fold_id"]
        heldout_domain = fold["heldout_domain"]
        fold_dir = split_root / fold_id
        fold_result = result_root / fold_id
        training_dir = fold_result / "training"
        evaluation_dir = fold_result / "heldout_evaluation"
        fold_result.mkdir(parents=True, exist_ok=True)
        metrics_path = evaluation_dir / "heldout_metrics.json"

        if metrics_path.exists():
            metrics = read_json(metrics_path)
            if metrics.get("status") != "complete" or metrics.get("heldout_domain") != heldout_domain:
                raise RuntimeError(f"Existing held-out result failed validation: {metrics_path}")
            print(json.dumps({"event": "fold_already_complete", "fold_id": fold_id, "heldout_domain": heldout_domain}), flush=True)
        else:
            summary_path = training_dir / "summary.json"
            if summary_path.exists():
                summary = read_json(summary_path)
                if summary.get("status") != "complete" or summary.get("model_mode") != "naive_concat":
                    raise RuntimeError(f"Existing fold training summary failed validation: {summary_path}")
                print(json.dumps({"event": "training_already_complete", "fold_id": fold_id}), flush=True)
            else:
                if training_dir.exists() and any(training_dir.iterdir()):
                    raise RuntimeError(
                        f"Incomplete non-empty training directory requires explicit audit before resume: {training_dir}"
                    )
                train_command = [
                    sys.executable,
                    str(train_script),
                    "--train-csv",
                    str(fold_dir / "train.csv"),
                    "--val-csv",
                    str(fold_dir / "val.csv"),
                    "--image-root",
                    str(image_root),
                    "--save-dir",
                    str(training_dir),
                    "--model-mode",
                    "naive_concat",
                    "--epochs",
                    "30",
                    "--batch-size",
                    "16",
                    "--num-workers",
                    "4",
                    "--input-size",
                    "480",
                    "--lr",
                    "0.0001",
                    "--weight-decay",
                    "0.0001",
                    "--dropout",
                    "0.2",
                    "--seed",
                    "20260814",
                    "--amp",
                    "--require-cuda",
                    "--verify-checkpoint-reload",
                ]
                print(json.dumps({"event": "training_start", "fold_id": fold_id, "heldout_domain": heldout_domain}), flush=True)
                run_and_log(train_command, fold_result / "training_console.log")

            if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
                raise RuntimeError(f"Incomplete non-empty held-out evaluation directory: {evaluation_dir}")
            eval_command = [
                sys.executable,
                str(eval_script),
                "--fold-dir",
                str(fold_dir),
                "--training-dir",
                str(training_dir),
                "--output",
                str(evaluation_dir),
                "--image-root",
                str(image_root),
            ]
            print(json.dumps({"event": "heldout_evaluation_start", "fold_id": fold_id, "heldout_domain": heldout_domain}), flush=True)
            run_and_log(eval_command, fold_result / "heldout_evaluation_console.log")
            metrics = read_json(metrics_path)

        completed = []
        for candidate in folds:
            candidate_path = result_root / candidate["fold_id"] / "heldout_evaluation" / "heldout_metrics.json"
            if candidate_path.exists() and read_json(candidate_path).get("status") == "complete":
                completed.append(candidate["fold_id"])
        state.update(
            {
                "status": "running" if len(completed) < len(folds) else "complete",
                "last_update_utc": datetime.now(timezone.utc).isoformat(),
                "folds_completed": len(completed),
                "completed_fold_ids": completed,
                "current_or_last_fold": fold_id,
            }
        )
        write_json(state_path, state)
        print(
            json.dumps(
                {
                    "event": "fold_complete",
                    "fold_id": fold_id,
                    "heldout_domain": heldout_domain,
                    "folds_completed": len(completed),
                    "heldout_auc": metrics["heldout_metrics"]["auc"],
                    "heldout_f1": metrics["heldout_metrics"]["f1"],
                }
            ),
            flush=True,
        )

    state["status"] = "complete"
    state["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(state_path, state)
    print(json.dumps({"event": "pipeline_complete", "folds_completed": len(folds)}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814\lodo_seed_20260814")
    parser.add_argument("--result-root", default=r"D:\CODE\LipidBench\PeakTruthLab\results\rtx4070_final_merged_20260814\lodo_concat_seed_20260814")
    parser.add_argument("--image-root", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814")
    parser.add_argument("--train-script", default=r"D:\CODE\LipidBench\PeakTruthLab\scripts\convnext\run_rtx4070_fusion_experiment.py")
    parser.add_argument("--eval-script", default=r"D:\CODE\LipidBench\PeakTruthLab\scripts\convnext\evaluate_rtx4070_lodo_heldout.py")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
