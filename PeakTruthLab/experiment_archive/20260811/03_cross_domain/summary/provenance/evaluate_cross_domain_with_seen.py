from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from evaluate_ablation_final import (  # noqa: E402
    detection_threshold_sweep,
    infer_manifest,
    seed_threshold_sweep,
    write_csv,
    write_json,
)
from domain_generalization.evaluate_cross_domain_seed import (  # noqa: E402
    evaluate_domain,
    flat_row,
    verify_hash,
)


CONDITIONS = (
    ("Seen-domain Control", "locked_original_test.jsonl", "Seen control", "seen_domain"),
    ("External A", "external_A.jsonl", "ST003127", "external_A"),
    ("External B", "external_B.jsonl", "ST003941", "external_B"),
    ("External C", "external_C.jsonl", "ST003514", "external_C"),
)


def extended_row(result: dict[str, Any], *, seed: int, study_id: str) -> dict[str, Any]:
    row = flat_row(result)
    seed_metrics = result["seed"]
    return {
        "seed": seed,
        "condition": row.pop("condition"),
        "study_id": study_id,
        "n_windows": row.pop("samples"),
        "seed_positive": int(seed_metrics["true_positive"] + seed_metrics["false_negative"]),
        "seed_negative": int(seed_metrics["true_negative"] + seed_metrics["false_positive"]),
        **row,
    }


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen":
        raise RuntimeError("Protocol lock is not frozen")
    if protocol.get("fusion_mode") != "naive_concat":
        raise RuntimeError("Only the frozen Naive concat model is allowed")
    expected = {
        "External A": "ST003127",
        "External B": "ST003941",
        "External C": "ST003514",
    }
    if protocol.get("external_domains") != expected:
        raise RuntimeError("External-domain identities changed")


def select_on_val(args: argparse.Namespace, protocol: dict[str, Any], device: torch.device) -> dict[str, Any]:
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
    write_csv(
        args.threshold_dir / "val_detection_threshold_sweep.csv",
        detection_sweep,
        list(detection_sweep[0]),
    )
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
    write_csv(
        args.threshold_dir / "val_seed_threshold_sweep.csv",
        seed_sweep,
        list(seed_sweep[0]),
    )
    del val_seed
    gc.collect()

    selection = {
        "status": "locked_before_seen_and_external_inference",
        "seen_test_used_during_checkpoint_or_threshold_selection": False,
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
    path = args.threshold_dir / "selection_before_seen_and_external.json"
    write_json(path, selection)
    if not path.is_file():
        raise RuntimeError("Selection lock was not persisted")
    return selection


def load_existing_selection(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("random_seed") != int(protocol["random_seed"]):
        raise RuntimeError("Existing selection lock belongs to a different seed")
    if selection.get("external_data_read_during_checkpoint_or_threshold_selection") is not False:
        raise RuntimeError("Existing selection lock does not prove isolation")
    return selection


def run(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    for filename in ("train.jsonl", "val.jsonl"):
        verify_hash(args.split_root / filename, protocol["split_sha256"][filename])

    training_summary = json.loads((args.experiment_dir / "summary.json").read_text(encoding="utf-8"))
    if training_summary.get("fusion_mode") != "naive_concat":
        raise RuntimeError("Training output is not Naive concat")
    if training_summary.get("epochs_completed") != 15:
        raise RuntimeError("Training output is not a complete 15-epoch run")
    if training_summary.get("test_manifest_used") is not False:
        raise RuntimeError("Training summary indicates test use")

    args.threshold_dir.mkdir(parents=True, exist_ok=True)
    args.seed_output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(int(protocol["random_seed"]))

    if args.reuse_selection_lock is None:
        selection = select_on_val(args, protocol, device)
        selection_path = args.threshold_dir / "selection_before_seen_and_external.json"
    else:
        selection_path = args.reuse_selection_lock
        selection = load_existing_selection(selection_path, protocol)

    requested = CONDITIONS if args.conditions == "all" else CONDITIONS[:1]
    for _, filename, _, _ in requested:
        verify_hash(args.split_root / filename, protocol["split_sha256"][filename])

    detection_threshold = float(selection["detection"]["selected_score_threshold"])
    seed_threshold = float(selection["seed"]["selected_probability_threshold"])
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for label, filename, study_id, directory_name in requested:
        print(f"[TEST START] {label} {study_id}", flush=True)
        result = evaluate_domain(
            label=label,
            manifest=args.split_root / filename,
            experiment_dir=args.experiment_dir,
            image_root=args.image_root,
            output_dir=args.seed_output_dir,
            device=device,
            batch_size=args.batch_size,
            detection_threshold=detection_threshold,
            seed_threshold=seed_threshold,
            directory_name=directory_name,
        )
        results.append(result)
        rows.append(extended_row(result, seed=int(protocol["random_seed"]), study_id=study_id))
        print(f"[TEST DONE] {label}", flush=True)

    if args.conditions == "all":
        external_rows = rows[1:]
        metric_fields = [
            key for key in external_rows[0]
            if key not in {"seed", "condition", "study_id", "n_windows", "seed_positive", "seed_negative"}
        ]
        macro = {
            "seed": int(protocol["random_seed"]),
            "condition": "External Macro",
            "study_id": "A/B/C unweighted macro",
            "n_windows": sum(int(row["n_windows"]) for row in external_rows),
            "seed_positive": sum(int(row["seed_positive"]) for row in external_rows),
            "seed_negative": sum(int(row["seed_negative"]) for row in external_rows),
            **{key: fmean(float(row[key]) for row in external_rows) for key in metric_fields},
        }
        rows.append(macro)

    summary_dir = args.seed_output_dir / "summary"
    write_csv(summary_dir / "condition_metrics.csv", rows, list(rows[0]))
    write_json(summary_dir / "evaluation_complete.json", {
        "status": "completed",
        "random_seed": int(protocol["random_seed"]),
        "selection_lock": str(selection_path.resolve()),
        "conditions": [result["condition"] for result in results],
        "seen_and_external_used_only_after_val_selection": True,
        "detection_threshold": detection_threshold,
        "seed_threshold": seed_threshold,
    })
    args.provenance_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), args.provenance_dir)
    shutil.copy2(WORK_ROOT / "evaluate_ablation_final.py", args.provenance_dir)
    shutil.copy2(
        WORK_ROOT / "domain_generalization" / "evaluate_cross_domain_seed.py",
        args.provenance_dir,
    )
    print(json.dumps({"status": "completed", "rows": rows}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--seed-output-dir", type=Path, required=True)
    parser.add_argument("--threshold-dir", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--reuse-selection-lock", type=Path)
    parser.add_argument("--conditions", choices=("seen", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
