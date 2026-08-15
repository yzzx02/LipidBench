from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from run_rtx4070_fusion_experiment import (
    ATTRS,
    RTXDataset,
    binary_metrics,
    environment_payload,
    predict,
    prediction_rows,
    seed_worker,
    write_csv,
)
from train_convnext_fusion import AttrScaler, FusionModel


RUNS = [
    ("Attribute-only", "A_attribute_only"),
    ("Image-only", "B_image_only"),
    ("Naive concat", "C_naive_concat"),
    ("Image-gated-Attribute", "D_gated_fusion"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_lock(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed Test lock line: {line}")
        values[parts[1].strip().lstrip("*")] = parts[0].lower()
    return values


def main(args: argparse.Namespace) -> None:
    test_csv = Path(args.test_csv).resolve()
    image_root = Path(args.image_root).resolve()
    run_root = Path(args.run_root).resolve()
    output = Path(args.output).resolve()
    override_path = Path(args.override).resolve()
    lock_path = Path(args.test_lock).resolve()

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Single-use main Test output already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    override = read_json(override_path)
    if override.get("selected_model") != "naive_concat" or override.get("test_accessed_before_override") is not False:
        raise RuntimeError("Concat override is missing or does not confirm pre-override Test isolation")

    locked = parse_lock(lock_path)
    expected_csv_hash = locked.get("test.csv")
    expected_jsonl_hash = locked.get("test.jsonl")
    observed_csv_hash = sha256(test_csv)
    if expected_csv_hash is None or observed_csv_hash != expected_csv_hash:
        raise RuntimeError(f"Locked Test CSV hash mismatch: expected={expected_csv_hash}, observed={observed_csv_hash}")

    test_frame = pd.read_csv(test_csv)
    if len(test_frame) != 1982 or test_frame["seed_id"].duplicated().any():
        raise RuntimeError("Locked Test row count or unique Seed ID check failed")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked main Test evaluation")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    summary_rows = []
    for model_name, directory in RUNS:
        source = run_root / directory
        config = read_json(source / "run_config.json")
        selection = read_json(source / "selection_on_val.json")
        scaler_values = read_json(source / "attr_scaler.json")
        if config.get("test_accessed") is not False or selection.get("test_accessed") is not False:
            raise RuntimeError(f"Pre-Test isolation flag failed for {model_name}")
        if config["train_csv_sha256"] != sha256(Path(config["train_csv"])) or config["val_csv_sha256"] != sha256(Path(config["val_csv"])):
            raise RuntimeError(f"Frozen Train/Val hash drift for {model_name}")
        if config["attribute_columns"] != ATTRS:
            raise RuntimeError(f"Attribute order mismatch for {model_name}")

        threshold = float(selection["selected_threshold"])
        scaler = AttrScaler(fill=scaler_values["fill"], mean=scaler_values["mean"], std=scaler_values["std"])
        load_image = config["model_mode"] != "attr_only"
        dataset = RTXDataset(test_csv, image_root, scaler, int(config["input_size"]), load_image)
        generator = torch.Generator()
        generator.manual_seed(int(config["seed"]))
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=True,
            persistent_workers=int(config["num_workers"]) > 0,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        model = FusionModel(
            attr_dim=16,
            out_dim=1,
            dropout=float(config["dropout"]),
            pretrained=False,
            vision_backbone=config["vision_backbone"],
            lwga_depth=int(config["lwga_depth"]),
            lwga_groups=int(config["lwga_groups"]),
            lwga_mlp_ratio=float(config["lwga_mlp_ratio"]),
            lwga_dropout=float(config["lwga_dropout"]),
            model_mode=config["model_mode"],
        ).to(device)
        checkpoint_path = source / "best_model.pth"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if int(checkpoint["epoch"]) != int(selection["best_epoch"]):
            raise RuntimeError(f"Best checkpoint epoch mismatch for {model_name}")
        model.load_state_dict(checkpoint["model_state"])
        labels, probabilities, indices, loss = predict(model, loader, device, use_amp=bool(config["amp"]))
        result = binary_metrics(labels, probabilities, threshold)
        model_output = output / directory
        model_output.mkdir(parents=True)
        write_csv(model_output / "test_predictions.csv", prediction_rows(dataset, indices, labels, probabilities, threshold))
        payload = {
            "model": model_name,
            "model_mode": config["model_mode"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "best_epoch_selected_on_val": int(selection["best_epoch"]),
            "threshold_selected_on_val": threshold,
            "test_loss": float(loss),
            "test_metrics": result,
            "test_rows": len(dataset),
            "test_csv_sha256": observed_csv_hash,
            "test_evaluated_once": True,
            "test_used_for_tuning": False,
        }
        write_json(model_output / "test_metrics.json", payload)
        summary_rows.append(
            {
                "model": model_name,
                "model_mode": config["model_mode"],
                "best_epoch_selected_on_val": selection["best_epoch"],
                "threshold_selected_on_val": threshold,
                "test_loss": loss,
                **result,
                "checkpoint_sha256": payload["checkpoint_sha256"],
            }
        )
        del model, checkpoint, loader, dataset
        torch.cuda.empty_cache()

    write_csv(output / "main_test_ablation_summary.csv", summary_rows)
    audit = {
        "status": "complete",
        "scope": "single evaluation of four locked main-ablation checkpoints",
        "selection_override": str(override_path),
        "selection_override_sha256": sha256(override_path),
        "test_csv": str(test_csv),
        "test_csv_sha256": observed_csv_hash,
        "test_jsonl_sha256_from_lock": expected_jsonl_hash,
        "models_evaluated": [name for name, _ in RUNS],
        "threshold_source": "locked validation selection_on_val.json for each model",
        "checkpoint_source": "locked best_model.pth for each model",
        "test_used_for_tuning": False,
        "environment": environment_payload(device),
    }
    write_json(output / "MAIN_TEST_EVALUATION_AUDIT.json", audit)
    print(json.dumps({"audit": audit, "summary": summary_rows}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814\splits\test.csv")
    parser.add_argument("--test-lock", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814\splits\LOCKED_TEST_MANIFEST.sha256")
    parser.add_argument("--image-root", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814")
    parser.add_argument("--run-root", default=r"D:\CODE\LipidBench\PeakTruthLab\results\rtx4070_final_merged_20260814\main_ablation\seed_20260814")
    parser.add_argument("--override", default=r"D:\CODE\LipidBench\PeakTruthLab\configs\rtx4070_concat_override_20260815.json")
    parser.add_argument("--output", default=r"D:\CODE\LipidBench\PeakTruthLab\results\rtx4070_final_merged_20260814\main_test_single_use_20260815")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
