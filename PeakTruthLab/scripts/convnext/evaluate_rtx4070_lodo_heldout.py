from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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


def main(args: argparse.Namespace) -> None:
    fold_dir = Path(args.fold_dir).resolve()
    training_dir = Path(args.training_dir).resolve()
    image_root = Path(args.image_root).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to repeat or overwrite held-out evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)

    split_audit = read_json(fold_dir / "split_audit.json")
    if split_audit.get("status") != "ok" or not split_audit.get("heldout_domain_absent_from_train_val"):
        raise RuntimeError("LODO split audit did not pass")
    heldout_csv = fold_dir / "heldout_test.csv"
    expected_hash = split_audit["sha256"]["heldout_test"]["csv"]
    observed_hash = sha256(heldout_csv)
    if observed_hash != expected_hash:
        raise RuntimeError(f"Held-out CSV hash mismatch: expected={expected_hash}, observed={observed_hash}")

    summary = read_json(training_dir / "summary.json")
    selection = read_json(training_dir / "selection_on_val.json")
    config = read_json(training_dir / "run_config.json")
    scaler_data = read_json(training_dir / "attr_scaler.json")
    if summary.get("status") != "complete" or summary.get("test_accessed") is not False:
        raise RuntimeError("Fold training is incomplete or its isolation flag failed")
    if config.get("model_mode") != "naive_concat" or config.get("seed") != 20260814:
        raise RuntimeError("Fold model mode or seed drifted from locked Concat protocol")
    if config.get("attribute_columns") != ATTRS or config.get("data_augmentation_enabled") is not False:
        raise RuntimeError("Fold attribute order or augmentation setting drifted")
    if Path(config["train_csv"]).resolve() != (fold_dir / "train.csv").resolve() or Path(config["val_csv"]).resolve() != (fold_dir / "val.csv").resolve():
        raise RuntimeError("Fold Train/Val paths do not match the frozen split")
    if config["train_csv_sha256"] != sha256(fold_dir / "train.csv") or config["val_csv_sha256"] != sha256(fold_dir / "val.csv"):
        raise RuntimeError("Fold Train/Val hash drift")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    scaler = AttrScaler(fill=scaler_data["fill"], mean=scaler_data["mean"], std=scaler_data["std"])
    dataset = RTXDataset(heldout_csv, image_root, scaler, int(config["input_size"]), load_image=True)
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
        model_mode="naive_concat",
    ).to(device)
    checkpoint_path = training_dir / "best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["epoch"]) != int(selection["best_epoch"]):
        raise RuntimeError("Fold best checkpoint epoch does not match Val selection")
    model.load_state_dict(checkpoint["model_state"])
    labels, probabilities, indices, loss = predict(model, loader, device, use_amp=bool(config["amp"]))
    threshold = float(selection["selected_threshold"])
    result = binary_metrics(labels, probabilities, threshold)
    write_csv(output / "heldout_predictions.csv", prediction_rows(dataset, indices, labels, probabilities, threshold))
    payload = {
        "status": "complete",
        "fold_id": split_audit["fold_id"],
        "heldout_domain": split_audit["heldout_domain"],
        "heldout_rows": len(dataset),
        "model_mode": "naive_concat",
        "seed": int(config["seed"]),
        "best_epoch_selected_on_fold_val": int(selection["best_epoch"]),
        "threshold_selected_on_fold_val": threshold,
        "heldout_loss": float(loss),
        "heldout_metrics": result,
        "heldout_csv_sha256": observed_hash,
        "checkpoint_sha256": sha256(checkpoint_path),
        "heldout_used_for_preprocessing": False,
        "heldout_used_for_checkpoint_selection": False,
        "heldout_used_for_threshold_selection": False,
        "heldout_evaluated_once": True,
        "environment": environment_payload(device),
    }
    write_json(output / "heldout_metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root", default=r"D:\CODE\LipidBench\PeakTruthLab\datasets\PeakTruthLab_final_merged_20260814")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
