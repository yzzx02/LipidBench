from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from train_convnext_fusion import AttrScaler, FusionModel, build_attr_scaler_from_train_csv, build_eval_transform


ATTRS = ["SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG", "SYM", "MOD", "EDGE"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass
class Sample:
    image_path: Path
    attrs: torch.Tensor
    label: int
    metadata: dict[str, object]


class RTXDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        scaler: AttrScaler,
        input_size: int,
        load_image: bool,
    ) -> None:
        frame = pd.read_csv(csv_path)
        required = {"seed_id", "image", "seed_label", *ATTRS}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")
        self.samples: list[Sample] = []
        self.transform = build_eval_transform(input_size)
        self.load_image = load_image
        for _, row in frame.iterrows():
            image_path = (image_root / str(row["image"])).resolve()
            if load_image and not image_path.is_file():
                raise FileNotFoundError(image_path)
            values = []
            for attr in ATTRS:
                value = pd.to_numeric(row[attr], errors="coerce")
                if pd.isna(value):
                    value = scaler.fill[attr]
                values.append((float(value) - scaler.mean[attr]) / scaler.std[attr])
            metadata = {
                key: row[key]
                for key in ("seed_id", "image_id", "image", "seed_label", "source_file", "domain_id", "old_new_batch", "difficulty_type")
                if key in row.index
            }
            self.samples.append(
                Sample(
                    image_path=image_path,
                    attrs=torch.tensor(values, dtype=torch.float32),
                    label=int(row["seed_label"]),
                    metadata=metadata,
                )
            )
        if not self.samples:
            raise RuntimeError(f"No samples in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if self.load_image:
            with Image.open(sample.image_path) as image:
                tensor = self.transform(image.convert("RGB"))
        else:
            tensor = torch.zeros((3, 1, 1), dtype=torch.float32)
        return tensor, sample.attrs, torch.tensor(sample.label, dtype=torch.long), index


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | int]:
    predictions = (probs >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(labels, probs))
    except ValueError:
        pr_auc = float("nan")
    return {
        "threshold": float(threshold),
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2.0),
        "accuracy": float((tp + tn) / max(len(labels), 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(labels: np.ndarray, probs: np.ndarray) -> tuple[float, list[dict]]:
    rows = [binary_metrics(labels, probs, index / 1000.0) for index in range(1001)]
    best = max(rows, key=lambda row: (row["balanced_accuracy"], row["f1"], -abs(row["threshold"] - 0.5)))
    return float(best["threshold"]), rows


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    labels_all: list[int] = []
    probs_all: list[float] = []
    indices_all: list[int] = []
    loss_sum = 0.0
    with torch.inference_mode():
        for images, attrs, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True).float()
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, attrs).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, labels_device)
            probabilities = torch.sigmoid(logits)
            loss_sum += float(loss.item()) * len(labels)
            labels_all.extend(labels.tolist())
            probs_all.extend(probabilities.detach().cpu().tolist())
            indices_all.extend(indices.tolist())
    return (
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(probs_all, dtype=np.float64),
        np.asarray(indices_all, dtype=np.int64),
        loss_sum / max(len(labels_all), 1),
    )


def prediction_rows(dataset: RTXDataset, indices: np.ndarray, labels: np.ndarray, probs: np.ndarray, threshold: float) -> list[dict]:
    rows = []
    for index, label, probability in zip(indices, labels, probs, strict=True):
        metadata = dict(dataset.samples[int(index)].metadata)
        metadata.update(
            {
                "label": int(label),
                "prob_true_peak": float(probability),
                "threshold": float(threshold),
                "predicted_label": int(probability >= threshold),
            }
        )
        rows.append(metadata)
    return rows


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    args: argparse.Namespace,
    attr_scaler: AttrScaler,
    epoch: int,
    best_auc: float,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "grad_scaler_state": scaler.state_dict(),
        "args": vars(args),
        "attr_columns": ATTRS,
        "attr_scaler": {"fill": attr_scaler.fill, "mean": attr_scaler.mean, "std": attr_scaler.std},
        "epoch": epoch,
        "best_metric_name": "val_auc",
        "best_metric_value": best_auc,
        "random_state": random.getstate(),
        "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
        "cuda_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def environment_payload(device: torch.device) -> dict:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        payload.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return payload


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be >=1")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA is required by protocol but unavailable")
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)

    train_csv = Path(args.train_csv).resolve()
    val_csv = Path(args.val_csv).resolve()
    image_root = Path(args.image_root).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    if any(save_dir.iterdir()) and not args.resume_checkpoint:
        raise FileExistsError(f"Refusing to start a fresh run in non-empty directory: {save_dir}")

    attr_scaler = build_attr_scaler_from_train_csv(train_csv, ATTRS)
    load_image = args.model_mode != "attr_only"
    train_ds = RTXDataset(train_csv, image_root, attr_scaler, args.input_size, load_image)
    val_ds = RTXDataset(val_csv, image_root, attr_scaler, args.input_size, load_image)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = FusionModel(
        attr_dim=16,
        out_dim=1,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
        vision_backbone=args.vision_backbone,
        lwga_depth=args.lwga_depth,
        lwga_groups=args.lwga_groups,
        lwga_mlp_ratio=args.lwga_mlp_ratio,
        lwga_dropout=args.lwga_dropout,
        model_mode=args.model_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_scaler = GradScaler(device.type, enabled=use_amp)
    start_epoch = 1
    best_auc = -float("inf")
    history: list[dict] = []
    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        amp_scaler.load_state_dict(checkpoint.get("grad_scaler_state", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_auc = float(checkpoint["best_metric_value"])
        history_path = save_dir / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))

    run_config = {
        **vars(args),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "image_root": str(image_root),
        "train_csv_sha256": sha256_file(train_csv),
        "val_csv_sha256": sha256_file(val_csv),
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "attribute_columns": ATTRS,
        "attribute_preprocessing": "median imputation + population z-score fitted only on Train",
        "data_augmentation_enabled": False,
        "checkpoint_selection": "maximum val_auc; earliest epoch wins exact ties",
        "threshold_selection": "Val sweep 0.000..1.000 step 0.001; maximize balanced accuracy, then F1, then closeness to 0.5",
        "test_accessed": False,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    atomic_write_json(save_dir / "run_config.json", run_config)
    atomic_write_json(save_dir / "environment.json", environment_payload(device))
    atomic_write_json(
        save_dir / "attr_scaler.json",
        {"fill": attr_scaler.fill, "mean": attr_scaler.mean, "std": attr_scaler.std, "attr_columns": ATTRS, "fit_split": "train"},
    )
    print(json.dumps({"device": str(device), "amp": use_amp, "mode": args.model_mode, "train": len(train_ds), "val": len(val_ds)}, ensure_ascii=False), flush=True)

    run_started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0
        for images, attrs, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            attrs = attrs.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images, attrs).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, labels_device)
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        labels, probs, indices, val_loss = predict(model, val_loader, device, use_amp)
        selected_threshold, sweep = threshold_sweep(labels, probs)
        fixed = binary_metrics(labels, probs, 0.5)
        selected = binary_metrics(labels, probs, selected_threshold)
        val_auc = float(selected["auc"])
        epoch_seconds = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_pr_auc": selected["pr_auc"],
            "val_selected_threshold": selected_threshold,
            "val_f1": selected["f1"],
            "val_precision": selected["precision"],
            "val_recall": selected["recall"],
            "val_specificity": selected["specificity"],
            "val_balanced_accuracy": selected["balanced_accuracy"],
            "val_f1_at_0_5": fixed["f1"],
            "epoch_seconds": epoch_seconds,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        }
        history.append(record)
        payload = checkpoint_payload(model, optimizer, amp_scaler, args, attr_scaler, epoch, max(best_auc, val_auc))
        torch.save(payload, save_dir / "last.pth")
        write_csv(save_dir / "val_predictions" / f"epoch_{epoch:03d}.csv", prediction_rows(val_ds, indices, labels, probs, selected_threshold))
        if val_auc > best_auc:
            best_auc = val_auc
            payload["best_metric_value"] = best_auc
            torch.save(payload, save_dir / "best_model.pth")
            write_csv(save_dir / "best_val_predictions.csv", prediction_rows(val_ds, indices, labels, probs, selected_threshold))
            write_csv(save_dir / "best_val_threshold_sweep.csv", sweep)
            atomic_write_json(
                save_dir / "selection_on_val.json",
                {
                    "best_epoch": epoch,
                    "best_val_auc": best_auc,
                    "selected_threshold": selected_threshold,
                    "selected_threshold_metrics": selected,
                    "checkpoint": "best_model.pth",
                    "test_accessed": False,
                },
            )
        atomic_write_json(save_dir / "history.json", history)
        write_csv(save_dir / "history.csv", history)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    best_checkpoint = torch.load(save_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    if args.verify_checkpoint_reload:
        labels, probs, _, _ = predict(model, val_loader, device, use_amp)
        reload_auc = float(roc_auc_score(labels, probs))
        if abs(reload_auc - best_auc) > 1e-10:
            raise RuntimeError(f"Checkpoint reload verification failed: {reload_auc} vs {best_auc}")
    total_seconds = time.perf_counter() - run_started
    summary = {
        "status": "complete",
        "model_mode": args.model_mode,
        "epochs_completed": args.epochs,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_val_auc": best_auc,
        "total_training_seconds_this_process": total_seconds,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "checkpoint_reload_verified": bool(args.verify_checkpoint_reload),
        "test_accessed": False,
    }
    atomic_write_json(save_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Frozen RTX4070 ConvNeXt fusion experiment runner")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--model-mode", required=True, choices=["attr_only", "image_only", "naive_concat", "gated_fusion"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=480)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--vision-backbone", default="convnext_tiny", choices=["convnext_tiny", "lwga_convnext"])
    parser.add_argument("--lwga-depth", type=int, default=2)
    parser.add_argument("--lwga-groups", type=int, default=8)
    parser.add_argument("--lwga-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--lwga-dropout", type=float, default=0.0)
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--verify-checkpoint-reload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
