from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PeakTruthLab.scripts.convnext.train_convnext_fusion import (
    AttrScaler,
    FusionModel,
    PeakFusionDataset,
    build_eval_transform,
)

MODEL_SPECS = [
    ("A", "image_only", "Image-only"),
    ("B", "attr_only", "Attr-only"),
    ("C", "naive_concat", "Naive Concat"),
    ("D", "gated_fusion", "Gated Fusion"),
]


def _load_attr_scaler(obj: dict | None) -> AttrScaler | None:
    if not obj:
        return None
    return AttrScaler(
        fill={str(k): float(v) for k, v in obj["fill"].items()},
        mean={str(k): float(v) for k, v in obj["mean"].items()},
        std={str(k): float(v) for k, v in obj["std"].items()},
    )


def _predict_binary(
    *,
    test_csv: Path,
    image_root: Path,
    image_col: str,
    label_col: str,
    attr_columns: list[str],
    attr_scaler: AttrScaler | None,
    model_mode: str,
    model: FusionModel,
    device: torch.device,
    input_size: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    load_image = str(model_mode).strip().lower() != "attr_only"
    ds = PeakFusionDataset(
        csv_path=test_csv,
        image_root=image_root,
        attr_columns=attr_columns,
        image_col=image_col,
        label_col=label_col,
        transform=(build_eval_transform(input_size=input_size) if load_image else None),
        class_to_id=None,
        attr_scaler=attr_scaler,
        load_image=load_image,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    src_df = pd.read_csv(test_csv).copy().reset_index(drop=True)
    probs: list[float] = []
    labels_out: list[int] = []

    model.eval()
    with torch.no_grad():
        for images, attrs, labels in loader:
            images = images.to(device)
            attrs = attrs.to(device)
            logits = model(images, attrs).squeeze(1)
            p = torch.sigmoid(logits)
            probs.extend(p.cpu().tolist())
            labels_out.extend(labels.cpu().tolist())

    out = src_df.iloc[: len(probs)].copy()
    out["label"] = labels_out
    out["prob_true_peak"] = probs
    out["pred_true_peak"] = (out["prob_true_peak"] >= 0.5).astype(int)
    return out


def _compute_binary_metrics(pred_df: pd.DataFrame) -> dict[str, float | int]:
    y_true = pred_df["label"].astype(int).to_numpy()
    y_prob = pred_df["prob_true_peak"].astype(float).to_numpy()
    y_pred = pred_df["pred_true_peak"].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")

    return {
        "test_auc": auc,
        "test_pr_auc": float(average_precision_score(y_true, y_prob)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _run_train_subprocess(args: argparse.Namespace, model_mode: str, save_dir: Path) -> None:
    train_script = PROJECT_ROOT / "PeakTruthLab" / "scripts" / "convnext" / "train_convnext_fusion.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--train-csv",
        str(args.train_csv),
        "--val-csv",
        str(args.val_csv),
        "--image-root",
        str(args.image_root),
        "--image-col",
        str(args.image_col),
        "--task-type",
        "binary",
        "--label-col",
        str(args.label_col),
        "--attr-columns",
        str(args.attr_columns),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--input-size",
        str(args.input_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--seed",
        str(args.seed),
        "--vision-backbone",
        str(args.vision_backbone),
        "--lwga-depth",
        str(args.lwga_depth),
        "--lwga-groups",
        str(args.lwga_groups),
        "--lwga-mlp-ratio",
        str(args.lwga_mlp_ratio),
        "--lwga-dropout",
        str(args.lwga_dropout),
        "--model-mode",
        str(model_mode),
        "--save-dir",
        str(save_dir),
    ]

    if args.enable_gaussian_blur:
        cmd.append("--enable-gaussian-blur")
    if args.no_pretrained:
        cmd.append("--no-pretrained")
    if args.no_standardize_attrs:
        cmd.append("--no-standardize-attrs")
    if args.amp:
        cmd.append("--amp")

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).resolve()
    models_dir = out_dir / "models"
    eval_dir = out_dir / "eval"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, float | int | str]] = []
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    for short_name, model_mode, display_name in MODEL_SPECS:
        model_save_dir = models_dir / f"model_{short_name}"
        model_save_dir.mkdir(parents=True, exist_ok=True)

        _run_train_subprocess(args, model_mode=model_mode, save_dir=model_save_dir)

        best_ckpt = model_save_dir / "best_model.pth"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Missing best checkpoint: {best_ckpt}")

        named_ckpt = ckpt_dir / f"best_model_{short_name}.pth"
        shutil.copy2(best_ckpt, named_ckpt)

        ckpt = torch.load(best_ckpt, map_location=device)
        ckpt_args = ckpt.get("args", {})
        attr_columns = list(ckpt.get("attr_columns") or [x.strip() for x in str(args.attr_columns).split(",") if x.strip()])
        attr_scaler = _load_attr_scaler(ckpt.get("attr_scaler"))

        model = FusionModel(
            attr_dim=len(attr_columns),
            out_dim=1,
            dropout=float(ckpt_args.get("dropout", args.dropout)),
            pretrained=False,
            vision_backbone=str(ckpt_args.get("vision_backbone", args.vision_backbone)),
            lwga_depth=int(ckpt_args.get("lwga_depth", args.lwga_depth)),
            lwga_groups=int(ckpt_args.get("lwga_groups", args.lwga_groups)),
            lwga_mlp_ratio=float(ckpt_args.get("lwga_mlp_ratio", args.lwga_mlp_ratio)),
            lwga_dropout=float(ckpt_args.get("lwga_dropout", args.lwga_dropout)),
            model_mode=str(ckpt_args.get("model_mode", model_mode)),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])

        pred_df = _predict_binary(
            test_csv=Path(args.test_csv),
            image_root=Path(args.image_root),
            image_col=str(ckpt_args.get("image_col", args.image_col)),
            label_col=str(ckpt_args.get("label_col", args.label_col)),
            attr_columns=attr_columns,
            attr_scaler=attr_scaler,
            model_mode=str(ckpt_args.get("model_mode", model_mode)),
            model=model,
            device=device,
            input_size=int(ckpt_args.get("input_size", args.input_size)),
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.eval_num_workers),
        )

        one_eval_dir = eval_dir / f"model_{short_name}"
        one_eval_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(one_eval_dir / "test_predictions.csv", index=False)

        metrics = _compute_binary_metrics(pred_df)
        (one_eval_dir / "confusion_matrix.json").write_text(
            json.dumps(
                {
                    "tn": int(metrics["tn"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "tp": int(metrics["tp"]),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (one_eval_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        row = {
            "model_id": short_name,
            "model_name": display_name,
            "model_mode": model_mode,
            "checkpoint": str(named_ckpt),
            **metrics,
        }
        summary_rows.append(row)
        print("[DONE]", row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        [
            "model_id",
            "model_name",
            "model_mode",
            "test_auc",
            "test_pr_auc",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_accuracy",
            "tn",
            "fp",
            "fn",
            "tp",
            "checkpoint",
        ]
    ]
    summary_df.to_csv(out_dir / "ablation_test_metrics.csv", index=False)
    (out_dir / "ablation_test_metrics.json").write_text(
        summary_df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("done")
    print(f"out_dir: {out_dir}")
    print(summary_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run 4-model grouped-split binary ablation pipeline")
    p.add_argument("--train-csv", type=str, default="PeakTruthLab/results/convnext_grouped_split_v1/train.csv")
    p.add_argument("--val-csv", type=str, default="PeakTruthLab/results/convnext_grouped_split_v1/val.csv")
    p.add_argument("--test-csv", type=str, default="PeakTruthLab/results/convnext_grouped_split_v1/test.csv")
    p.add_argument("--image-root", type=str, default="PeakTruthLab/datasets/eic_images_flat")
    p.add_argument("--image-col", type=str, default="image")
    p.add_argument("--label-col", type=str, default="is_true_peak")
    p.add_argument("--attr-columns", type=str, default="SNR,CV,GS,TPAS,H2B,ZZ,DZZ,PCC,SKEW,DENT,DM,ENT,JAG")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--eval-num-workers", type=int, default=0)
    p.add_argument("--input-size", type=int, default=480)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--enable-gaussian-blur", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-standardize-attrs", action="store_true")
    p.add_argument("--amp", action="store_true", help="Enable AMP in sub-training runs when CUDA is available")

    p.add_argument("--vision-backbone", type=str, default="convnext_tiny", choices=["convnext_tiny", "lwga_convnext"])
    p.add_argument("--lwga-depth", type=int, default=2)
    p.add_argument("--lwga-groups", type=int, default=8)
    p.add_argument("--lwga-mlp-ratio", type=float, default=2.0)
    p.add_argument("--lwga-dropout", type=float, default=0.0)

    p.add_argument("--out-dir", type=str, default="PeakTruthLab/results/ablation_grouped_v1")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be >= 1")
    if args.input_size < 64:
        raise ValueError("--input-size must be >= 64")
    return args


if __name__ == "__main__":
    run(parse_args())
