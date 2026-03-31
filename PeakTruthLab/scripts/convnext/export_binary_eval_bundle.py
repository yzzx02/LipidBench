from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from PeakTruthLab.scripts.convnext.train_convnext_fusion import (
    AttrScaler,
    FusionModel,
    PeakFusionDataset,
    build_eval_transform,
)


def _load_attr_scaler(obj: dict | None) -> AttrScaler | None:
    if not obj:
        return None
    return AttrScaler(
        fill={str(k): float(v) for k, v in obj["fill"].items()},
        mean={str(k): float(v) for k, v in obj["mean"].items()},
        std={str(k): float(v) for k, v in obj["std"].items()},
    )


def _predict_dataset(
    *,
    df_csv: Path,
    image_root: Path,
    image_col: str,
    label_col: str,
    attr_columns: list[str],
    attr_scaler: AttrScaler | None,
    model: FusionModel,
    device: torch.device,
    input_size: int,
    batch_size: int,
    num_workers: int,
    model_mode: str,
) -> pd.DataFrame:
    load_image = str(model_mode).strip().lower() != "attr_only"
    ds = PeakFusionDataset(
        csv_path=df_csv,
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

    src_df = pd.read_csv(df_csv).copy().reset_index(drop=True)
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


def export_bundle(args: argparse.Namespace) -> None:
    ckpt_path = Path(args.checkpoint).resolve()
    train_csv = Path(args.train_csv).resolve()
    val_csv = Path(args.val_csv).resolve()
    image_root = Path(args.image_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    attr_columns = list(ckpt.get("attr_columns") or [x.strip() for x in str(args.attr_columns).split(",") if x.strip()])
    attr_scaler = _load_attr_scaler(ckpt.get("attr_scaler"))

    model = FusionModel(
        attr_dim=len(attr_columns),
        out_dim=1,
        dropout=float(ckpt_args.get("dropout", 0.2)),
        pretrained=False,
        vision_backbone=str(ckpt_args.get("vision_backbone", "convnext_tiny")),
        lwga_depth=int(ckpt_args.get("lwga_depth", 2)),
        lwga_groups=int(ckpt_args.get("lwga_groups", 8)),
        lwga_mlp_ratio=float(ckpt_args.get("lwga_mlp_ratio", 2.0)),
        lwga_dropout=float(ckpt_args.get("lwga_dropout", 0.0)),
        model_mode=str(ckpt_args.get("model_mode", "gated_fusion")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    image_col = str(ckpt_args.get("image_col", args.image_col))
    label_col = str(ckpt_args.get("label_col", args.label_col))
    input_size = int(ckpt_args.get("input_size", args.input_size))

    val_pred = _predict_dataset(
        df_csv=val_csv,
        image_root=image_root,
        image_col=image_col,
        label_col=label_col,
        attr_columns=attr_columns,
        attr_scaler=attr_scaler,
        model=model,
        device=device,
        input_size=input_size,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        model_mode=str(ckpt_args.get("model_mode", "gated_fusion")),
    )
    train_pred = _predict_dataset(
        df_csv=train_csv,
        image_root=image_root,
        image_col=image_col,
        label_col=label_col,
        attr_columns=attr_columns,
        attr_scaler=attr_scaler,
        model=model,
        device=device,
        input_size=input_size,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        model_mode=str(ckpt_args.get("model_mode", "gated_fusion")),
    )

    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)
    train_pred.to_csv(out_dir / "train_predictions.csv", index=False)

    y_true = val_pred["label"].astype(int).to_numpy()
    y_prob = val_pred["prob_true_peak"].astype(float).to_numpy()
    y_pred = val_pred["pred_true_peak"].astype(int).to_numpy()

    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(out_dir / "roc_curve.csv", index=False)

    precision, recall, pr_thr = precision_recall_curve(y_true, y_prob)
    pr_df = pd.DataFrame({"precision": precision, "recall": recall})
    pr_df["threshold"] = np.nan
    if len(pr_thr) > 0:
        pr_df.loc[: len(pr_thr) - 1, "threshold"] = pr_thr
    pr_df.to_csv(out_dir / "pr_curve.csv", index=False)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    conf = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    with (out_dir / "confusion_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2)

    summary = {
        "checkpoint": str(ckpt_path),
        "device_used_for_eval": str(device),
        "val_auc": float(roc_auc_score(y_true, y_prob)),
        "val_pr_auc": float(average_precision_score(y_true, y_prob)),
        "val_f1": float(f1_score(y_true, y_pred)),
        "val_acc": float(accuracy_score(y_true, y_pred)),
        "train_rows": int(len(train_pred)),
        "val_rows": int(len(val_pred)),
        "train_label_counts": {str(k): int(v) for k, v in train_pred["label"].value_counts().to_dict().items()},
        "val_label_counts": {str(k): int(v) for k, v in val_pred["label"].value_counts().to_dict().items()},
        "attr_columns": attr_columns,
    }
    with (out_dir / "dataset_and_metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    per_source_rows: list[dict[str, float | int | str]] = []
    for src, grp in val_pred.groupby("source_file"):
        labels = grp["label"].astype(int).to_numpy()
        probs = grp["prob_true_peak"].astype(float).to_numpy()
        preds = grp["pred_true_peak"].astype(int).to_numpy()
        row = {
            "source_file": str(src),
            "n": int(len(grp)),
            "pos": int((labels == 1).sum()),
            "neg": int((labels == 0).sum()),
            "acc": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        }
        if len(np.unique(labels)) == 2:
            row["auc"] = float(roc_auc_score(labels, probs))
            row["pr_auc"] = float(average_precision_score(labels, probs))
        else:
            row["auc"] = np.nan
            row["pr_auc"] = np.nan
        per_source_rows.append(row)
    pd.DataFrame(per_source_rows).sort_values("source_file").to_csv(out_dir / "per_source_metrics.csv", index=False)

    full_labeled = pd.concat([train_pred, val_pred], ignore_index=True)
    attr_rows: list[dict[str, float | int | str]] = []
    for col in attr_columns:
        for label_val, grp in full_labeled.groupby("label"):
            s = pd.to_numeric(grp[col], errors="coerce")
            attr_rows.append(
                {
                    "attr": col,
                    "label": int(label_val),
                    "n": int(s.notna().sum()),
                    "mean": float(s.mean()) if s.notna().any() else np.nan,
                    "std": float(s.std(ddof=0)) if s.notna().any() else np.nan,
                    "median": float(s.median()) if s.notna().any() else np.nan,
                }
            )
    pd.DataFrame(attr_rows).to_csv(out_dir / "attribute_summary_by_label.csv", index=False)

    errors = val_pred[val_pred["label"] != val_pred["pred_true_peak"]].copy()
    errors["error_type"] = np.where(errors["label"] == 1, "FN", "FP")
    errors.to_csv(out_dir / "val_error_cases.csv", index=False)

    print("done")
    print(f"checkpoint: {ckpt_path}")
    print(f"out_dir:    {out_dir}")
    print(f"summary:    {summary}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export plot-ready evaluation bundle for binary fusion model")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--image-root", type=str, required=True)
    p.add_argument("--image-col", type=str, default="image")
    p.add_argument("--label-col", type=str, default="is_true_peak")
    p.add_argument("--attr-columns", type=str, default="SNR,CV,GS,TPAS,H2B,ZZ,DZZ,PCC,SKEW,DENT,DM,ENT,JAG")
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    export_bundle(parse_args())
