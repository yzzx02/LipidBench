from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def assemble(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv).resolve()
    train_csv = Path(args.train_csv).resolve()
    val_csv = Path(args.val_csv).resolve()
    model_dir = Path(args.model_dir).resolve()
    eval_bundle_dir = Path(args.eval_bundle_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    train_pred = pd.read_csv(eval_bundle_dir / "train_predictions.csv")
    val_pred = pd.read_csv(eval_bundle_dir / "val_predictions.csv")

    history_json = model_dir / "history.json"
    if history_json.exists():
        history = json.loads(history_json.read_text(encoding="utf-8"))
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        (out_dir / "training_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    labeled = df[df["is_true_peak"].notna()].copy()
    labeled["is_true_peak"] = labeled["is_true_peak"].astype(int)

    class_balance = (
        labeled["is_true_peak"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )
    class_balance.to_csv(out_dir / "class_balance.csv", index=False)

    src_dist = (
        labeled.groupby(["source_file", "is_true_peak"])
        .size()
        .reset_index(name="count")
        .sort_values(["source_file", "is_true_peak"])
        .reset_index(drop=True)
    )
    src_dist.to_csv(out_dir / "full_labeled_source_distribution.csv", index=False)

    train_pred = train_pred.copy()
    val_pred = val_pred.copy()
    train_pred["split"] = "train"
    val_pred["split"] = "val"
    full_pred = pd.concat([train_pred, val_pred], ignore_index=True)
    full_pred.to_csv(out_dir / "full_labeled_plot_table.csv", index=False)

    summary = {
        "input_csv": str(input_csv),
        "model_dir": str(model_dir),
        "eval_bundle_dir": str(eval_bundle_dir),
        "rows_total_csv": int(len(df)),
        "rows_labeled": int(len(labeled)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "class_balance": {str(int(k)): int(v) for k, v in labeled["is_true_peak"].value_counts().to_dict().items()},
    }
    (out_dir / "bundle_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in [
        "roc_curve.csv",
        "pr_curve.csv",
        "confusion_matrix.json",
        "dataset_and_metrics_summary.json",
        "per_source_metrics.csv",
        "attribute_summary_by_label.csv",
        "val_error_cases.csv",
        "train_predictions.csv",
        "val_predictions.csv",
    ]:
        src = eval_bundle_dir / name
        if src.exists():
            (out_dir / name).write_bytes(src.read_bytes())

    attr_scaler = model_dir / "attr_scaler.json"
    best_model = model_dir / "best_model.pth"
    if attr_scaler.exists():
        (out_dir / "attr_scaler.json").write_bytes(attr_scaler.read_bytes())
    if best_model.exists():
        # Do not duplicate the model checkpoint into the paper bundle to keep it lightweight.
        pass

    print("done")
    print(f"out_dir: {out_dir}")
    print(f"rows_labeled: {len(labeled)}")
    print(f"train_rows: {len(train_df)}")
    print(f"val_rows: {len(val_df)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Assemble a plot-ready local bundle for binary fusion experiments")
    p.add_argument("--input-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--train-csv", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--eval-bundle-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    assemble(parse_args())
