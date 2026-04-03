from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


MODEL_ORDER = ["A", "B", "C", "D"]
MODEL_LABELS = {
    "A": "Model A: Image-only",
    "B": "Model B: Attr-only",
    "C": "Model C: Naive Concat",
    "D": "Model D: Gated Fusion",
}
MODEL_COLORS = {
    "A": "#1f77b4",
    "B": "#d62728",
    "C": "#2ca02c",
    "D": "#ff7f0e",
}


def _save_current_figure(out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(out_base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")


def _read_prediction_table(eval_root: Path, model_id: str) -> pd.DataFrame:
    pred_csv = eval_root / f"model_{model_id}" / "test_predictions.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(f"missing prediction file: {pred_csv}")
    df = pd.read_csv(pred_csv)
    need = {"label", "prob_true_peak", "pred_true_peak"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise ValueError(f"{pred_csv} missing columns: {miss}")
    return df


def plot_roc(metrics_df: pd.DataFrame, eval_root: Path, out_dir: Path, dpi: int) -> None:
    plt.figure(figsize=(6.8, 5.4))
    for model_id in MODEL_ORDER:
        pred = _read_prediction_table(eval_root, model_id)
        fpr, tpr, _ = roc_curve(pred["label"].astype(int), pred["prob_true_peak"].astype(float))
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            lw=2.2,
            color=MODEL_COLORS[model_id],
            label=f"{MODEL_LABELS[model_id]} (AUC={roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", lw=1.2, label="Chance")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison on Grouped Test Set", fontsize=13)
    plt.legend(frameon=False, fontsize=10, loc="lower right")
    plt.grid(alpha=0.18, linestyle=":")
    plt.tight_layout()
    _save_current_figure(out_dir / "roc_curve_comparison", dpi=dpi)
    plt.close()


def plot_pr(metrics_df: pd.DataFrame, eval_root: Path, out_dir: Path, dpi: int) -> None:
    plt.figure(figsize=(6.8, 5.4))
    for model_id in MODEL_ORDER:
        pred = _read_prediction_table(eval_root, model_id)
        precision, recall, _ = precision_recall_curve(
            pred["label"].astype(int),
            pred["prob_true_peak"].astype(float),
        )
        pr_auc = auc(recall, precision)
        plt.plot(
            recall,
            precision,
            lw=2.2,
            color=MODEL_COLORS[model_id],
            label=f"{MODEL_LABELS[model_id]} (PR-AUC={pr_auc:.3f})",
        )

    prevalence = None
    try:
        ref_pred = _read_prediction_table(eval_root, "A")
        prevalence = float(ref_pred["label"].mean())
    except Exception:
        prevalence = None
    if prevalence is not None:
        plt.axhline(prevalence, linestyle="--", color="#7f7f7f", lw=1.2, label=f"Prevalence={prevalence:.3f}")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve Comparison on Grouped Test Set", fontsize=13)
    plt.legend(frameon=False, fontsize=10, loc="lower left")
    plt.grid(alpha=0.18, linestyle=":")
    plt.tight_layout()
    _save_current_figure(out_dir / "pr_curve_comparison", dpi=dpi)
    plt.close()


def plot_confusions(metrics_df: pd.DataFrame, eval_root: Path, out_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    axes = axes.flatten()

    for ax, model_id in zip(axes, MODEL_ORDER, strict=False):
        pred = _read_prediction_table(eval_root, model_id)
        y_true = pred["label"].astype(int)
        y_pred = pred["pred_true_peak"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_pct = cm / cm.sum()

        annot = [
            [f"{cm[i, j]}\n({cm_pct[i, j]*100:.1f}%)" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ]

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            square=True,
            linewidths=0.8,
            linecolor="white",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax,
            annot_kws={"fontsize": 10},
        )

        row = metrics_df.loc[metrics_df["model_id"] == model_id].iloc[0]
        ax.set_title(
            f"{MODEL_LABELS[model_id]}\nPrecision={row['test_precision']:.3f}, F1={row['test_f1']:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)

    fig.suptitle("Confusion Matrices on Grouped Test Set", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    _save_current_figure(out_dir / "confusion_matrix_heatmaps", dpi=dpi)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    metrics_csv = Path(args.metrics_csv).resolve()
    eval_root = Path(args.eval_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics csv not found: {metrics_csv}")
    if not eval_root.exists():
        raise FileNotFoundError(f"eval root not found: {eval_root}")

    metrics_df = pd.read_csv(metrics_csv)
    need = {"model_id", "test_auc", "test_pr_auc", "test_f1", "test_precision", "test_recall"}
    miss = sorted(need - set(metrics_df.columns))
    if miss:
        raise ValueError(f"metrics csv missing columns: {miss}")

    metrics_df["model_id"] = metrics_df["model_id"].astype(str)
    metrics_df = metrics_df.set_index("model_id").reindex(MODEL_ORDER).reset_index()

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_roc(metrics_df, eval_root, out_dir, dpi=int(args.dpi))
    plot_pr(metrics_df, eval_root, out_dir, dpi=int(args.dpi))
    plot_confusions(metrics_df, eval_root, out_dir, dpi=int(args.dpi))

    print("done")
    print(f"metrics_csv: {metrics_csv}")
    print(f"eval_root:   {eval_root}")
    print(f"out_dir:     {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Plot ROC/PR/confusion-matrix figures for grouped binary ablation results")
    p.add_argument("--metrics-csv", type=str, default="PeakTruthLab/results/ablation_grouped_v1/ablation_test_metrics.csv")
    p.add_argument("--eval-root", type=str, default="PeakTruthLab/results/ablation_grouped_v1/eval")
    p.add_argument("--out-dir", type=str, default="PeakTruthLab/results/ablation_grouped_v1/figures")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
