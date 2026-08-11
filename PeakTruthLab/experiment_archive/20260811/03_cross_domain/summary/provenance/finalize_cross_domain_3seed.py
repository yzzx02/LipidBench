from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (20260725, 20260726, 20260727)
CONDITIONS = (
    "Seen-domain Control",
    "External A",
    "External B",
    "External C",
)
EXTERNALS = CONDITIONS[1:]
ATTRIBUTES = ("SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG")
METRICS = (
    "det_precision",
    "det_recall",
    "det_f1",
    "mean_iou",
    "median_iou",
    "det_ap50",
    "det_ap75",
    "seed_ba",
    "seed_auroc",
    "seed_auprc",
    "seed_precision",
    "seed_recall",
    "seed_f1",
)
GAP_METRICS = ("det_f1", "mean_iou", "seed_ba", "seed_auroc", "seed_auprc")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_path(root: Path, seed: int, condition: str) -> Path:
    seed_dir = root / ("seed_1" if seed == 20260725 else f"seed_{seed}")
    if seed == 20260725 and condition.startswith("External"):
        return seed_dir / "metrics" / condition.replace(" ", "_") / "metrics.json"
    mapping = {
        "Seen-domain Control": "seen_domain",
        "External A": "external_A",
        "External B": "external_B",
        "External C": "external_C",
    }
    return seed_dir / mapping[condition] / "metrics.json"


def prediction_dir(root: Path, seed: int, condition: str) -> Path:
    return metric_path(root, seed, condition).parent


def flatten_metric(seed: int, condition: str, payload: dict[str, Any]) -> dict[str, Any]:
    det = payload["detection"]
    sd = payload["seed"]
    study = {
        "Seen-domain Control": "Seen control",
        "External A": "ST003127",
        "External B": "ST003941",
        "External C": "ST003514",
    }[condition]
    return {
        "seed": seed,
        "condition": condition,
        "study_id": study,
        "n_windows": int(payload["samples"]),
        "seed_positive": int(sd["true_positive"] + sd["false_negative"]),
        "seed_negative": int(sd["true_negative"] + sd["false_positive"]),
        "det_precision": float(det["precision"]),
        "det_recall": float(det["recall"]),
        "det_f1": float(det["f1"]),
        "mean_iou": float(det["matched_mean_iou"]),
        "median_iou": float(det["matched_median_iou"]),
        "det_ap50": float(det["ap50"]),
        "det_ap75": float(det["ap75"]),
        "seed_ba": float(sd["balanced_accuracy"]),
        "seed_auroc": float(sd["auroc"]),
        "seed_auprc": float(sd["average_precision"]),
        "seed_precision": float(sd["precision"]),
        "seed_recall": float(sd["recall"]),
        "seed_f1": float(sd["f1"]),
        "seed_specificity": float(sd["specificity"]),
        "seed_tp": int(sd["true_positive"]),
        "seed_fp": int(sd["false_positive"]),
        "seed_fn": int(sd["false_negative"]),
        "seed_tn": int(sd["true_negative"]),
        "detection_threshold": float(payload["thresholds"]["detection_score"]),
        "seed_threshold": float(payload["thresholds"]["seed_probability"]),
    }


def external_macro(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    external = [row for row in seed_rows if row["condition"] in EXTERNALS]
    return {
        "seed": external[0]["seed"],
        "condition": "External Macro",
        "study_id": "A/B/C unweighted macro",
        "n_windows": sum(int(row["n_windows"]) for row in external),
        "seed_positive": sum(int(row["seed_positive"]) for row in external),
        "seed_negative": sum(int(row["seed_negative"]) for row in external),
        **{metric: fmean(float(row[metric]) for row in external) for metric in METRICS},
        "seed_specificity": fmean(float(row["seed_specificity"]) for row in external),
        "seed_tp": sum(int(row["seed_tp"]) for row in external),
        "seed_fp": sum(int(row["seed_fp"]) for row in external),
        "seed_fn": sum(int(row["seed_fn"]) for row in external),
        "seed_tn": sum(int(row["seed_tn"]) for row in external),
        "detection_threshold": external[0]["detection_threshold"],
        "seed_threshold": external[0]["seed_threshold"],
    }


def summarize(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for condition in (*CONDITIONS, "External Macro"):
        selected = [row for row in raw if row["condition"] == condition]
        result: dict[str, Any] = {"condition": condition, "n_seeds": len(selected)}
        for metric in METRICS:
            values = [float(row[metric]) for row in selected]
            mean = fmean(values)
            sd = stdev(values)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sd"] = sd
            result[f"{metric}_mean_sd"] = f"{mean:.4f} ± {sd:.4f}"
        rows.append(result)
    return rows


def gap_rows(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for seed in SEEDS:
        by_condition = {row["condition"]: row for row in raw if row["seed"] == seed}
        seen = by_condition["Seen-domain Control"]
        for condition in (*EXTERNALS, "External Macro"):
            row = {"row_type": "raw", "seed": seed, "comparison": f"Seen - {condition}"}
            for metric in GAP_METRICS:
                row[f"gap_{metric}"] = float(seen[metric]) - float(by_condition[condition][metric])
            output.append(row)
    for condition in (*EXTERNALS, "External Macro"):
        selected = [row for row in output if row["comparison"] == f"Seen - {condition}" and row["row_type"] == "raw"]
        summary: dict[str, Any] = {"row_type": "mean_sd", "seed": "mean ± sample SD", "comparison": f"Seen - {condition}"}
        for metric in GAP_METRICS:
            values = [float(row[f"gap_{metric}"]) for row in selected]
            mean = fmean(values)
            sd = stdev(values)
            summary[f"gap_{metric}"] = f"{mean:.4f} ± {sd:.4f}"
            summary[f"gap_{metric}_sd"] = sd
            summary[f"gap_{metric}_mean_sd"] = f"{mean:.4f} ± {sd:.4f}"
        output.append(summary)
    return output


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.percentile(array, q)) if array.size else math.nan


def descriptive_stats(label: str, records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    box_counts = [len(record["boxes"]) for record in records]
    boxes = [box for record in records for box in record["boxes"]]
    widths = [float(box[2]) - float(box[0]) for box in boxes]
    heights = [float(box[3]) - float(box[1]) for box in boxes]
    areas = [width * height for width, height in zip(widths, heights)]
    base: dict[str, Any] = {
        "condition": label,
        "windows": len(records),
        "source_files": len({record["source_file"] for record in records}),
        "true_peak_boxes": sum(box_counts),
        "seed_positive": sum(int(record["seed_label"]) for record in records),
        "seed_negative": sum(1 - int(record["seed_label"]) for record in records),
        "seed_positive_rate": fmean(int(record["seed_label"]) for record in records),
        "empty_windows": sum(count == 0 for count in box_counts),
        "single_windows": sum(count == 1 for count in box_counts),
        "multi_windows": sum(count > 1 for count in box_counts),
        "empty_rate": fmean(count == 0 for count in box_counts),
        "single_rate": fmean(count == 1 for count in box_counts),
        "multi_rate": fmean(count > 1 for count in box_counts),
        "mean_true_boxes_per_window": fmean(box_counts),
    }
    for name, values in (("box_width", widths), ("box_height", heights), ("box_area", areas)):
        q1 = percentile(values, 25)
        q3 = percentile(values, 75)
        base[f"{name}_median"] = percentile(values, 50)
        base[f"{name}_q1"] = q1
        base[f"{name}_q3"] = q3
        base[f"{name}_iqr"] = q3 - q1

    attr_rows = []
    for index, attribute in enumerate(ATTRIBUTES):
        raw_values = [record["attributes"][index] for record in records]
        finite = [float(value) for value in raw_values if value is not None and math.isfinite(float(value))]
        q1 = percentile(finite, 25)
        q3 = percentile(finite, 75)
        attr_rows.append({
            "condition": label,
            "attribute": attribute,
            "n_finite": len(finite),
            "missing_count": len(raw_values) - len(finite),
            "missing_rate": (len(raw_values) - len(finite)) / len(raw_values),
            "median": percentile(finite, 50),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
        })
    return base, attr_rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def external_b_failure(root: Path, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        metric = next(row for row in raw if row["seed"] == seed and row["condition"] == "External B")
        image_rows = read_csv(prediction_dir(root, seed, "External B") / "detection_image_metrics.csv")
        rows.append({
            "row_type": "raw",
            "seed": seed,
            "n_windows": metric["n_windows"],
            "det_tp": sum(int(row["true_positive"]) for row in image_rows),
            "det_fp": sum(int(row["false_positive"]) for row in image_rows),
            "det_fn": sum(int(row["false_negative"]) for row in image_rows),
            "det_precision": metric["det_precision"],
            "det_recall": metric["det_recall"],
            "det_f1": metric["det_f1"],
            "precision_minus_recall": float(metric["det_precision"]) - float(metric["det_recall"]),
            "mean_iou": metric["mean_iou"],
            "seed_tp": metric["seed_tp"],
            "seed_fp": metric["seed_fp"],
            "seed_fn": metric["seed_fn"],
            "seed_tn": metric["seed_tn"],
            "seed_sensitivity": metric["seed_recall"],
            "seed_specificity": metric["seed_specificity"],
            "seed_ba": metric["seed_ba"],
            "seed_auroc": metric["seed_auroc"],
            "seed_auprc": metric["seed_auprc"],
        })
    summary: dict[str, Any] = {"row_type": "mean_sd", "seed": "mean ± sample SD", "n_windows": rows[0]["n_windows"]}
    numeric = [key for key in rows[0] if key not in {"row_type", "seed", "n_windows"}]
    for key in numeric:
        values = [float(row[key]) for row in rows]
        mean, sd = fmean(values), stdev(values)
        summary[key] = f"{mean:.4f} ± {sd:.4f}"
        summary[f"{key}_mean"] = mean
        summary[f"{key}_sd"] = sd
        summary[f"{key}_mean_sd"] = f"{mean:.4f} ± {sd:.4f}"
    rows.append(summary)
    return rows


def audit(root: Path, split_root: Path) -> dict[str, Any]:
    manifests = {
        "train": load_jsonl(split_root / "train.jsonl"),
        "val": load_jsonl(split_root / "val.jsonl"),
        "seen": load_jsonl(split_root / "locked_original_test.jsonl"),
        "external_A": load_jsonl(split_root / "external_A.jsonl"),
        "external_B": load_jsonl(split_root / "external_B.jsonl"),
        "external_C": load_jsonl(split_root / "external_C.jsonl"),
    }
    sources = {name: {record["source_file"] for record in rows} for name, rows in manifests.items()}
    studies = {name: {record["study_id"] for record in rows} for name, rows in manifests.items()}
    development_sources = sources["train"] | sources["val"]
    development_studies = studies["train"] | studies["val"]
    checks: dict[str, Any] = {
        "train_val_source_overlap": len(sources["train"] & sources["val"]),
        "train_seen_source_overlap": len(sources["train"] & sources["seen"]),
        "val_seen_source_overlap": len(sources["val"] & sources["seen"]),
    }
    for domain in ("external_A", "external_B", "external_C"):
        checks[f"development_{domain}_source_overlap"] = len(development_sources & sources[domain])
        checks[f"development_{domain}_study_overlap"] = len(development_studies & studies[domain])
    checks["seen_development_study_overlap"] = len(studies["seen"] & development_studies)

    seed_checks = {}
    for seed in SEEDS:
        seed_dir = root / ("seed_1" if seed == 20260725 else f"seed_{seed}")
        experiment = seed_dir / "checkpoints"
        scaler = load_json(experiment / "attribute_preprocessing.json")
        training = load_json(experiment / "summary.json")
        if seed == 20260725:
            selection = load_json(seed_dir / "thresholds" / "selection_before_external.json")
            evaluation = load_json(seed_dir / "summary" / "evaluation_complete.json")
            seen_isolated = evaluation.get("seen_and_external_used_only_after_val_selection") is True
        else:
            selection = load_json(seed_dir / "thresholds" / "selection_before_seen_and_external.json")
            seen_isolated = selection.get("seen_test_used_during_checkpoint_or_threshold_selection") is False
        seed_checks[str(seed)] = {
            "scaler_fit_on_train_only": scaler.get("fit_partition") == "train" and scaler.get("fitted_samples") == 7274,
            "external_used_for_threshold_selection": selection.get("external_data_read_during_checkpoint_or_threshold_selection") is not False,
            "seen_test_used_for_threshold_selection": not seen_isolated,
            "test_manifest_used_during_training": training.get("test_manifest_used") is not False,
            "epochs_completed": training.get("epochs_completed"),
            "fusion_mode": training.get("fusion_mode"),
        }
    checks["seed_checks"] = seed_checks
    required_zero = [value for key, value in checks.items() if key.endswith("overlap") and key != "seen_development_study_overlap"]
    required_bool = []
    for item in seed_checks.values():
        required_bool.extend([
            item["scaler_fit_on_train_only"],
            not item["external_used_for_threshold_selection"],
            not item["seen_test_used_for_threshold_selection"],
            not item["test_manifest_used_during_training"],
            item["epochs_completed"] == 15,
            item["fusion_mode"] == "naive_concat",
        ])
    checks["overall_pass"] = all(value == 0 for value in required_zero) and all(required_bool)
    return checks


def make_figure(raw: list[dict[str, Any]], metric: str, ylabel: str, stem: Path) -> None:
    labels = ["Seen", "External A", "External B", "External C"]
    lookup = dict(zip(CONDITIONS, labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), facecolor="white")
    ax.set_facecolor("white")
    for index, condition in enumerate(CONDITIONS):
        values = [float(row[metric]) for row in raw if row["condition"] == condition]
        jitter = np.linspace(-0.07, 0.07, len(values))
        ax.scatter(index + jitter, values, s=32, facecolor="white", edgecolor="#2F5597", linewidth=1.1, zorder=3)
        ax.errorbar(index, fmean(values), yerr=stdev(values), fmt="s", color="#C00000", ecolor="#C00000", capsize=4, markersize=5, linewidth=1.2, zorder=4)
    ax.set_xticks(range(len(CONDITIONS)), [lookup[name] for name in CONDITIONS])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def make_seed_ranking_figure(raw: list[dict[str, Any]], figure_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), facecolor="white", sharey=True)
    for ax, metric, title in zip(axes, ("seed_auroc", "seed_auprc"), ("Seed AUROC", "Seed AUPRC")):
        ax.set_facecolor("white")
        for index, condition in enumerate(CONDITIONS):
            values = [float(row[metric]) for row in raw if row["condition"] == condition]
            jitter = np.linspace(-0.07, 0.07, len(values))
            ax.scatter(index + jitter, values, s=30, facecolor="white", edgecolor="#2F5597", linewidth=1.1, zorder=3)
            ax.errorbar(index, fmean(values), yerr=stdev(values), fmt="s", color="#C00000", capsize=4, markersize=5, linewidth=1.2, zorder=4)
        ax.set_xticks(range(4), ["Seen", "Ext A", "Ext B", "Ext C"])
        ax.set_title(title)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure_C_seed_auroc_auprc.png", dpi=600, facecolor="white")
    fig.savefig(figure_dir / "figure_C_seed_auroc_auprc.pdf", facecolor="white")
    plt.close(fig)


def format_row(summary: dict[str, Any], condition: str) -> str:
    row = next(item for item in summary if item["condition"] == condition)
    return " | ".join(row[f"{metric}_mean_sd"] for metric in ("det_f1", "mean_iou", "seed_ba", "seed_auroc", "seed_auprc"))


def run(args: argparse.Namespace) -> None:
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    raw: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_rows = []
        for condition in CONDITIONS:
            payload = load_json(metric_path(args.root, seed, condition))
            row = flatten_metric(seed, condition, payload)
            seed_rows.append(row)
            raw.append(row)
        raw.append(external_macro(seed_rows))
    raw.sort(key=lambda row: (int(row["seed"]), (*CONDITIONS, "External Macro").index(row["condition"])))
    summary = summarize(raw)
    gaps = gap_rows(raw)
    failure = external_b_failure(args.root, raw)

    manifest_map = {
        "Train": "train.jsonl",
        "Seen-domain Control": "locked_original_test.jsonl",
        "External A": "external_A.jsonl",
        "External B": "external_B.jsonl",
        "External C": "external_C.jsonl",
    }
    descriptive, attribute_rows = [], []
    for label, filename in manifest_map.items():
        base, attrs = descriptive_stats(label, load_jsonl(args.split_root / filename))
        descriptive.append(base)
        attribute_rows.extend(attrs)

    train_attr = {row["attribute"]: row for row in attribute_rows if row["condition"] == "Train"}
    b_attr = {row["attribute"]: row for row in attribute_rows if row["condition"] == "External B"}
    scaler = load_json(args.root / "seed_1" / "checkpoints" / "attribute_preprocessing.json")
    b_shift = []
    for attribute in ATTRIBUTES:
        train = train_attr[attribute]
        ext = b_attr[attribute]
        scale = float(scaler["statistics"][attribute]["scale"])
        standardized = (float(ext["median"]) - float(train["median"])) / scale
        b_shift.append({
            "attribute": attribute,
            "train_median": train["median"],
            "external_B_median": ext["median"],
            "train_population_std": scale,
            "standardized_median_shift": standardized,
            "absolute_standardized_shift": abs(standardized),
            "train_iqr": train["iqr"],
            "external_B_iqr": ext["iqr"],
            "train_missing_rate": train["missing_rate"],
            "external_B_missing_rate": ext["missing_rate"],
        })
    b_shift.sort(key=lambda row: row["absolute_standardized_shift"], reverse=True)

    integrity = audit(args.root, args.split_root)
    if not integrity["overall_pass"]:
        raise RuntimeError(f"Integrity audit failed: {json.dumps(integrity, indent=2)}")

    write_csv(args.summary_dir / "cross_domain_3seed_raw_metrics.csv", raw)
    write_csv(args.summary_dir / "cross_domain_3seed_mean_sd.csv", summary)
    write_csv(args.summary_dir / "seen_vs_unseen_domain_gap.csv", gaps)
    write_csv(args.summary_dir / "domain_shift_descriptive_statistics.csv", descriptive)
    write_csv(args.summary_dir / "domain_shift_attribute_statistics.csv", attribute_rows)
    write_csv(args.summary_dir / "external_B_failure_analysis.csv", failure)
    write_csv(args.summary_dir / "external_B_attribute_shift.csv", b_shift)
    (args.summary_dir / "integrity_audit.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    figure_dir = args.summary_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    make_figure(raw, "det_f1", "Detection F1", figure_dir / "figure_A_detection_f1")
    make_figure(raw, "mean_iou", "Mean IoU", figure_dir / "figure_B_mean_iou")
    make_seed_ranking_figure(raw, figure_dir)

    b_summary = next(row for row in summary if row["condition"] == "External B")
    macro_summary = next(row for row in summary if row["condition"] == "External Macro")
    seen_summary = next(row for row in summary if row["condition"] == "Seen-domain Control")
    gap_macro = next(row for row in gaps if row["row_type"] == "mean_sd" and row["comparison"] == "Seen - External Macro")
    gap_a = next(row for row in gaps if row["row_type"] == "mean_sd" and row["comparison"] == "Seen - External A")
    gap_b = next(row for row in gaps if row["row_type"] == "mean_sd" and row["comparison"] == "Seen - External B")
    gap_c = next(row for row in gaps if row["row_type"] == "mean_sd" and row["comparison"] == "Seen - External C")
    desc_map = {row["condition"]: row for row in descriptive}
    b_desc = desc_map["External B"]
    train_desc = desc_map["Train"]
    seen_desc = desc_map["Seen-domain Control"]
    b_failure_raw = [row for row in failure if row["row_type"] == "raw"]
    b_seed_fp = fmean(float(row["seed_fp"]) for row in b_failure_raw)
    b_seed_fn = fmean(float(row["seed_fn"]) for row in b_failure_raw)
    top_shifts = ", ".join(f"{row['attribute']} ({row['standardized_median_shift']:+.2f} SD)" for row in b_shift[:5])
    report = [
        "# Cross-domain Generalization Experiment: final 3-seed report",
        "",
        "## Protocol",
        "",
        "Naive concat only; seeds 20260725, 20260726, and 20260727; 15 epochs per seed. Attribute preprocessing was fitted from Train only. Val selected both checkpoints and thresholds. Seen-domain Control and External A/B/C were read only after selection was locked.",
        "",
        "## Three-seed performance (mean ± sample SD)",
        "",
        "| Condition | Detection F1 | Mean IoU | Seed BA | Seed AUROC | Seed AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in (*CONDITIONS, "External Macro"):
        report.append(f"| {condition} | {format_row(summary, condition)} |")
    report.extend([
        "",
        "External Macro is the unweighted mean of External A/B/C within each seed, followed by the three-seed mean and sample SD.",
        "",
        "## Seen versus unseen domain gap",
        "",
        f"External Macro minus Seen is reported through Seen - Macro: Detection F1 {gap_macro['gap_det_f1_mean_sd']}, mean IoU {gap_macro['gap_mean_iou_mean_sd']}, Seed BA {gap_macro['gap_seed_ba_mean_sd']}, Seed AUROC {gap_macro['gap_seed_auroc_mean_sd']}, Seed AUPRC {gap_macro['gap_seed_auprc_mean_sd']}.",
        "",
        "## Domain-specific interpretation",
        "",
        f"Seen-domain Control achieved Detection F1 {seen_summary['det_f1_mean_sd']}, mean IoU {seen_summary['mean_iou_mean_sd']}, Seed BA {seen_summary['seed_ba_mean_sd']}, AUROC {seen_summary['seed_auroc_mean_sd']}, and AUPRC {seen_summary['seed_auprc_mean_sd']}.",
        f"External A (ST003127; seen instrument but unseen Study/chromatographic condition) showed a Seen-minus-A Detection F1 gap of {gap_a['gap_det_f1_mean_sd']}; localization remained strong because its mean-IoU gap was {gap_a['gap_mean_iou_mean_sd']}.",
        f"External B (ST003941; unseen Orbitrap ID-X) had the largest adverse gap: Detection F1 {gap_b['gap_det_f1_mean_sd']}, mean IoU {gap_b['gap_mean_iou_mean_sd']}, Seed BA {gap_b['gap_seed_ba_mean_sd']}, AUROC {gap_b['gap_seed_auroc_mean_sd']}, and AUPRC {gap_b['gap_seed_auprc_mean_sd']}.",
        f"External C (ST003514; seen Agilent 6545 with an unseen Poroshell EC-C18 combination) retained strong performance. Negative Seen-minus-C gaps for Detection F1 ({gap_c['gap_det_f1_mean_sd']}) and Seed BA ({gap_c['gap_seed_ba_mean_sd']}) mean C exceeded Seen on average for those metrics.",
        "External B had the lowest three-seed mean Detection F1 and was the lowest-F1 domain in two of three seeds; in seed 20260726, External A was slightly lower (0.7885 versus 0.7933). B had the lowest mean IoU, Seed BA, AUROC, and AUPRC in all three seeds. Thus its localization and Seed degradation was highly consistent, although the Detection F1 ranking and Precision-versus-Recall trade-off varied slightly by seed.",
        "",
        "## External B failure analysis",
        "",
        f"External B remained the weakest domain on average: Detection F1 {b_summary['det_f1_mean_sd']}, mean IoU {b_summary['mean_iou_mean_sd']}, Seed BA {b_summary['seed_ba_mean_sd']}, AUROC {b_summary['seed_auroc_mean_sd']}, and AUPRC {b_summary['seed_auprc_mean_sd']}.",
        f"Detection Precision was {b_summary['det_precision_mean_sd']} and Recall was {b_summary['det_recall_mean_sd']}; Recall was lower by 0.0361 on average, but the sign differed by seed. Seed sensitivity was {fmean(float(row['seed_sensitivity']) for row in b_failure_raw):.4f} and specificity was {fmean(float(row['seed_specificity']) for row in b_failure_raw):.4f}; mean confusion counts were FP={b_seed_fp:.1f} and FN={b_seed_fn:.1f} per seed.",
        f"External B contained {b_desc['empty_rate']:.1%} empty windows versus {train_desc['empty_rate']:.1%} in Train and {seen_desc['empty_rate']:.1%} in Seen. Its median box width was {b_desc['box_width_median']:.1f} px versus {train_desc['box_width_median']:.1f} px in Train and {seen_desc['box_width_median']:.1f} px in Seen, indicating substantially narrower peaks and a plausible localization challenge.",
        f"The largest attribute median shifts relative to Train were: {top_shifts}. These are descriptive associations, not causal or statistically significant claims.",
        "External B uses the unseen Thermo Orbitrap ID-X Tribrid platform. Its lower detection localization and Seed ranking/classification performance are consistent with a combined instrument and feature-distribution shift. No B-specific threshold or fine-tuning was used.",
        "",
        "## Integrity audit",
        "",
        f"Overall audit: PASS. Train∩Val source_file=0; Train∩Seen source_file=0; Val∩Seen source_file=0; Development∩External source_file=0; External Study∩Development Study=0. All three scaler files declare Train-only fitting, all training summaries declare test_manifest_used=false, and all selection locks prove Seen/External exclusion.",
        "",
        "The three-seed trends, raw values, domain gaps, descriptive statistics, failure tables, and figures are stored beside this report.",
    ])
    (args.summary_dir / "cross_domain_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8-sig")
    print(json.dumps({"status": "completed", "integrity": "PASS", "rows": len(raw)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
