from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


REPO = Path(r"D:\CODE\LipidBench")
DATASET = REPO / "PeakTruthLab" / "datasets" / "PeakTruthLab_final_merged_20260814"
RESULTS = REPO / "PeakTruthLab" / "results" / "rtx4070_final_merged_20260814"
DELIVERY = REPO / "PeakTruthLab" / "artifacts" / "rtx4070_gate_stop_20260815"
ARCHIVE = DELIVERY.parent / "rtx4070_gate_stop_20260815_core.zip"

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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer columns for empty table: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source: Path, relative: Path) -> None:
    target = DELIVERY / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree_files(source: Path, relative: Path, *, omit_suffixes: set[str] | None = None) -> None:
    omit_suffixes = omit_suffixes or set()
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() not in omit_suffixes:
            copy_file(path, relative / path.relative_to(source))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    if DELIVERY.exists():
        shutil.rmtree(DELIVERY)
    DELIVERY.mkdir(parents=True)

    selfcheck = read_json(DATASET / "audits" / "merge_split_selfcheck.json")
    independent = read_json(DATASET / "audits" / "independent_verification.json")

    comparison: list[dict] = []
    learning_curves: list[dict] = []
    checkpoint_rows: list[dict] = []
    test_access_flags: list[bool] = []

    for display_name, directory_name in RUNS:
        run_dir = RESULTS / "main_ablation" / "seed_20260814" / directory_name
        summary = read_json(run_dir / "summary.json")
        selection = read_json(run_dir / "selection_on_val.json")
        config = read_json(run_dir / "run_config.json")
        selected = selection["selected_threshold_metrics"]
        test_access_flags.extend(
            [
                bool(summary.get("test_accessed")),
                bool(selection.get("test_accessed")),
                bool(config.get("test_accessed")),
            ]
        )
        comparison.append(
            {
                "model": display_name,
                "model_mode": summary["model_mode"],
                "seed": config["seed"],
                "epochs_completed": summary["epochs_completed"],
                "best_epoch": summary["best_epoch"],
                "val_roc_auc": summary["best_val_auc"],
                "val_pr_auc": selected["pr_auc"],
                "val_threshold": selection["selected_threshold"],
                "val_f1": selected["f1"],
                "val_precision": selected["precision"],
                "val_recall": selected["recall"],
                "val_specificity": selected["specificity"],
                "val_balanced_accuracy": selected["balanced_accuracy"],
                "val_tn": selected["tn"],
                "val_fp": selected["fp"],
                "val_fn": selected["fn"],
                "val_tp": selected["tp"],
                "training_seconds": summary["total_training_seconds_this_process"],
                "training_minutes": summary["total_training_seconds_this_process"] / 60,
                "gpu_peak_allocated_bytes": summary["gpu_peak_allocated_bytes"],
                "gpu_peak_allocated_gib": summary["gpu_peak_allocated_bytes"] / 1024**3,
                "checkpoint_reload_verified": summary["checkpoint_reload_verified"],
                "test_accessed": summary["test_accessed"],
            }
        )

        with (run_dir / "history.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                learning_curves.append({"model": display_name, "seed": config["seed"], **row})

        for checkpoint_name in ("best_model.pth", "last.pth"):
            checkpoint = run_dir / checkpoint_name
            checkpoint_rows.append(
                {
                    "model": display_name,
                    "checkpoint": checkpoint_name,
                    "relative_local_path": checkpoint.relative_to(REPO).as_posix(),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": sha256(checkpoint),
                    "included_in_core_archive": False,
                }
            )

        copy_tree_files(
            run_dir,
            Path("training") / "main_ablation" / "seed_20260814" / directory_name,
            omit_suffixes={".pth"},
        )

    comparison.sort(key=lambda row: float(row["val_roc_auc"]), reverse=True)
    for rank, row in enumerate(comparison, start=1):
        row["val_auc_rank"] = rank

    winner = comparison[0]
    concat = next(row for row in comparison if row["model"] == "Naive concat")
    concat_is_best = winner["model"] == "Naive concat"
    all_test_unaccessed = not any(test_access_flags)
    if not all_test_unaccessed:
        raise RuntimeError("A Test access flag is true; refusing to package as a pre-Test gate report")

    gate = {
        "status": "STOPPED_BY_PREDECLARED_VALIDATION_GATE",
        "decision_date": "2026-08-15",
        "selection_metric": "maximum validation ROC-AUC",
        "winner": winner["model"],
        "winner_val_roc_auc": winner["val_roc_auc"],
        "concat_val_roc_auc": concat["val_roc_auc"],
        "winner_minus_concat": float(winner["val_roc_auc"]) - float(concat["val_roc_auc"]),
        "concat_is_best": concat_is_best,
        "continue_to_three_seed_confirmation": False,
        "continue_to_main_test": False,
        "continue_to_lodo": False,
        "reason": (
            "The frozen protocol requires Naive concat to rank first on Val before Test or LODO. "
            "Image-gated-Attribute ranked first, so training and evaluation stop here."
        ),
        "main_test_accessed": False,
        "test_manifest_remains_locked": True,
        "no_test_based_tuning": True,
    }

    write_csv(DELIVERY / "main_ablation_validation_summary.csv", comparison)
    write_csv(DELIVERY / "main_ablation_learning_curves.csv", learning_curves)
    write_csv(DELIVERY / "checkpoint_sha256.csv", checkpoint_rows)
    write_json(DELIVERY / "GATE_DECISION.json", gate)

    copy_file(DATASET / "README.md", Path("data") / "README.md")
    copy_tree_files(DATASET / "audits", Path("data") / "audits")
    copy_tree_files(DATASET / "splits", Path("data") / "splits")
    for name in (
        "domain_definitions.csv",
        "field_dictionary.csv",
        "seed_master_16attrs.csv",
        "peak_instances_master_16attrs.csv",
    ):
        copy_file(DATASET / "tables" / name, Path("data") / "tables" / name)
    copy_file(DATASET / "SHA256SUMS_core.csv", Path("data") / "SHA256SUMS_core.csv")
    copy_tree_files(
        RESULTS / "smoke" / "naive_concat_b16_e1_deterministic",
        Path("training") / "smoke" / "naive_concat_b16_e1_deterministic",
        omit_suffixes={".pth"},
    )

    counts = selfcheck["counts"]
    duplicates = selfcheck["duplicates"]
    split = selfcheck["split"]
    test_lock = selfcheck["test_lock"]
    attrs = "SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE"
    report_lines = [
        "# RTX 4070 merged-data and validation-gate report",
        "",
        "## Outcome",
        "",
        "The four frozen main-ablation runs completed successfully. The predeclared gate stopped the experiment before Test and LODO because Image-gated-Attribute, not Naive concat, achieved the highest validation ROC-AUC. No Test predictions or Test-derived tuning were performed.",
        "",
        "## Protected and merged data",
        "",
        f"- Protected old-final images: {counts['old_images']:,}",
        f"- New v2 images: {counts['new_images']:,}",
        f"- Merged images / Seed candidates: {counts['merged_images']:,}",
        f"- Seed positives / negatives: {counts['seed_positive']:,} / {counts['seed_negative']:,}",
        f"- Peak instances: {counts['merged_peak_instances']:,} ({counts['true_peak_instances']:,} True_Peak; {counts['out_fig_instances']:,} OUT_FIG)",
        f"- Exact ID or image-SHA duplicate groups: {duplicates['exact_id_groups']} / {duplicates['exact_image_sha256_groups']}",
        f"- Near-duplicate groups: {duplicates['near_duplicate_groups']}; audited label-conflict groups: {duplicates['near_label_conflict_groups']}; Test-excluded conflict rows: {duplicates['test_exclusion_rows']}",
        f"- Sixteen raw attributes: {attrs}",
        "- Missing old-final values were preserved; median imputation and population z-score parameters were fitted on Train only for every run.",
        "",
        "## Frozen split and leakage checks",
        "",
        f"- Train: {split['train']['rows']:,} ({split['train']['seed_positive']:,} positive / {split['train']['seed_negative']:,} negative)",
        f"- Val: {split['val']['rows']:,} ({split['val']['seed_positive']:,} positive / {split['val']['seed_negative']:,} negative)",
        f"- Locked Test: {split['test']['rows']:,} ({split['test']['seed_positive']:,} positive / {split['test']['seed_negative']:,} negative)",
        "- Train/Val/Test intersections are zero for image ID, Seed ID, path, annotation path, image SHA-256 and duplicate/split group.",
        f"- Locked Test CSV SHA-256: `{test_lock['test_csv_sha256']}`",
        f"- Locked Test JSONL SHA-256: `{test_lock['test_jsonl_sha256']}`",
        f"- Independent verification status: `{independent.get('status', 'unknown')}`",
        "",
        "## Frozen training configuration",
        "",
        "Input 480x480; batch size 16; 30 epochs; seed 20260814; FP16; AdamW; learning rate 1e-4; weight decay 1e-4; dropout 0.2; no augmentation; pretrained ConvNeXt-Tiny where images are used. Checkpoint selection used maximum Val ROC-AUC. The classification threshold was selected on Val by a 0.001 grid maximizing balanced accuracy, then F1, then closeness to 0.5.",
        "",
        "## Main-ablation validation results",
        "",
        "| Rank | Model | Best epoch | ROC-AUC | PR-AUC | F1 | Precision | Recall | Balanced accuracy | Threshold | Minutes | Peak GiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        report_lines.append(
            f"| {row['val_auc_rank']} | {row['model']} | {row['best_epoch']} | {fmt(float(row['val_roc_auc']))} | "
            f"{fmt(float(row['val_pr_auc']))} | {fmt(float(row['val_f1']))} | {fmt(float(row['val_precision']))} | "
            f"{fmt(float(row['val_recall']))} | {fmt(float(row['val_balanced_accuracy']))} | {float(row['val_threshold']):.3f} | "
            f"{float(row['training_minutes']):.2f} | {float(row['gpu_peak_allocated_gib']):.2f} |"
        )
    report_lines.extend(
        [
            "",
            "## Gate decision",
            "",
            f"Image-gated-Attribute exceeded Naive concat by {gate['winner_minus_concat']:.9f} Val ROC-AUC. Because Naive concat was not rank 1, the frozen protocol requires an immediate stop. No extra seeds were launched, the locked main Test was not evaluated, and LODO was not started.",
            "",
            "## Reproducibility and archive policy",
            "",
            "The core archive contains source tables, split manifests, QC/audit tables, smoke-test outputs, all four run configurations, environments, Train-only scalers, epoch histories, Val predictions, threshold sweeps and selection summaries. Large `.pth` files remain in the local result directories and are represented by path, size and SHA-256 in `checkpoint_sha256.csv`.",
            "",
        ]
    )
    (DELIVERY / "DATA_AND_VALIDATION_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    readme = """# RTX 4070 gate-stop reproducibility bundle

This bundle records the completed data merge/QC/split, deterministic smoke test and four-model validation ablation. It intentionally contains no Test metrics: the frozen Concat continuation gate failed because Image-gated-Attribute ranked first on Val.

Start with `DATA_AND_VALIDATION_REPORT.md`, `main_ablation_validation_summary.csv`, `main_ablation_learning_curves.csv`, `GATE_DECISION.json` and `checkpoint_sha256.csv`.
"""
    (DELIVERY / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    for path in sorted(DELIVERY.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(DELIVERY).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    write_csv(DELIVERY / "FILE_MANIFEST_SHA256.csv", manifest_rows)

    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(DELIVERY.rglob("*")):
            if path.is_file():
                archive.write(path, Path(DELIVERY.name) / path.relative_to(DELIVERY))

    archive_record = {
        "archive": ARCHIVE.name,
        "bytes": ARCHIVE.stat().st_size,
        "sha256": sha256(ARCHIVE),
        "checkpoint_files_included": False,
        "test_metrics_included": False,
    }
    write_json(DELIVERY.parent / "rtx4070_gate_stop_20260815_core.zip.sha256.json", archive_record)
    print(json.dumps({"delivery": str(DELIVERY), "archive": archive_record, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
