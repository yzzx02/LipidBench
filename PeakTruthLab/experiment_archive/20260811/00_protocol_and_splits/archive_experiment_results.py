from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path


REPO = Path(r"D:\CODE\LipidBench")
RESULTS = REPO / "PeakTruthLab" / "results"
PAPER = RESULTS / "paper_final_reviewed_20260725"
DOMAIN = RESULTS / "domain_generalization"
DEST = REPO / "PeakTruthLab" / "experiment_archive" / "20260811"
SCRIPT = Path(__file__).resolve()

PILOT_ROOTS = {
    "pilot_2000_train_500_val": RESULTS / "nvidia_2000_500_b8_e5",
    "full_image_only_10epochs": RESULTS / "nvidia_full_10525_2507_b8_e10_image_only",
}

CORE_TRAINING_NAMES = {
    "attribute_preprocessing.json",
    "history.json",
    "run_config.json",
    "summary.json",
}

RAW_EVAL_NAMES = {
    "selection_before_test.json",
    "test_metrics.json",
    "metrics.json",
    "test_detection_ap_summary.csv",
    "detection_ap_summary.csv",
    "test_detection_image_metrics.csv",
    "detection_image_metrics.csv",
    "test_detection_subgroups.csv",
    "detection_subgroups.csv",
    "test_seed_predictions.csv",
    "seed_predictions.csv",
    "test_seed_subgroups.csv",
    "seed_subgroups.csv",
    "val_detection_threshold_sweep.csv",
    "detection_threshold_sweep.csv",
    "val_seed_threshold_sweep.csv",
    "seed_threshold_sweep.csv",
    "threshold_selection.json",
    "threshold_selection_lock.json",
}

STATUS_NAMES = {
    "all_val_selections_complete.json",
    "evaluation_complete.json",
    "evaluation_status.json",
    "gated_hyperparameter_lock.json",
    "gated_postprocess_status.json",
    "gated_training_status.json",
    "hyperparameter_lock.json",
    "postprocess_status.json",
    "protocol_lock.json",
    "seed_20260725_reference.json",
    "training_status.json",
}


copied: list[tuple[Path, Path, str]] = []


def copy_file(source: Path, relative_dest: Path, category: str) -> None:
    if not source.is_file():
        return
    target = DEST / relative_dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied.append((source, target, category))


def copy_tree(source: Path, relative_dest: Path, category: str, predicate) -> None:
    if not source.is_dir():
        return
    for item in sorted(source.rglob("*")):
        if item.is_file() and predicate(item, item.relative_to(source)):
            copy_file(item, relative_dest / item.relative_to(source), category)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_text_table_document(path: Path) -> bool:
    return path.suffix.lower() in {
        ".csv", ".json", ".jsonl", ".md", ".mjs", ".py", ".xlsx", ".yaml", ".yml"
    }


def add_pilot_runs() -> None:
    for label, root in PILOT_ROOTS.items():
        for name in sorted(CORE_TRAINING_NAMES):
            copy_file(root / name, Path("01_pilot_and_image_only") / label / name, "pilot/image-only")


def add_data_quality_and_paper_tables() -> None:
    copy_tree(
        PAPER / "data_quality",
        Path("02_mixed_domain_ablation") / "data_quality",
        "data quality",
        lambda path, rel: is_text_table_document(path),
    )
    copy_tree(
        PAPER / "paper_tables_figures",
        Path("02_mixed_domain_ablation") / "paper_tables",
        "paper tables",
        lambda path, rel: is_text_table_document(path)
        and path.suffix.lower() not in {".png", ".pdf"},
    )


def add_mixed_domain_training() -> None:
    for section in ("experiments", "epoch15_probe"):
        root = PAPER / section
        copy_tree(
            root,
            Path("02_mixed_domain_ablation") / section,
            "mixed-domain training",
            lambda path, rel: path.name in CORE_TRAINING_NAMES,
        )

    stability = PAPER / "stability_3seeds_20260729"
    for item in sorted(stability.iterdir()):
        if item.is_file() and (item.name in STATUS_NAMES or is_text_table_document(item)):
            copy_file(
                item,
                Path("02_mixed_domain_ablation") / "stability_3seeds" / item.name,
                "stability protocol/status",
            )

    copy_tree(
        stability / "experiments",
        Path("02_mixed_domain_ablation") / "stability_3seeds" / "experiments",
        "stability training",
        lambda path, rel: path.name in CORE_TRAINING_NAMES,
    )
    copy_tree(
        stability / "summary",
        Path("02_mixed_domain_ablation") / "stability_3seeds" / "summary",
        "stability summary",
        lambda path, rel: is_text_table_document(path)
        and path.suffix.lower() not in {".png", ".pdf", ".ndjson"},
    )
    copy_tree(
        stability / "test_evaluation",
        Path("02_mixed_domain_ablation") / "stability_3seeds" / "test_evaluation",
        "stability raw evaluation",
        lambda path, rel: path.name in RAW_EVAL_NAMES or path.name in STATUS_NAMES,
    )

    copy_tree(
        PAPER / "test_evaluation",
        Path("02_mixed_domain_ablation") / "single_seed_test_evaluation",
        "single-seed raw evaluation",
        lambda path, rel: path.name in RAW_EVAL_NAMES
        or path.name in STATUS_NAMES
        or path.name in {"ablation_test_metrics.csv", "ablation_test_metrics.json"},
    )

    copy_tree(
        PAPER / "00_provenance" / "code",
        Path("02_mixed_domain_ablation") / "provenance",
        "mixed-domain provenance",
        lambda path, rel: path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".md"},
    )


def add_cross_domain() -> None:
    split_root = DOMAIN / "splits"
    for item in sorted(split_root.glob("*")):
        if item.suffix.lower() in {".csv", ".json"}:
            copy_file(item, Path("03_cross_domain") / "splits" / item.name, "cross-domain split")

    copy_tree(
        DOMAIN / "metadata",
        Path("03_cross_domain") / "metadata",
        "cross-domain metadata",
        lambda path, rel: is_text_table_document(path),
    )

    for seed_dir in ("seed_1", "seed_20260726", "seed_20260727"):
        root = DOMAIN / seed_dir
        if not root.exists():
            continue
        copy_tree(
            root,
            Path("03_cross_domain") / seed_dir,
            "cross-domain raw results",
            lambda path, rel: (
                path.name in CORE_TRAINING_NAMES
                or path.name in RAW_EVAL_NAMES
                or path.name in STATUS_NAMES
                or path.name.startswith("protocol_lock")
                or path.name in {"condition_metrics.csv", "cross_domain_stage1_report.md"}
            ),
        )

    copy_tree(
        DOMAIN / "summary",
        Path("03_cross_domain") / "summary",
        "cross-domain summary",
        lambda path, rel: is_text_table_document(path)
        and path.suffix.lower() not in {".png", ".pdf", ".ndjson"}
        and "figures" not in rel.parts,
    )

    for item in DOMAIN.glob("*"):
        if item.is_file() and is_text_table_document(item):
            copy_file(item, Path("03_cross_domain") / item.name, "cross-domain report")


def add_canonical_manifests() -> None:
    datasets = REPO / "PeakTruthLab" / "datasets"
    for name in ("source_domain_metadata.csv",):
        copy_file(
            datasets / name,
            Path("00_protocol_and_splits") / name,
            "canonical metadata",
        )
    copy_file(SCRIPT, Path("00_protocol_and_splits") / "archive_experiment_results.py", "archive provenance")


def write_checkpoint_index() -> None:
    rows = []
    roots = [PAPER, DOMAIN, *PILOT_ROOTS.values()]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pt")):
            rows.append(
                {
                    "source_path": str(path),
                    "bytes": path.stat().st_size,
                    "size_gib": f"{path.stat().st_size / (1024 ** 3):.4f}",
                    "checkpoint_name": path.name,
                    "uploaded": "no",
                    "reason": "large binary model artifact; retained locally",
                }
            )
    target = DEST / "CHECKPOINT_INDEX.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_exclusion_summary() -> None:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    roots = [PAPER, DOMAIN, *PILOT_ROOTS.values()]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".pt":
                reason = "model checkpoints"
            elif path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                reason = "images and overlays"
            elif path.suffix.lower() == ".pdf":
                reason = "generated figures in PDF"
            elif "dataset_release" in path.parts:
                reason = "dataset release copy"
            else:
                continue
            stats[reason][0] += 1
            stats[reason][1] += path.stat().st_size

    target = DEST / "EXCLUDED_ARTIFACTS_SUMMARY.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["category", "file_count", "bytes", "size_gib", "reason"])
        for category, (count, size) in sorted(stats.items()):
            writer.writerow(
                [category, count, size, f"{size / (1024 ** 3):.4f}", "kept locally; not needed for numerical re-analysis"]
            )


README = """# LipidBench experiment result archive (2026-08-11)

This directory consolidates the numerical outputs and provenance from the LipidBench / PeakTruthLab experiments completed in this working session. It is intentionally optimized for paper tables, plots, auditing and downstream re-analysis.

## Contents

- `00_protocol_and_splits/`: canonical source-domain metadata and the exact archive-building script.
- `01_pilot_and_image_only/`: the 2,000/500 pilot and the full 10-epoch image-only baseline (config, preprocessing, epoch history and summary).
- `02_mixed_domain_ablation/`: reviewed-data quality report, single-seed ablation, epoch-15 probe, four-model three-seed stability experiment, threshold locks, compact per-image Test outputs and paper-ready tables.
- `03_cross_domain/`: frozen Train/Val/Seen/External A/B/C split tables, per-seed training histories, validation threshold selection, per-domain raw numerical evaluation and final three-seed summary.
- `MANIFEST.csv`: SHA-256 and original source path for every archived file.
- `CHECKPOINT_INDEX.csv`: local location and size of every excluded checkpoint.
- `EXCLUDED_ARTIFACTS_SUMMARY.csv`: excluded binary/image categories and sizes.

The full domain manifest remains at `PeakTruthLab/datasets/domain_generalization_manifest.jsonl`; it is tracked once at its canonical location rather than duplicated here.

## Locked experimental protocol

- Input: 480 x 480; batch size 8; FP16.
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4.
- No data augmentation; 15 epochs for the final stability and cross-domain protocols.
- Attribute preprocessing was fitted from Train only.
- Checkpoints and Detection/Seed thresholds were selected using Val only.
- Test and External A/B/C were evaluated only after selections were locked.
- Cross-domain external sets were frozen as A = ST003127, B = ST003941 and C = ST003514, and were never pooled into one reported Test metric.

## Core three-seed stability results (mean +/- sample SD)

| Model | Detection F1 | Mean IoU | Seed balanced accuracy | Seed AUROC |
|---|---:|---:|---:|---:|
| Image only | 0.8608 +/- 0.0045 | 0.8696 +/- 0.0060 | 0.9651 +/- 0.0019 | 0.9938 +/- 0.0009 |
| Attributes only | 0.8521 +/- 0.0103 | 0.8777 +/- 0.0016 | 0.9520 +/- 0.0033 | 0.9901 +/- 0.0006 |
| Naive concat | 0.8613 +/- 0.0060 | 0.8765 +/- 0.0029 | 0.9663 +/- 0.0058 | 0.9954 +/- 0.0014 |
| Gated fusion | 0.8502 +/- 0.0083 | 0.8727 +/- 0.0065 | 0.9652 +/- 0.0048 | 0.9947 +/- 0.0013 |

## Core cross-domain Naive-concat results (mean +/- sample SD, n = 3)

| Domain | Detection F1 | Mean IoU | Seed balanced accuracy | Seed AUROC |
|---|---:|---:|---:|---:|
| Seen | 0.8460 +/- 0.0317 | 0.8637 +/- 0.0138 | 0.9261 +/- 0.0096 | 0.9796 +/- 0.0057 |
| External A | 0.8081 +/- 0.0192 | 0.8760 +/- 0.0095 | 0.9085 +/- 0.0235 | 0.9903 +/- 0.0026 |
| External B | 0.7900 +/- 0.0073 | 0.8331 +/- 0.0068 | 0.8663 +/- 0.0033 | 0.9431 +/- 0.0090 |
| External C | 0.8787 +/- 0.0053 | 0.8719 +/- 0.0068 | 0.9541 +/- 0.0086 | 0.9889 +/- 0.0007 |
| External macro | 0.8256 +/- 0.0065 | 0.8603 +/- 0.0062 | 0.9096 +/- 0.0115 | 0.9741 +/- 0.0038 |

External B was consistently the hardest domain, particularly for mean IoU and Seed classification. See `03_cross_domain/summary/external_B_failure_analysis.csv` and `external_B_attribute_shift.csv` for the quantitative shift analysis.

## Deliberate exclusions

No training was restarted for this archive. Large `.pt` files, dataset-release copies, review images, prediction overlays and generated PNG/PDF figures were not uploaded. These artifacts remain on the local D: drive and are indexed or summarized here. The preserved CSV/JSON/XLSX files are sufficient to redraw figures without depending on the excluded artwork.
"""


def write_readme() -> None:
    (DEST / "README.md").write_text(README, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for source, target, category in copied:
        rows.append(
            {
                "archive_path": target.relative_to(DEST).as_posix(),
                "category": category,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "original_source_path": str(source),
            }
        )
    for target, category in (
        (DEST / "README.md", "archive documentation"),
        (DEST / "CHECKPOINT_INDEX.csv", "excluded checkpoint index"),
        (DEST / "EXCLUDED_ARTIFACTS_SUMMARY.csv", "exclusion summary"),
    ):
        rows.append(
            {
                "archive_path": target.relative_to(DEST).as_posix(),
                "category": category,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "original_source_path": "generated during curation",
            }
        )
    rows.sort(key=lambda row: row["archive_path"])
    with (DEST / "MANIFEST.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite existing archive: {DEST}")
    DEST.mkdir(parents=True)
    add_canonical_manifests()
    add_pilot_runs()
    add_data_quality_and_paper_tables()
    add_mixed_domain_training()
    add_cross_domain()
    write_checkpoint_index()
    write_exclusion_summary()
    write_readme()
    write_manifest()
    total_bytes = sum(path.stat().st_size for path in DEST.rglob("*") if path.is_file())
    print(json.dumps({"files": sum(1 for p in DEST.rglob('*') if p.is_file()), "bytes": total_bytes, "mib": round(total_bytes / (1024 ** 2), 2)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
