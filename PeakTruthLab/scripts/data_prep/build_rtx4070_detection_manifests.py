from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ATTRIBUTE_NAMES = (
    "SNR",
    "CV",
    "GS",
    "TPAS",
    "H2B",
    "ZZ",
    "DZZ",
    "PCC",
    "SKEW",
    "DENT",
    "DM",
    "ENT",
    "JAG",
    "SYM",
    "MOD",
    "EDGE",
)
VALID_SPLITS = ("train", "val", "test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_optional_float(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    result = float(text)
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def normalise_box(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{name} must contain four coordinates")
    box = [float(item) for item in value]
    if not all(math.isfinite(item) for item in box):
        raise ValueError(f"{name} contains non-finite coordinates")
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid {name}: {box}")
    return box


def box_from_shape(shape: dict[str, Any], *, name: str) -> list[float]:
    if str(shape.get("shape_type", "rectangle")).lower() != "rectangle":
        raise ValueError(f"{name} is not a LabelMe rectangle")
    points = shape.get("points")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError(f"{name} must contain two rectangle points")
    coordinates: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"invalid point in {name}: {point!r}")
        coordinates.append((float(point[0]), float(point[1])))
    x_values = [point[0] for point in coordinates]
    y_values = [point[1] for point in coordinates]
    return normalise_box(
        [min(x_values), min(y_values), max(x_values), max(y_values)],
        name=name,
    )


def subsets(seed_label: int, box_count: int) -> list[str]:
    if box_count == 0:
        detection = "detection_empty"
    elif box_count == 1:
        detection = "detection_single"
    else:
        detection = "detection_multi"
    return ["seed_positive" if seed_label else "seed_negative", detection]


def build_record(
    row: dict[str, str],
    *,
    dataset_root: Path,
    label_counts: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = row["image_id"].strip()
    image_relative = row["image"].replace("\\", "/").strip()
    annotation_relative = row["annotation_json"].replace("\\", "/").strip()
    image_path = dataset_root / Path(image_relative)
    annotation_path = dataset_root / Path(annotation_relative)
    if not image_path.is_file():
        raise FileNotFoundError(f"{sample_id}: missing image {image_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"{sample_id}: missing annotation {annotation_path}")

    annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    shapes = annotation.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError(f"{sample_id}: LabelMe shapes is not a list")
    boxes: list[list[float]] = []
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise ValueError(f"{sample_id}: shape {index} is not an object")
        label = str(shape.get("label", "")).strip()
        label_counts[label] += 1
        if label.casefold() == "true_peak":
            boxes.append(box_from_shape(shape, name=f"{sample_id}.shapes[{index}]"))

    declared_box_count = int(row["true_peak_count"])
    if len(boxes) != declared_box_count:
        raise ValueError(
            f"{sample_id}: LabelMe True_Peak count {len(boxes)} != "
            f"master true_peak_count {declared_box_count}"
        )
    seed_box = normalise_box(
        json.loads(row["seed_box_json"]),
        name=f"{sample_id}.seed_box",
    )
    seed_label = int(row["seed_label"])
    if seed_label not in (0, 1):
        raise ValueError(f"{sample_id}: invalid seed_label {seed_label}")
    attributes = [parse_optional_float(row[name]) for name in ATTRIBUTE_NAMES]
    split = row["split"].strip().lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"{sample_id}: invalid split {split!r}")

    record = {
        "sample_id": sample_id,
        "image_path": image_relative,
        "boxes": boxes,
        "seed_box": seed_box,
        "seed_label": seed_label,
        "attributes": attributes,
        "source_file": row["source_file"].strip(),
        "study_id": row["domain_id"].strip(),
        "instrument_id": "unknown",
        "subsets": subsets(seed_label, len(boxes)),
        "metadata": {
            "seed_id": row["seed_id"].strip(),
            "original_image_id": row["original_image_id"].strip(),
            "original_seed_id": row["original_seed_id"].strip(),
            "old_new_batch": row["old_new_batch"].strip(),
            "batch_name": row["batch_name"].strip(),
            "domain_id": row["domain_id"].strip(),
            "domain_name": row["domain_name"].strip(),
            "source_mzml_name": row["source_mzml_name"].strip(),
            "mz": parse_optional_float(row["mz"]),
            "difficulty_type": row["difficulty_type"].strip(),
            "annotation_json": annotation_relative,
            "true_peak_count": len(boxes),
            "out_fig_count": int(row["out_fig_count"]),
            "split": split,
            "split_group_id": row["split_group_id"].strip(),
            "attribute_provenance": row["attribute_provenance"].strip(),
            "image_sha256": row["image_sha256"].strip(),
            "annotation_sha256": row["annotation_sha256"].strip(),
        },
    }
    audit = {
        "sample_id": sample_id,
        "split": split,
        "study_id": record["study_id"],
        "old_new_batch": row["old_new_batch"].strip(),
        "seed_label": seed_label,
        "box_count": len(boxes),
        "missing_attribute_count": sum(value is None for value in attributes),
        "image_path": image_relative,
        "annotation_json": annotation_relative,
    }
    return record, audit


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build checked 16-attribute multi-task manifests from the frozen merged dataset."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    master_path = dataset_root / "tables" / "seed_master_16attrs.csv"
    rows = read_csv(master_path)
    if not rows:
        raise ValueError("seed master is empty")
    sample_ids = [row["image_id"].strip() for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("seed master contains duplicate image_id values")

    label_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        record, audit = build_record(
            row,
            dataset_root=dataset_root,
            label_counts=label_counts,
        )
        records.append(record)
        audit_rows.append(audit)
        by_id[record["sample_id"]] = record
        if index % 2000 == 0:
            print(f"validated {index}/{len(rows)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Path] = {}
    for split in VALID_SPLITS:
        selected = [record for record in records if record["metadata"]["split"] == split]
        path = output_dir / "main" / f"{split}.jsonl"
        write_jsonl(path, selected)
        manifests[f"main/{split}.jsonl"] = path
    all_path = output_dir / "all.jsonl"
    write_jsonl(all_path, records)
    manifests["all.jsonl"] = all_path

    lodo_root = dataset_root / "lodo_seed_20260814"
    fold_rows = read_csv(lodo_root / "LODO_FOLD_MANIFEST.csv")
    lodo_summary: list[dict[str, Any]] = []
    for fold in fold_rows:
        fold_name = (
            fold.get("fold_id", "").strip()
            or fold.get("fold_dir", "").strip()
            or fold.get("fold_name", "").strip()
        )
        if not fold_name:
            fold_number = fold.get("fold_index", "").strip() or fold.get("fold", "").strip()
            candidates = sorted(lodo_root.glob(f"fold_{int(fold_number):02d}_*"))
            if len(candidates) != 1:
                raise ValueError(f"cannot resolve LODO fold directory from {fold}")
            fold_name = candidates[0].name
        source_dir = lodo_root / fold_name
        target_dir = output_dir / "lodo" / fold_name
        summary_row: dict[str, Any] = {"fold": fold_name}
        fold_sets: dict[str, set[str]] = {}
        for split_name in ("train", "val", "heldout"):
            source_name = "heldout_test.csv" if split_name == "heldout" else f"{split_name}.csv"
            split_rows = read_csv(source_dir / source_name)
            ids = [row["image_id"].strip() for row in split_rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{fold_name}/{split_name}: duplicate image_id")
            missing = sorted(set(ids).difference(by_id))
            if missing:
                raise ValueError(f"{fold_name}/{split_name}: unknown ids {missing[:3]}")
            selected = [by_id[sample_id] for sample_id in ids]
            path = target_dir / f"{split_name}.jsonl"
            write_jsonl(path, selected)
            manifests[f"lodo/{fold_name}/{split_name}.jsonl"] = path
            fold_sets[split_name] = set(ids)
            summary_row[f"{split_name}_samples"] = len(ids)
            summary_row[f"{split_name}_boxes"] = sum(len(record["boxes"]) for record in selected)
        if fold_sets["train"] & fold_sets["val"]:
            raise ValueError(f"{fold_name}: train/val sample overlap")
        if fold_sets["heldout"] & (fold_sets["train"] | fold_sets["val"]):
            raise ValueError(f"{fold_name}: heldout sample overlap")
        lodo_summary.append(summary_row)

    write_audit_csv(output_dir / "manifest_sample_audit.csv", audit_rows)
    write_audit_csv(output_dir / "lodo_manifest_summary.csv", lodo_summary)
    checksums = {
        relative: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for relative, path in sorted(manifests.items())
    }
    split_counts = Counter(record["metadata"]["split"] for record in records)
    box_classes = Counter(
        "empty" if not record["boxes"] else "single" if len(record["boxes"]) == 1 else "multi"
        for record in records
    )
    qc = {
        "status": "ok",
        "dataset_root": str(dataset_root),
        "source_master": str(master_path),
        "source_master_sha256": sha256(master_path),
        "attribute_names": list(ATTRIBUTE_NAMES),
        "attribute_state": "raw; train-fitted preprocessing is applied by the trainer",
        "samples": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "seed_positive": sum(record["seed_label"] == 1 for record in records),
        "seed_negative": sum(record["seed_label"] == 0 for record in records),
        "true_peak_boxes": sum(len(record["boxes"]) for record in records),
        "detection_classes": dict(sorted(box_classes.items())),
        "rows_with_missing_attributes": sum(
            any(value is None for value in record["attributes"]) for record in records
        ),
        "labelme_shape_labels": dict(sorted(label_counts.items())),
        "lodo_folds": len(lodo_summary),
        "checksums": checksums,
        "guarantees": [
            "True_Peak rectangles are detection targets; OUT_FIG shapes are excluded.",
            "Every LabelMe True_Peak count equals seed_master true_peak_count.",
            "No image, JSON or source split was modified.",
            "Test manifests were generated but are not read during training or Val selection.",
        ],
    }
    write_json(output_dir / "detection_manifest_qc.json", qc)
    print(json.dumps(qc, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
