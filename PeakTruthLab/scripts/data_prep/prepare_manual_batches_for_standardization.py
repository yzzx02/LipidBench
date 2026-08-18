from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


ATTRIBUTE_COLUMNS = (
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
VALID_LABELS = {"True_Peak", "OUT_FIG"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> None:
    images_root = Path(args.images_root).resolve()
    output_csv = Path(args.output_csv).resolve()
    audit_json = Path(args.audit_json).resolve()
    batch_roots = [Path(value).resolve() for value in args.batch_root]

    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    label_counts: Counter[str] = Counter()
    shape_count_distribution: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    batch_counts: dict[str, int] = {}
    json_hashes: dict[str, str] = {}

    for batch_root in batch_roots:
        manifest_path = batch_root / "annotation_manifest.csv"
        image_dir = batch_root / "eic_images"
        if not manifest_path.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(f"invalid annotation batch: {batch_root}")

        manifest = pd.read_csv(manifest_path)
        required = {
            "Feature_ID",
            "source_file",
            "source_path",
            "mz",
            "RTmin",
            "RT",
            "RTmax",
            *ATTRIBUTE_COLUMNS,
        }
        missing = sorted(required.difference(manifest.columns))
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {missing}")

        batch_counts[batch_root.name] = int(len(manifest))
        for _, source_row in manifest.iterrows():
            feature_id = str(source_row["Feature_ID"])
            if feature_id in seen_ids:
                raise ValueError(f"duplicate Feature_ID across batches: {feature_id}")
            seen_ids.add(feature_id)

            png_path = image_dir / f"{feature_id}.png"
            json_path = image_dir / f"{feature_id}.json"
            if not png_path.is_file() or not json_path.is_file():
                raise FileNotFoundError(f"missing PNG/JSON pair for {feature_id}")

            payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
            shapes = payload.get("shapes")
            if not isinstance(shapes, list):
                raise ValueError(f"LabelMe shapes is not a list: {json_path}")
            labels: list[str] = []
            for shape_index, shape in enumerate(shapes):
                if not isinstance(shape, dict):
                    raise ValueError(f"invalid shape {shape_index}: {json_path}")
                label = str(shape.get("label", "")).strip()
                if label not in VALID_LABELS:
                    raise ValueError(
                        f"unsupported label {label!r} in shape {shape_index}: {json_path}"
                    )
                if str(shape.get("shape_type", "")) != "rectangle":
                    raise ValueError(f"non-rectangle shape {shape_index}: {json_path}")
                points = shape.get("points")
                if not isinstance(points, list) or len(points) < 2:
                    raise ValueError(f"invalid rectangle points {shape_index}: {json_path}")
                labels.append(label)

            label_counts.update(labels)
            shape_count_distribution[len(labels)] += 1
            source_counts[str(source_row["source_file"])] += 1
            true_count = labels.count("True_Peak")
            out_count = labels.count("OUT_FIG")
            if true_count and out_count:
                annotation_status = "positive_with_out_fig"
            elif true_count:
                annotation_status = "positive"
            elif out_count:
                annotation_status = "out_fig_only"
            else:
                annotation_status = "negative"

            row = {
                "Feature_ID": feature_id,
                "source_file": str(source_row["source_file"]),
                "source_path": str(source_row["source_path"]),
                "mz": source_row["mz"],
                "RTmin": source_row["RTmin"],
                "RT": source_row["RT"],
                "RTmax": source_row["RTmax"],
                "feature_row_source": batch_root.name,
                "image_path": png_path.relative_to(images_root).as_posix(),
                "json_path": json_path.relative_to(images_root).as_posix(),
                "is_true_peak": int(true_count > 0),
                "annotation_status": annotation_status,
                "annotation_labels": "|".join(labels),
                "n_annotation_shapes": len(labels),
                "n_true_peak_boxes": true_count,
                "n_out_fig_boxes": out_count,
            }
            for name in ATTRIBUTE_COLUMNS:
                row[name] = source_row[name]
            rows.append(row)
            json_hashes[feature_id] = _sha256(json_path)

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("no annotation rows were prepared")
    if output["Feature_ID"].duplicated().any():
        raise AssertionError("prepared table contains duplicate Feature_ID values")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "images_root": str(images_root),
        "batch_roots": [str(value) for value in batch_roots],
        "output_csv": str(output_csv),
        "rows": int(len(output)),
        "unique_feature_ids": int(output["Feature_ID"].nunique()),
        "batch_counts": batch_counts,
        "source_counts": dict(sorted(source_counts.items())),
        "source_files": int(len(source_counts)),
        "label_counts": dict(sorted(label_counts.items())),
        "shape_count_distribution": {
            str(key): int(value)
            for key, value in sorted(shape_count_distribution.items())
        },
        "empty_negative_images": int(shape_count_distribution[0]),
        "multi_box_images": int(sum(
            value for key, value in shape_count_distribution.items() if key > 1
        )),
        "attribute_columns": list(ATTRIBUTE_COLUMNS),
        "attribute_nan_counts": {
            name: int(pd.to_numeric(output[name], errors="coerce").isna().sum())
            for name in ATTRIBUTE_COLUMNS
        },
        "input_json_sha256": json_hashes,
        "checks": {
            "rows_match_unique_ids": bool(len(output) == output["Feature_ID"].nunique()),
            "all_labels_supported": True,
            "all_shapes_rectangles": True,
            "all_png_json_pairs_exist": True,
        },
    }
    audit_json.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "input_json_sha256"}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Prepare one or more completed flat LabelMe batches for the established standardization pipeline"
    )
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--batch-root", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--audit-json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
