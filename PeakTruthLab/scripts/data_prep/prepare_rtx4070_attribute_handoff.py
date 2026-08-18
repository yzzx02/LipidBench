from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd


ATTR13 = [
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
]
VALID_LABELS = {"True_Peak", "OUT_FIG"}
PEAK_COORD_COLS = [
    "修正后框_xmin_px",
    "修正后框_ymin_px",
    "修正后框_xmax_px",
    "修正后框_ymax_px",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def review_tree_sha256(review_dir: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(review_dir.glob("*.json"), key=lambda p: p.name):
        h.update(path.name.encode("utf-8"))
        h.update(b"\0")
        h.update(bytes.fromhex(sha256_file(path)))
    return h.hexdigest()


def normalize_box(values: Any) -> list[float]:
    if isinstance(values, str):
        values = json.loads(values)
    vals = [float(x) for x in values]
    if len(vals) != 4:
        raise ValueError(f"box must contain four values, got {values!r}")
    x1, y1, x2, y2 = vals
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def shape_box(shape: dict[str, Any]) -> list[float]:
    points = shape.get("points") or []
    if len(points) < 2:
        raise ValueError(f"invalid LabelMe shape: {shape!r}")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def max_coord_diff(a: list[float], b: list[float]) -> float:
    return float(max(abs(a[i] - b[i]) for i in range(4)))


def assign_boxes(
    targets: list[list[float]],
    candidates: list[list[float]],
    *,
    min_iou: float = 0.0,
    exact_tolerance: float = 1e-4,
) -> dict[int, tuple[int, float, float]]:
    pairs: list[tuple[int, float, float, int, int]] = []
    for ti, target in enumerate(targets):
        for ci, candidate in enumerate(candidates):
            iou = box_iou(target, candidate)
            diff = max_coord_diff(target, candidate)
            exact = int(diff <= exact_tolerance)
            pairs.append((exact, iou, -diff, ti, ci))
    pairs.sort(reverse=True)

    used_targets: set[int] = set()
    used_candidates: set[int] = set()
    assignments: dict[int, tuple[int, float, float]] = {}
    for exact, iou, neg_diff, ti, ci in pairs:
        if ti in used_targets or ci in used_candidates:
            continue
        diff = -neg_diff
        if not exact and iou < min_iou:
            continue
        used_targets.add(ti)
        used_candidates.add(ci)
        assignments[ti] = (ci, float(iou), float(diff))
    return assignments


def load_manifests(manifest_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        hashes[split] = sha256_file(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["_old_split"] = split
                rows.append(row)
    return rows, hashes


def load_review_shapes(review_dir: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(review_dir.glob("*.json"), key=lambda p: p.name):
        if "__" not in path.stem:
            raise ValueError(f"review JSON name lacks queue prefix: {path.name}")
        sample_id = path.stem.split("__", 1)[1]
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
        shapes = []
        for shape in obj.get("shapes", []):
            label = str(shape.get("label", ""))
            if label not in VALID_LABELS:
                continue
            shapes.append(
                {
                    "label": label,
                    "box": shape_box(shape),
                    "review_json": path.name,
                }
            )
        output[sample_id] = shapes
    return output


def source_mzml_name(manifest_row: dict[str, Any]) -> str:
    metadata = manifest_row.get("metadata") or {}
    raw_path = str(metadata.get("mzml_path") or "")
    if raw_path:
        return PureWindowsPath(raw_path).name
    source = str(manifest_row.get("source_file") or "")
    return source if source.lower().endswith(".mzml") else f"{source}.mzML"


def parse_plot_area(value: Any) -> tuple[float, float, float, float]:
    parsed = ast.literal_eval(str(value)) if isinstance(value, str) else value
    vals = [float(x) for x in parsed]
    if len(vals) != 4:
        raise ValueError(f"invalid plot area: {value!r}")
    return vals[0], vals[1], vals[2], vals[3]


def pixel_x_to_rt(x: float, plot_xmin: float, plot_xmax: float, center_rt: float) -> float:
    if plot_xmax <= plot_xmin:
        raise ValueError("plot_xmax must exceed plot_xmin")
    return float(center_rt - 1.0 + (float(x) - plot_xmin) * 2.0 / (plot_xmax - plot_xmin))


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare authoritative old-final Seed and peak jobs for RTX4070 16-attribute completion."
    )
    parser.add_argument("--seed-table", type=Path, required=True)
    parser.add_argument("--peak-table", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_table = args.seed_table.resolve()
    peak_table = args.peak_table.resolve()
    manifest_dir = args.manifest_dir.resolve()
    review_dir = args.review_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_df = pd.read_csv(seed_table, encoding="utf-8-sig", low_memory=False)
    peak_df = pd.read_csv(peak_table, encoding="utf-8-sig", low_memory=False)
    manifests, manifest_hashes = load_manifests(manifest_dir)
    review_shapes = load_review_shapes(review_dir)

    seed_by_sample = {
        str(row["原始特征编号"]): row for _, row in seed_df.iterrows()
    }
    old_peaks_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, row in peak_df.iterrows():
        old_peaks_by_sample[str(row["原始特征编号"])].append(
            {
                "row": row,
                "box": [float(row[col]) for col in PEAK_COORD_COLS],
                "label": str(row["原始标签"]),
                "peak_id": str(row["真峰编号"]),
            }
        )

    manifest_ids = [str(row["sample_id"]) for row in manifests]
    if len(manifest_ids) != 15317 or len(set(manifest_ids)) != 15317:
        raise RuntimeError(
            f"expected 15,317 unique manifest samples, got rows={len(manifest_ids)} unique={len(set(manifest_ids))}"
        )
    if set(manifest_ids) != set(seed_by_sample):
        missing_seed = sorted(set(manifest_ids) - set(seed_by_sample))[:20]
        missing_manifest = sorted(set(seed_by_sample) - set(manifest_ids))[:20]
        raise RuntimeError(
            f"manifest/Seed ID mismatch; missing_seed={missing_seed}, missing_manifest={missing_manifest}"
        )
    if len(review_shapes) != 150:
        raise RuntimeError(f"expected 150 reviewed samples, got {len(review_shapes)}")

    seed_jobs: list[dict[str, Any]] = []
    peak_jobs: list[dict[str, Any]] = []
    reviewed_manifest_box_count = 0
    reviewed_extra_out_fig_count = 0
    label_source_counts: Counter[str] = Counter()
    legacy_status_counts: Counter[str] = Counter()
    seed_attribute_max_abs_diff = {name: 0.0 for name in ATTR13}
    seed_attribute_missing_counts = {name: 0 for name in ATTR13}

    for manifest in manifests:
        sample_id = str(manifest["sample_id"])
        seed_row = seed_by_sample[sample_id]
        metadata = manifest.get("metadata") or {}
        attr_values = list(manifest.get("attributes") or [])
        if len(attr_values) != 13:
            raise RuntimeError(f"{sample_id}: expected 13 manifest attributes, got {len(attr_values)}")

        seed_job: dict[str, Any] = {
            "job_id": f"seed::{sample_id}",
            "entity_type": "seed",
            "sample_id": sample_id,
            "image_id": str(metadata.get("image_id") or seed_row["图片编号"]),
            "old_split": str(manifest["_old_split"]),
            "source_file": str(manifest.get("source_file") or ""),
            "source_mzml_name": source_mzml_name(manifest),
            "mz": float(metadata["mz"]),
            "RT": float(metadata["original_seed_rt"]),
            "RTmin": float(metadata["original_seed_left_rt"]),
            "RTmax": float(metadata["original_seed_right_rt"]),
            "eic_tolerance_ppm": float(metadata["eic_tolerance_ppm"]),
            "eic_method": "nearest",
            "tolerance_unit": "ppm",
            "apex_rule": "argmax_in_original_seed_window",
            "final_seed_label": int(manifest["seed_label"]),
            "final_detection_box_count": int(len(manifest.get("boxes") or [])),
            "reviewed_sample": sample_id in review_shapes,
        }
        for name, value in zip(ATTR13, attr_values):
            manifest_value = finite_float(value)
            table_value = finite_float(seed_row[f"原始{name}"])
            if (manifest_value is None) != (table_value is None):
                raise RuntimeError(
                    f"{sample_id}: manifest/old Seed table missing-state mismatch for {name}"
                )
            if manifest_value is None:
                seed_job[f"legacy_{name}"] = np.nan
                seed_attribute_missing_counts[name] += 1
            else:
                seed_job[f"legacy_{name}"] = manifest_value
                seed_attribute_max_abs_diff[name] = max(
                    seed_attribute_max_abs_diff[name], abs(manifest_value - table_value)
                )
        seed_jobs.append(seed_job)

        manifest_boxes = [normalize_box(box) for box in manifest.get("boxes") or []]
        reviewed = sample_id in review_shapes
        final_entities: list[dict[str, Any]] = []
        if reviewed:
            true_shapes = [shape for shape in review_shapes[sample_id] if shape["label"] == "True_Peak"]
            out_shapes = [shape for shape in review_shapes[sample_id] if shape["label"] == "OUT_FIG"]
            if len(manifest_boxes) != len(true_shapes):
                raise RuntimeError(
                    f"{sample_id}: v2 manifest boxes={len(manifest_boxes)} but reviewed True_Peak shapes={len(true_shapes)}"
                )
            assignments = assign_boxes(
                manifest_boxes,
                [shape["box"] for shape in true_shapes],
                min_iou=0.99,
            )
            if len(assignments) != len(manifest_boxes):
                raise RuntimeError(f"{sample_id}: failed to match all v2 boxes to reviewed True_Peak shapes")
            for index, box in enumerate(manifest_boxes):
                shape_index, match_iou, match_diff = assignments[index]
                final_entities.append(
                    {
                        "box": box,
                        "label": "True_Peak",
                        "included_in_v2_manifest": True,
                        "box_source": "reviewed_labelme_true_peak",
                        "review_json": true_shapes[shape_index]["review_json"],
                        "label_match_iou": match_iou,
                        "label_match_max_coord_diff_px": match_diff,
                    }
                )
            for shape in out_shapes:
                final_entities.append(
                    {
                        "box": shape["box"],
                        "label": "OUT_FIG",
                        "included_in_v2_manifest": False,
                        "box_source": "reviewed_labelme_out_fig_only",
                        "review_json": shape["review_json"],
                        "label_match_iou": 1.0,
                        "label_match_max_coord_diff_px": 0.0,
                    }
                )
                reviewed_extra_out_fig_count += 1
            reviewed_manifest_box_count += len(manifest_boxes)
        else:
            legacy_candidates = old_peaks_by_sample.get(sample_id, [])
            if len(manifest_boxes) != len(legacy_candidates):
                raise RuntimeError(
                    f"{sample_id}: unreviewed v2 boxes={len(manifest_boxes)} but legacy rows={len(legacy_candidates)}"
                )
            assignments = assign_boxes(
                manifest_boxes,
                [candidate["box"] for candidate in legacy_candidates],
                min_iou=0.999,
            )
            if len(assignments) != len(manifest_boxes):
                raise RuntimeError(f"{sample_id}: failed exact legacy matching for unreviewed sample")
            for index, box in enumerate(manifest_boxes):
                candidate_index, match_iou, match_diff = assignments[index]
                if match_diff > 1e-3:
                    raise RuntimeError(
                        f"{sample_id}: unreviewed box drifted by {match_diff:.6g} px"
                    )
                candidate = legacy_candidates[candidate_index]
                final_entities.append(
                    {
                        "box": box,
                        "label": candidate["label"],
                        "included_in_v2_manifest": True,
                        "box_source": "v2_manifest_matched_legacy",
                        "review_json": "",
                        "label_match_iou": match_iou,
                        "label_match_max_coord_diff_px": match_diff,
                    }
                )

        legacy_candidates = old_peaks_by_sample.get(sample_id, [])
        legacy_assignments = assign_boxes(
            [entity["box"] for entity in final_entities],
            [candidate["box"] for candidate in legacy_candidates],
            min_iou=0.05,
        )

        plot_xmin, plot_ymin, plot_xmax, plot_ymax = parse_plot_area(seed_row["绘图区坐标"])
        center_rt = float(seed_row["绘图中心RT"])
        for box_index, entity in enumerate(final_entities, start=1):
            box = entity["box"]
            rt_left = pixel_x_to_rt(box[0], plot_xmin, plot_xmax, center_rt)
            rt_right = pixel_x_to_rt(box[2], plot_xmin, plot_xmax, center_rt)
            if rt_right <= rt_left:
                raise RuntimeError(f"{sample_id}: non-positive final peak RT window")

            legacy: dict[str, Any] | None = None
            legacy_iou = np.nan
            legacy_diff = np.nan
            legacy_status = "new_no_legacy"
            if (box_index - 1) in legacy_assignments:
                legacy_index, legacy_iou, legacy_diff = legacy_assignments[box_index - 1]
                legacy = legacy_candidates[legacy_index]
                legacy_status = "exact" if legacy_diff <= 1e-3 else "adjusted"
            legacy_status_counts[legacy_status] += 1
            label_source_counts[str(entity["label"])] += 1

            legacy_apex = None
            if legacy is not None:
                legacy_apex = finite_float(legacy["row"].get("峰顶RT"))
            rt_hint = (
                legacy_apex
                if legacy_apex is not None and rt_left <= legacy_apex <= rt_right
                else (rt_left + rt_right) / 2.0
            )

            peak_job: dict[str, Any] = {
                "job_id": f"peak::{sample_id}::B{box_index:02d}",
                "entity_type": "peak_instance",
                "sample_id": sample_id,
                "image_id": str(metadata.get("image_id") or seed_row["图片编号"]),
                "old_split": str(manifest["_old_split"]),
                "source_file": str(manifest.get("source_file") or ""),
                "source_mzml_name": source_mzml_name(manifest),
                "mz": float(metadata["mz"]),
                "RT": float(rt_hint),
                "RTmin": float(rt_left),
                "RTmax": float(rt_right),
                "eic_tolerance_ppm": float(metadata["eic_tolerance_ppm"]),
                "eic_method": "nearest",
                "tolerance_unit": "ppm",
                "apex_rule": "argmax_in_final_human_box_rt_window",
                "final_label": str(entity["label"]),
                "final_box_index": box_index,
                "final_box_xmin_px": box[0],
                "final_box_ymin_px": box[1],
                "final_box_xmax_px": box[2],
                "final_box_ymax_px": box[3],
                "plot_xmin_px": plot_xmin,
                "plot_xmax_px": plot_xmax,
                "plot_center_rt": center_rt,
                "reviewed_sample": reviewed,
                "included_in_v2_manifest": bool(entity["included_in_v2_manifest"]),
                "box_source": str(entity["box_source"]),
                "review_json": str(entity["review_json"]),
                "label_match_iou": float(entity["label_match_iou"]),
                "label_match_max_coord_diff_px": float(entity["label_match_max_coord_diff_px"]),
                "legacy_peak_id": legacy["peak_id"] if legacy is not None else "",
                "legacy_label": legacy["label"] if legacy is not None else "",
                "legacy_match_status": legacy_status,
                "legacy_match_iou": legacy_iou,
                "legacy_match_max_coord_diff_px": legacy_diff,
                "attribute_policy": (
                    "preserve_legacy_13_add_SYM_MOD_EDGE"
                    if legacy is not None
                    else "compute_all_16_no_legacy_values"
                ),
            }
            for name in ATTR13:
                peak_job[f"legacy_{name}"] = (
                    finite_float(legacy["row"].get(name)) if legacy is not None else np.nan
                )
            peak_jobs.append(peak_job)

    seed_jobs_df = pd.DataFrame(seed_jobs)
    peak_jobs_df = pd.DataFrame(peak_jobs)

    if len(seed_jobs_df) != 15317 or seed_jobs_df["job_id"].nunique() != 15317:
        raise RuntimeError("Seed handoff count/uniqueness check failed")
    if len(peak_jobs_df) != 18580 or peak_jobs_df["job_id"].nunique() != 18580:
        raise RuntimeError(
            f"peak handoff expected 18,580 rows including five reviewed OUT_FIG-only shapes, got {len(peak_jobs_df)}"
        )
    final_label_counts = peak_jobs_df["final_label"].value_counts().to_dict()
    if final_label_counts != {"True_Peak": 18487, "OUT_FIG": 93}:
        raise RuntimeError(f"unexpected final label counts: {final_label_counts}")
    if reviewed_extra_out_fig_count != 5:
        raise RuntimeError(
            f"expected five review-only OUT_FIG shapes, got {reviewed_extra_out_fig_count}"
        )

    source_rows: list[dict[str, Any]] = []
    for source_name, group in seed_jobs_df.groupby("source_mzml_name", sort=True):
        peak_group = peak_jobs_df[peak_jobs_df["source_mzml_name"] == source_name]
        tolerances = sorted(set(float(x) for x in group["eic_tolerance_ppm"]))
        if len(tolerances) != 1:
            raise RuntimeError(f"{source_name}: expected one tolerance, got {tolerances}")
        source_rows.append(
            {
                "source_mzml_name": source_name,
                "seed_job_count": int(len(group)),
                "peak_job_count": int(len(peak_group)),
                "true_peak_job_count": int((peak_group["final_label"] == "True_Peak").sum()),
                "out_fig_job_count": int((peak_group["final_label"] == "OUT_FIG").sum()),
                "eic_tolerance_ppm": tolerances[0],
                "required": True,
            }
        )
    source_df = pd.DataFrame(source_rows)
    if len(source_df) != 29:
        raise RuntimeError(f"expected 29 mzML sources, got {len(source_df)}")

    seed_path = output_dir / "old_final_seed_attribute_jobs.csv"
    peak_path = output_dir / "old_final_peak_attribute_jobs.csv"
    source_path = output_dir / "source_inventory.csv"
    seed_jobs_df.to_csv(seed_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    peak_jobs_df.to_csv(peak_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    source_df.to_csv(source_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    manifest = {
        "format_version": 1,
        "created": "2026-08-14",
        "purpose": "Compute SYM, MOD and EDGE for old-final Seed candidates and complete 16 attributes for final human peak instances on the mzML-holding workstation.",
        "attribute_order": [*ATTR13, "SYM", "MOD", "EDGE"],
        "counts": {
            "seed_jobs": int(len(seed_jobs_df)),
            "peak_jobs_total": int(len(peak_jobs_df)),
            "peak_jobs_in_v2_manifest": int(peak_jobs_df["included_in_v2_manifest"].sum()),
            "review_only_out_fig_jobs": int((~peak_jobs_df["included_in_v2_manifest"]).sum()),
            "true_peak_jobs": int((peak_jobs_df["final_label"] == "True_Peak").sum()),
            "out_fig_jobs": int((peak_jobs_df["final_label"] == "OUT_FIG").sum()),
            "reviewed_samples": int(len(review_shapes)),
            "source_mzml_files": int(len(source_df)),
        },
        "legacy_match_status_counts": dict(sorted(legacy_status_counts.items())),
        "seed_manifest_vs_table_max_abs_diff": seed_attribute_max_abs_diff,
        "seed_legacy_missing_counts": seed_attribute_missing_counts,
        "old_final_inputs": {
            "seed_table": str(seed_table),
            "seed_table_sha256": sha256_file(seed_table),
            "peak_table": str(peak_table),
            "peak_table_sha256": sha256_file(peak_table),
            "v2_manifest_sha256": manifest_hashes,
            "review_json_count": len(review_shapes),
            "review_json_tree_sha256": review_tree_sha256(review_dir),
        },
        "outputs": {
            seed_path.name: sha256_file(seed_path),
            peak_path.name: sha256_file(peak_path),
            source_path.name: sha256_file(source_path),
        },
        "policies": {
            "seed_legacy_13": "authoritative and must not be overwritten; calculate only SYM, MOD and EDGE for final merge",
            "peak_with_legacy_match": "preserve legacy 13 in the final merged table; calculate SYM, MOD and EDGE on the final human box",
            "peak_without_legacy_match": "calculate all 16 because no legacy peak-instance values exist",
            "human_boundaries": "authoritative; never refine or replace",
            "review_only_out_fig": "retain and calculate even though five shapes were omitted from the v2 detector manifest",
        },
    }
    manifest_path = output_dir / "handoff_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"seed_jobs={seed_path} sha256={manifest['outputs'][seed_path.name]}")
    print(f"peak_jobs={peak_path} sha256={manifest['outputs'][peak_path.name]}")
    print(f"source_inventory={source_path} sha256={manifest['outputs'][source_path.name]}")
    print(f"manifest={manifest_path} sha256={sha256_file(manifest_path)}")


if __name__ == "__main__":
    main()
