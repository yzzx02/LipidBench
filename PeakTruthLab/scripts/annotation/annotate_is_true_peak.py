from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

DEFAULT_TRUE_LABELS = {
    "true_peak",
    "true peak",
    "truepeak",
    "tp",
    "true",
    "真峰",
    "高质量峰",
    "拖尾峰",
    "hq",
    "tailing",
}

DEFAULT_FALSE_LABELS = {
    "false_peak",
    "false peak",
    "falsepeak",
    "fp",
    "false",
    "假峰",
    "噪声",
    "noise",
    "artifact",
}


def _build_image_index(images_root: Path) -> dict[str, str]:
    if not images_root.exists():
        raise FileNotFoundError(f"images root not found: {images_root}")

    idx: dict[str, str] = {}
    for p in images_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem
        rel = p.relative_to(images_root).as_posix()
        if stem in idx:
            # Keep deterministic first path on name collisions.
            continue
        idx[stem] = rel
    return idx


def _normalize_label_value(v: Any) -> int | None:
    if pd.isna(v):
        return None

    s = str(v).strip().lower()
    if s == "":
        return None

    true_set = {"1", "true", "t", "yes", "y", "tp", "true_peak", "true-peak"}
    false_set = {"0", "false", "f", "no", "n", "fp", "false_peak", "false-peak"}

    if s in true_set:
        return 1
    if s in false_set:
        return 0

    try:
        num = float(s)
    except ValueError:
        return None

    if abs(num - 1.0) < 1e-9:
        return 1
    if abs(num - 0.0) < 1e-9:
        return 0
    return None


def _collect_feature_ids_from_image_dir(image_dir: Path) -> set[str]:
    if not image_dir.exists():
        return set()
    out: set[str] = set()
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.add(p.stem.strip())
    return out


def _normalize_label_name(s: str) -> str:
    t = str(s).strip().lower()
    t = t.replace("-", "_")
    return t


def _load_labelme_shape_labels(json_path: Path) -> list[str]:
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    shapes = obj.get("shapes")
    if not isinstance(shapes, list):
        return []
    out: list[str] = []
    for s in shapes:
        if not isinstance(s, dict):
            continue
        lab = s.get("label")
        if isinstance(lab, str) and lab.strip():
            out.append(_normalize_label_name(lab))
    return out


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _place_file(src: Path, dst: Path, mode: str) -> None:
    _ensure_parent(dst)
    if dst.exists():
        return
    try:
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "move":
            shutil.move(str(src), str(dst))
        elif mode == "hardlink":
            dst.hardlink_to(src)
        elif mode == "symlink":
            dst.symlink_to(src)
        else:
            raise ValueError(f"unsupported mode: {mode}")
    except FileExistsError:
        # Make reruns idempotent when destination has been created by a prior run.
        return


def split_by_labelme(args: argparse.Namespace) -> None:
    images_root = Path(args.images_root).resolve()
    json_root = Path(args.json_root).resolve() if args.json_root else images_root
    true_dir = Path(args.true_dir).resolve()
    false_dir = Path(args.false_dir).resolve()
    report_csv = Path(args.report_csv).resolve() if args.report_csv else None

    if not images_root.exists():
        raise FileNotFoundError(f"images root not found: {images_root}")
    if not json_root.exists():
        raise FileNotFoundError(f"json root not found: {json_root}")

    true_labels = {
        _normalize_label_name(x)
        for x in (args.true_labels.split(",") if args.true_labels else [])
        if x.strip()
    }
    false_labels = {
        _normalize_label_name(x)
        for x in (args.false_labels.split(",") if args.false_labels else [])
        if x.strip()
    }
    if not true_labels:
        true_labels = set(DEFAULT_TRUE_LABELS)
    if not false_labels:
        false_labels = set(DEFAULT_FALSE_LABELS)

    images = [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise RuntimeError(f"no images found under: {images_root}")

    rows: list[dict[str, Any]] = []
    n_true = 0
    n_false = 0
    n_missing_json = 0
    n_conflict = 0
    n_unresolved = 0

    for img in images:
        rel = img.relative_to(images_root)
        js = (json_root / rel).with_suffix(".json")
        status = ""
        target = ""
        labels: list[str] = []

        if not js.exists():
            n_missing_json += 1
            status = "missing_json"
        else:
            try:
                labels = _load_labelme_shape_labels(js)
            except Exception as e:
                status = f"json_error:{e}"
            else:
                has_true = any(l in true_labels for l in labels)
                has_false = any(l in false_labels for l in labels)
                if has_true and has_false:
                    n_conflict += 1
                    status = "conflict"
                elif has_true:
                    dst_img = true_dir / img.name
                    _place_file(img, dst_img, args.file_mode)
                    if args.with_json:
                        _place_file(js, (true_dir / js.name), args.file_mode)
                    n_true += 1
                    status = "true"
                    target = str(dst_img)
                elif has_false:
                    dst_img = false_dir / img.name
                    _place_file(img, dst_img, args.file_mode)
                    if args.with_json:
                        _place_file(js, (false_dir / js.name), args.file_mode)
                    n_false += 1
                    status = "false"
                    target = str(dst_img)
                elif len(labels) == 0 and args.empty_as_false:
                    dst_img = false_dir / img.name
                    _place_file(img, dst_img, args.file_mode)
                    if args.with_json:
                        _place_file(js, (false_dir / js.name), args.file_mode)
                    n_false += 1
                    status = "false_empty"
                    target = str(dst_img)
                else:
                    n_unresolved += 1
                    status = "unresolved"

        rows.append(
            {
                "image": str(img),
                "json": str(js),
                "labels": "|".join(labels),
                "status": status,
                "target": target,
            }
        )

    if report_csv:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(report_csv, index=False)

    print("done")
    print(f"images_root:        {images_root}")
    print(f"json_root:          {json_root}")
    print(f"true_dir:           {true_dir}")
    print(f"false_dir:          {false_dir}")
    print(f"file_mode:          {args.file_mode}")
    print(f"total_images:       {len(images)}")
    print(f"true_images:        {n_true}")
    print(f"false_images:       {n_false}")
    print(f"missing_json:       {n_missing_json}")
    print(f"conflict_labels:    {n_conflict}")
    print(f"unresolved_labels:  {n_unresolved}")
    if report_csv:
        print(f"report_csv:         {report_csv}")


def export_tasks(args: argparse.Namespace) -> None:
    feature_csv = Path(args.feature_csv).resolve()
    images_root = Path(args.images_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not feature_csv.exists():
        raise FileNotFoundError(f"feature csv not found: {feature_csv}")

    df = pd.read_csv(feature_csv)
    required = [args.id_col, args.label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in feature csv: {missing}")

    image_index = _build_image_index(images_root)

    if args.only_unlabeled:
        mask = df[args.label_col].isna() | (df[args.label_col].astype(str).str.strip() == "")
        df = df[mask].copy()

    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    missing_images = 0

    for _, row in df.iterrows():
        fid = str(row[args.id_col]).strip()
        if not fid:
            continue

        image_rel = image_index.get(fid, "")
        if image_rel == "":
            missing_images += 1

        rows.append(
            {
                "Feature_ID": fid,
                "image": image_rel,
                "is_true_peak": "",
                "annotator": "",
                "comment": "",
                "source_file": row.get("source_file", ""),
                "mz": row.get("mz", ""),
                "RT": row.get("RT", ""),
            }
        )

    out = pd.DataFrame(rows)
    if args.drop_missing_images:
        out = out[out["image"].astype(str).str.strip() != ""].copy()

    out.to_csv(out_csv, index=False)
    print("done")
    print(f"feature_csv:         {feature_csv}")
    print(f"images_root:         {images_root}")
    print(f"out_csv:             {out_csv}")
    print(f"rows_exported:       {len(out)}")
    print(f"rows_missing_images: {missing_images}")


def merge_labels(args: argparse.Namespace) -> None:
    feature_csv = Path(args.feature_csv).resolve()
    annotation_csv = Path(args.annotation_csv).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else feature_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not feature_csv.exists():
        raise FileNotFoundError(f"feature csv not found: {feature_csv}")
    if not annotation_csv.exists():
        raise FileNotFoundError(f"annotation csv not found: {annotation_csv}")

    feat = pd.read_csv(feature_csv)
    ann = pd.read_csv(annotation_csv)

    required_feat = [args.id_col, args.label_col]
    required_ann = [args.id_col, args.label_col]
    missing_feat = [c for c in required_feat if c not in feat.columns]
    missing_ann = [c for c in required_ann if c not in ann.columns]
    if missing_feat:
        raise ValueError(f"missing columns in feature csv: {missing_feat}")
    if missing_ann:
        raise ValueError(f"missing columns in annotation csv: {missing_ann}")

    ann = ann.copy()
    ann[args.id_col] = ann[args.id_col].astype(str).str.strip()
    ann = ann[ann[args.id_col] != ""].copy()

    ann["_norm_label"] = ann[args.label_col].map(_normalize_label_value)
    invalid = ann[ann["_norm_label"].isna()]
    valid = ann[~ann["_norm_label"].isna()].copy()

    if args.dedupe_last:
        valid = valid.drop_duplicates(subset=[args.id_col], keep="last")
    else:
        dup = valid[valid.duplicated(subset=[args.id_col], keep=False)]
        if not dup.empty:
            raise ValueError(
                f"annotation csv has duplicate {args.id_col} rows; pass --dedupe-last to keep the last one"
            )

    label_map = dict(zip(valid[args.id_col], valid["_norm_label"].astype(int)))

    feat = feat.copy()
    feat[args.id_col] = feat[args.id_col].astype(str).str.strip()

    updated = 0
    for i, fid in feat[args.id_col].items():
        if fid in label_map:
            feat.at[i, args.label_col] = int(label_map[fid])
            updated += 1

    feat.to_csv(out_csv, index=False)

    labeled = pd.to_numeric(feat[args.label_col], errors="coerce")
    labeled_valid = labeled[labeled.isin([0, 1])]
    n_true = int((labeled_valid == 1).sum())
    n_false = int((labeled_valid == 0).sum())

    print("done")
    print(f"feature_csv:          {feature_csv}")
    print(f"annotation_csv:       {annotation_csv}")
    print(f"out_csv:              {out_csv}")
    print(f"annotation_valid:     {len(valid)}")
    print(f"annotation_invalid:   {len(invalid)}")
    print(f"rows_updated:         {updated}")
    print(f"dataset_labeled_rows: {len(labeled_valid)}")
    print(f"dataset_true_rows:    {n_true}")
    print(f"dataset_false_rows:   {n_false}")

    if len(invalid) > 0 and args.invalid_out_csv:
        invalid_out = Path(args.invalid_out_csv).resolve()
        invalid_out.parent.mkdir(parents=True, exist_ok=True)
        invalid.to_csv(invalid_out, index=False)
        print(f"invalid_out_csv:      {invalid_out}")


def merge_labels_from_image_dirs(args: argparse.Namespace) -> None:
    feature_csv = Path(args.feature_csv).resolve()
    true_dir = Path(args.true_dir).resolve()
    false_dir = Path(args.false_dir).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else feature_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not feature_csv.exists():
        raise FileNotFoundError(f"feature csv not found: {feature_csv}")

    feat = pd.read_csv(feature_csv)
    required_feat = [args.id_col, args.label_col]
    missing_feat = [c for c in required_feat if c not in feat.columns]
    if missing_feat:
        raise ValueError(f"missing columns in feature csv: {missing_feat}")

    true_ids = _collect_feature_ids_from_image_dir(true_dir)
    false_ids = _collect_feature_ids_from_image_dir(false_dir)

    overlap = true_ids & false_ids
    if overlap and args.strict_no_overlap:
        raise ValueError(
            f"found {len(overlap)} duplicated Feature_ID in both true/false folders; example={next(iter(overlap))}"
        )

    # If overlap exists and not strict, true label wins by default.
    label_map: dict[str, int] = {fid: 0 for fid in false_ids}
    for fid in true_ids:
        label_map[fid] = 1

    feat = feat.copy()
    feat[args.id_col] = feat[args.id_col].astype(str).str.strip()

    updated = 0
    for i, fid in feat[args.id_col].items():
        if fid in label_map:
            feat.at[i, args.label_col] = int(label_map[fid])
            updated += 1

    feat.to_csv(out_csv, index=False)

    labeled = pd.to_numeric(feat[args.label_col], errors="coerce")
    labeled_valid = labeled[labeled.isin([0, 1])]
    n_true = int((labeled_valid == 1).sum())
    n_false = int((labeled_valid == 0).sum())

    print("done")
    print(f"feature_csv:          {feature_csv}")
    print(f"true_dir:             {true_dir}")
    print(f"false_dir:            {false_dir}")
    print(f"out_csv:              {out_csv}")
    print(f"true_folder_count:    {len(true_ids)}")
    print(f"false_folder_count:   {len(false_ids)}")
    print(f"folder_overlap_count: {len(overlap)}")
    print(f"rows_updated:         {updated}")
    print(f"dataset_labeled_rows: {len(labeled_valid)}")
    print(f"dataset_true_rows:    {n_true}")
    print(f"dataset_false_rows:   {n_false}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export/merge CSV workflow for manual is_true_peak annotation")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export annotation task CSV from feature table")
    p_export.add_argument(
        "--feature-csv",
        type=str,
        default="PeakTruthLab/datasets/feature_table_final_10000.csv",
        help="Feature table CSV",
    )
    p_export.add_argument(
        "--images-root",
        type=str,
        default="PeakTruthLab/datasets/eic_images_pool",
        help="Root folder of EIC images",
    )
    p_export.add_argument(
        "--out-csv",
        type=str,
        default="PeakTruthLab/datasets/annotation_tasks_10000.csv",
        help="Output task CSV for manual labeling",
    )
    p_export.add_argument("--id-col", type=str, default="Feature_ID")
    p_export.add_argument("--label-col", type=str, default="is_true_peak")
    p_export.add_argument("--only-unlabeled", action="store_true", help="Export only unlabeled rows")
    p_export.add_argument("--drop-missing-images", action="store_true", help="Drop rows with no matched image")
    p_export.add_argument("--max-rows", type=int, default=0, help="Optional cap on exported rows (0 means all)")
    p_export.add_argument("--seed", type=int, default=42)

    p_merge = sub.add_parser("merge", help="Merge annotation CSV back into feature table")
    p_merge.add_argument(
        "--feature-csv",
        type=str,
        default="PeakTruthLab/datasets/feature_table_final_10000.csv",
        help="Feature table CSV",
    )
    p_merge.add_argument(
        "--annotation-csv",
        type=str,
        default="PeakTruthLab/datasets/annotation_tasks_10000.csv",
        help="CSV with manual labels",
    )
    p_merge.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Output feature table CSV (default: overwrite feature-csv)",
    )
    p_merge.add_argument("--id-col", type=str, default="Feature_ID")
    p_merge.add_argument("--label-col", type=str, default="is_true_peak")
    p_merge.add_argument("--dedupe-last", action="store_true", help="Keep last label when Feature_ID appears multiple times")
    p_merge.add_argument(
        "--invalid-out-csv",
        type=str,
        default="PeakTruthLab/datasets/annotation_invalid_rows.csv",
        help="Where to save annotation rows with invalid label values",
    )

    p_merge_img = sub.add_parser(
        "merge-image-dirs",
        help="Merge labels from image folders back into feature table (image-first workflow)",
    )
    p_merge_img.add_argument(
        "--feature-csv",
        type=str,
        default="PeakTruthLab/datasets/feature_table_final_10000.csv",
        help="Feature table CSV",
    )
    p_merge_img.add_argument(
        "--true-dir",
        type=str,
        default="PeakTruthLab/datasets/annotation_by_image/true_peak",
        help="Folder containing images judged as true peaks",
    )
    p_merge_img.add_argument(
        "--false-dir",
        type=str,
        default="PeakTruthLab/datasets/annotation_by_image/false_peak",
        help="Folder containing images judged as false peaks",
    )
    p_merge_img.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Output feature table CSV (default: overwrite feature-csv)",
    )
    p_merge_img.add_argument("--id-col", type=str, default="Feature_ID")
    p_merge_img.add_argument("--label-col", type=str, default="is_true_peak")
    p_merge_img.add_argument(
        "--strict-no-overlap",
        action="store_true",
        help="Fail if same Feature_ID appears in both true and false folders",
    )

    p_split = sub.add_parser(
        "split-from-labelme",
        help="Split images into true/false folders using LabelMe JSON labels",
    )
    p_split.add_argument(
        "--images-root",
        type=str,
        default="PeakTruthLab/datasets/eic_images_flat",
        help="Root folder of images (recursive)",
    )
    p_split.add_argument(
        "--json-root",
        type=str,
        default="",
        help="Root folder of json files mirroring images-root (default: same as images-root)",
    )
    p_split.add_argument(
        "--true-dir",
        type=str,
        default="PeakTruthLab/datasets/annotation_by_image/true_peak",
        help="Output folder for images labeled as true peak",
    )
    p_split.add_argument(
        "--false-dir",
        type=str,
        default="PeakTruthLab/datasets/annotation_by_image/false_peak",
        help="Output folder for images labeled as false peak",
    )
    p_split.add_argument(
        "--true-labels",
        type=str,
        default=",".join(sorted(DEFAULT_TRUE_LABELS)),
        help="Comma-separated labels to be considered true peak",
    )
    p_split.add_argument(
        "--false-labels",
        type=str,
        default=",".join(sorted(DEFAULT_FALSE_LABELS)),
        help="Comma-separated labels to be considered false peak",
    )
    p_split.add_argument(
        "--file-mode",
        type=str,
        default="hardlink",
        choices=["copy", "move", "hardlink", "symlink"],
        help="How to place files into true/false folders",
    )
    p_split.add_argument("--with-json", action="store_true", help="Also place same-name json to true/false folder")
    p_split.add_argument(
        "--report-csv",
        type=str,
        default="PeakTruthLab/datasets/labelme_split_report.csv",
        help="Detailed per-image split report CSV",
    )
    p_split.add_argument(
        "--empty-as-false",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat JSON with empty shapes as False_Peak (default: true)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "export":
        export_tasks(args)
    elif args.cmd == "merge":
        merge_labels(args)
    elif args.cmd == "merge-image-dirs":
        merge_labels_from_image_dirs(args)
    elif args.cmd == "split-from-labelme":
        split_by_labelme(args)
    else:
        raise ValueError(f"unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
