from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def _backup_final_csv_if_needed(target_csv: Path, backup_dir: Path) -> Path | None:
    target_csv = target_csv.resolve()
    if target_csv.name != "feature_table_final_10000.csv":
        return None
    if not target_csv.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"{target_csv.stem}__backup_batch_merge_{ts}{target_csv.suffix}"
    shutil.copy2(target_csv, backup_path)
    return backup_path


def _parse_targets(v: str) -> set[str]:
    return {x.strip() for x in str(v).split(",") if x.strip()}


def merge(args: argparse.Namespace) -> None:
    batch_root = Path(args.batch_root).resolve()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    images_root = Path(args.images_root).resolve()
    backup_dir = Path(args.backup_dir).resolve()
    report_json = Path(args.report_json).resolve()

    features_csv = batch_root / "sampled_features_random550_per_file.csv"
    batch_images_root = batch_root / "eic_images"

    if not batch_root.exists():
        raise FileNotFoundError(f"batch_root not found: {batch_root}")
    if not features_csv.exists():
        raise FileNotFoundError(f"features csv not found: {features_csv}")
    if not batch_images_root.exists():
        raise FileNotFoundError(f"batch images root not found: {batch_images_root}")
    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    final_df = pd.read_csv(input_csv)
    batch_df = pd.read_csv(features_csv)

    if "Feature_ID" not in final_df.columns:
        raise ValueError("input csv missing Feature_ID")
    if "Feature_ID" not in batch_df.columns:
        raise ValueError("batch csv missing Feature_ID")
    if "source_file" not in batch_df.columns:
        raise ValueError("batch csv missing source_file")

    targets = _parse_targets(args.only_source_files)
    if targets:
        batch_df = batch_df[batch_df["source_file"].astype(str).isin(targets)].copy()
        if batch_df.empty:
            raise ValueError(f"no rows left after --only-source-files filter: {sorted(targets)}")

    batch_df["Feature_ID"] = batch_df["Feature_ID"].astype(str)
    final_df["Feature_ID"] = final_df["Feature_ID"].astype(str)

    dup_in_batch = batch_df["Feature_ID"].duplicated()
    if dup_in_batch.any():
        dup_id = batch_df.loc[dup_in_batch, "Feature_ID"].iloc[0]
        raise ValueError(f"duplicate Feature_ID in batch csv: {dup_id}")

    overlap = sorted(set(final_df["Feature_ID"]) & set(batch_df["Feature_ID"]))
    if overlap:
        raise ValueError(f"batch contains existing Feature_ID values, first overlap: {overlap[0]}")

    out_cols = list(final_df.columns)
    add = pd.DataFrame(index=range(len(batch_df)), columns=out_cols)
    for c in out_cols:
        if c in batch_df.columns:
            add[c] = batch_df[c].values
        elif c == "is_true_peak":
            add[c] = pd.NA

    source_files = sorted(batch_df["source_file"].astype(str).unique())
    stems = sorted({Path(x).stem for x in source_files})

    copy_rows: list[dict[str, str | int | bool]] = []
    for stem in stems:
        src_dir = batch_images_root / stem
        dst_dir = images_root / stem
        if not src_dir.exists():
            raise FileNotFoundError(f"missing source image dir: {src_dir}")
        if dst_dir.exists():
            raise FileExistsError(f"target image dir already exists: {dst_dir}")
        png_count = len(list(src_dir.glob("*.png")))
        json_count = len(list(src_dir.glob("*.json")))
        if png_count == 0 or json_count == 0:
            raise ValueError(f"empty image/json dir: {src_dir}")
        shutil.copytree(src_dir, dst_dir)
        copy_rows.append(
            {
                "stem": stem,
                "src_dir": str(src_dir),
                "dst_dir": str(dst_dir),
                "png_count": png_count,
                "json_count": json_count,
            }
        )

    out_df = pd.concat([final_df, add], ignore_index=True)
    out_df = out_df.drop_duplicates(subset=["Feature_ID"], keep="first").reset_index(drop=True)

    backup_path = _backup_final_csv_if_needed(output_csv, backup_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    report = {
        "batch_root": str(batch_root),
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "batch_rows_added": int(len(add)),
        "source_files": source_files,
        "stems_copied": stems,
        "copy_rows": copy_rows,
        "final_rows_after_merge": int(len(out_df)),
        "backup_csv": str(backup_path) if backup_path is not None else "",
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print(f"batch_root:         {batch_root}")
    print(f"input_csv:          {input_csv}")
    print(f"output_csv:         {output_csv}")
    print(f"batch_rows_added:   {len(add)}")
    print(f"stems_copied:       {len(stems)}")
    print(f"final_rows_after:   {len(out_df)}")
    print(f"report_json:        {report_json}")
    if backup_path is not None:
        print(f"backup_csv:         {backup_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Merge an isolated annotation batch into the active dataset and image root")
    p.add_argument("--batch-root", type=str, required=True)
    p.add_argument("--input-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--output-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--images-root", type=str, default="PeakTruthLab/datasets/eic_images_flat")
    p.add_argument("--backup-dir", type=str, default="PeakTruthLab/datasets/backups")
    p.add_argument("--report-json", type=str, default="PeakTruthLab/results/merge_annotation_batch_report.json")
    p.add_argument("--only-source-files", type=str, default="", help="Optional comma-separated source_file names to merge")
    return p.parse_args()


if __name__ == "__main__":
    merge(parse_args())
