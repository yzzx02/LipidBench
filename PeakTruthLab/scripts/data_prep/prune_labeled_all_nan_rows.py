from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ATTR_COLUMNS = ["SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG"]


def _backup_final_csv_if_needed(target_csv: Path, backup_dir: Path) -> Path | None:
    target_csv = target_csv.resolve()
    if target_csv.name != "feature_table_final_10000.csv":
        return None
    if not target_csv.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"{target_csv.stem}__backup_prune_nan_{ts}{target_csv.suffix}"
    shutil.copy2(target_csv, backup_path)
    return backup_path


def prune(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    images_root = Path(args.images_root).resolve()
    backup_dir = Path(args.backup_dir).resolve()
    report_json = Path(args.report_json).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    df = pd.read_csv(input_csv)
    miss = [c for c in ["Feature_ID", "source_file", "is_true_peak", *ATTR_COLUMNS] if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns: {miss}")

    targets = set()
    if args.only_source_files:
        targets = {x.strip() for x in str(args.only_source_files).split(",") if x.strip()}

    work = df[df["is_true_peak"].notna()].copy()
    mask = work[ATTR_COLUMNS].isna().all(axis=1)
    if targets:
        mask = mask & work["source_file"].astype(str).isin(targets)
    drop_ids = set(work.loc[mask, "Feature_ID"].astype(str))

    rows = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_root = backup_dir / f"removed_labeled_all_nan_{ts}"
    archive_root.mkdir(parents=True, exist_ok=True)

    for _, r in df[df["Feature_ID"].astype(str).isin(drop_ids)].iterrows():
        fid = str(r["Feature_ID"])
        stem = Path(str(r["source_file"])).stem
        png = images_root / stem / f"{fid}.png"
        js = images_root / stem / f"{fid}.json"
        archived = {}
        for src in [png, js]:
            if src.exists():
                dst = archive_root / stem / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                archived[src.suffix.lower()] = str(dst)
        rows.append(
            {
                "Feature_ID": fid,
                "source_file": str(r["source_file"]),
                "png_archived": archived.get(".png", ""),
                "json_archived": archived.get(".json", ""),
            }
        )

    out = df[~df["Feature_ID"].astype(str).isin(drop_ids)].copy().reset_index(drop=True)
    backup_path = _backup_final_csv_if_needed(output_csv, backup_dir)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    report = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "removed_rows": int(len(drop_ids)),
        "removed_feature_ids": sorted(drop_ids),
        "archive_root": str(archive_root),
        "rows": rows,
        "backup_csv": str(backup_path) if backup_path is not None else "",
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("done")
    print(f"input_csv:        {input_csv}")
    print(f"output_csv:       {output_csv}")
    print(f"removed_rows:     {len(drop_ids)}")
    print(f"archive_root:     {archive_root}")
    print(f"report_json:      {report_json}")
    if backup_path is not None:
        print(f"backup_csv:       {backup_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Remove labeled rows whose 13 peak attributes are all NaN")
    p.add_argument("--input-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--output-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--images-root", type=str, default="PeakTruthLab/datasets/eic_images_flat")
    p.add_argument("--backup-dir", type=str, default="PeakTruthLab/datasets/backups")
    p.add_argument("--report-json", type=str, default="PeakTruthLab/results/prune_labeled_all_nan_rows.json")
    p.add_argument("--only-source-files", type=str, default="", help="Optional comma-separated source_file names to limit pruning")
    return p.parse_args()


if __name__ == "__main__":
    prune(parse_args())
