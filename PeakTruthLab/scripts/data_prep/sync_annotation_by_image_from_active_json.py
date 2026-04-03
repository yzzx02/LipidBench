from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def _build_file_index(root: Path, suffix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob(f"*{suffix}"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if any(str(part).startswith(".") for part in rel_parts):
            continue
        out.setdefault(p.stem, p)
    return out


def _parse_exact_label_set(v: str) -> set[str]:
    return {x.strip() for x in str(v).split(",") if x.strip()}


def _clean_label(v: str) -> str:
    return str(v).strip()


def _json_state(path: Path, *, delete_labels: set[str], uncertain_labels: set[str]) -> str:
    obj = json.loads(path.read_text(encoding="utf-8"))
    shapes = obj.get("shapes", [])
    if not isinstance(shapes, list) or len(shapes) == 0:
        return "no_box"
    labels = {_clean_label(s.get("label", "")) for s in shapes if isinstance(s, dict) and _clean_label(s.get("label", ""))}
    if labels & delete_labels:
        return "delete"
    if labels & uncertain_labels:
        return "uncertain"
    return "box"


def sync(args: argparse.Namespace) -> None:
    csv_path = Path(args.input_csv).resolve()
    active_root = Path(args.images_root).resolve()
    reserved_root = Path(args.reserved_images_root).resolve()
    annotation_root = Path(args.annotation_root).resolve()
    backup_dir = Path(args.backup_dir).resolve()
    report_csv = Path(args.report_csv).resolve()
    enable_backup = bool(args.enable_backup)
    delete_labels = _parse_exact_label_set(args.delete_labels)
    uncertain_labels = _parse_exact_label_set(args.uncertain_labels)

    overlap_labels = sorted(delete_labels & uncertain_labels)
    if overlap_labels:
        raise ValueError(f"delete-labels and uncertain-labels overlap: {overlap_labels}")

    df = pd.read_csv(csv_path)
    csv_ids = set(df["Feature_ID"].astype(str))

    active_png = _build_file_index(active_root, ".png")
    active_json = _build_file_index(active_root, ".json")
    reserved_png = _build_file_index(reserved_root, ".png")
    reserved_json = _build_file_index(reserved_root, ".json")

    active_ids = sorted(set(active_png) & set(active_json) & csv_ids)
    reserved_ids = sorted(set(reserved_png) & set(reserved_json) & csv_ids)

    state_rows: list[dict[str, str]] = []
    true_ids: list[str] = []
    false_ids: list[str] = []
    delete_ids: list[str] = []
    uncertain_ids: list[str] = []

    for fid in active_ids:
        state = _json_state(
            active_json[fid],
            delete_labels=delete_labels,
            uncertain_labels=uncertain_labels,
        )
        row = {
            "Feature_ID": fid,
            "state": state,
            "png_path": str(active_png[fid]),
            "json_path": str(active_json[fid]),
        }
        state_rows.append(row)
        if state == "box":
            true_ids.append(fid)
        elif state == "no_box":
            false_ids.append(fid)
        elif state == "delete":
            delete_ids.append(fid)
        elif state == "uncertain":
            uncertain_ids.append(fid)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_root = backup_dir / f"annotation_by_image_sync_backup_{ts}"
    true_dir = annotation_root / "true_peak"
    false_dir = annotation_root / "false_peak"

    if not args.dry_run:
        if enable_backup:
            backup_root.mkdir(parents=True, exist_ok=True)
            if true_dir.exists():
                shutil.move(str(true_dir), str(backup_root / "true_peak"))
            if false_dir.exists():
                shutil.move(str(false_dir), str(backup_root / "false_peak"))
        else:
            if true_dir.exists():
                shutil.rmtree(true_dir)
            if false_dir.exists():
                shutil.rmtree(false_dir)
        true_dir.mkdir(parents=True, exist_ok=True)
        false_dir.mkdir(parents=True, exist_ok=True)

        for fid in true_ids:
            shutil.copy2(active_png[fid], true_dir / f"{fid}.png")
        for fid in false_ids:
            shutil.copy2(active_png[fid], false_dir / f"{fid}.png")

    report_df = pd.DataFrame(state_rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)

    print("done")
    print(f"input_csv:            {csv_path}")
    print(f"images_root:          {active_root}")
    print(f"reserved_images_root: {reserved_root}")
    print(f"annotation_root:      {annotation_root}")
    print(f"report_csv:           {report_csv}")
    print(f"dry_run:              {bool(args.dry_run)}")
    print(f"active_total:         {len(active_ids)}")
    print(f"enable_backup:        {enable_backup}")
    print(f"delete_labels:        {sorted(delete_labels)}")
    print(f"uncertain_labels:     {sorted(uncertain_labels)}")
    print(f"true_peak_pngs:       {len(true_ids)}")
    print(f"false_peak_pngs:      {len(false_ids)}")
    print(f"delete_ids_skipped:   {len(delete_ids)}")
    print(f"uncertain_ids_skipped:{len(uncertain_ids)}")
    print(f"reserved_skipped:     {len(reserved_ids)}")
    if not args.dry_run and enable_backup:
        print(f"backup_root:          {backup_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sync annotation_by_image from eic_images_flat JSON state")
    p.add_argument("--input-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--images-root", type=str, default="PeakTruthLab/datasets/eic_images_flat")
    p.add_argument("--reserved-images-root", type=str, default="PeakTruthLab/datasets/eic_images_flat_2")
    p.add_argument("--annotation-root", type=str, default="PeakTruthLab/datasets/annotation_by_image")
    p.add_argument("--backup-dir", type=str, default="PeakTruthLab/datasets/backups")
    p.add_argument("--enable-backup", action="store_true")
    p.add_argument("--report-csv", type=str, default="PeakTruthLab/results/annotation_by_image_sync_report.csv")
    p.add_argument("--delete-labels", type=str, default="D")
    p.add_argument("--uncertain-labels", type=str, default="d")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    sync(parse_args())
