from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.utils.peak_attributes import (
    LITERATURE_TOP_COLUMNS,
    _compute_one_feature_attributes,
    _extract_trace,
    load_ms1_spectra,
)
from lipidbench.utils.rt_boundary_refiner import refine_peak_boundaries


def _backup_final_csv_if_needed(target_csv: Path, backup_dir: Path) -> Path | None:
    target_csv = target_csv.resolve()
    if target_csv.name != "feature_table_final_10000.csv":
        return None
    if not target_csv.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = backup_dir / f"{target_csv.stem}__backup_{ts}{target_csv.suffix}"
    shutil.copy2(target_csv, backup_path)
    return backup_path


def _parse_source_filter(v: str) -> set[str]:
    if not v.strip():
        return set()
    return {x.strip() for x in v.split(",") if x.strip()}


def recompute(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    report_csv = Path(args.report_csv).resolve() if args.report_csv else None
    backup_dir = Path(args.backup_dir).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")

    df = pd.read_csv(input_csv)
    req = {"Feature_ID", "source_path", "source_file", "mz", "RT", "RTmin", "RTmax"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"input csv missing columns: {miss}")

    source_filter = _parse_source_filter(args.only_source_files)

    mask = pd.Series(True, index=df.index)
    if source_filter:
        mask &= df["source_file"].astype(str).isin(source_filter)
    if args.feature_id:
        mask &= df["Feature_ID"].astype(str) == str(args.feature_id)

    work = df.loc[mask].copy()
    if work.empty:
        raise RuntimeError("no rows selected by filters")

    work["mz"] = pd.to_numeric(work["mz"], errors="coerce")
    work["RT"] = pd.to_numeric(work["RT"], errors="coerce")
    work["RTmin"] = pd.to_numeric(work["RTmin"], errors="coerce")
    work["RTmax"] = pd.to_numeric(work["RTmax"], errors="coerce")
    work = work.dropna(subset=["mz", "RT"]).copy()

    updates: dict[int, dict[str, Any]] = {}
    report_rows: list[dict[str, Any]] = []

    for source_path, sub in work.groupby("source_path", sort=False):
        mzml_path = Path(str(source_path))
        if not mzml_path.exists():
            print(f"[WARN] mzML missing, skip source: {mzml_path}")
            continue

        print(f"[LOAD] {mzml_path.name}: rows={len(sub)}")
        spectra = load_ms1_spectra(mzml_path)
        if not spectra:
            print(f"[WARN] no MS1 spectra: {mzml_path}")
            continue

        for idx, row in sub.iterrows():
            mz = float(row["mz"])
            rt_hint = float(row["RT"])
            old_rtmin = float(row["RTmin"]) if pd.notna(row["RTmin"]) else np.nan
            old_rtmax = float(row["RTmax"]) if pd.notna(row["RTmax"]) else np.nan

            rt_arr, eic_arr, mass_arr = _extract_trace(
                spectra,
                target_mz=mz,
                tolerance=float(args.mz_tolerance),
                unit=str(args.tolerance_unit),
                method=str(args.method),
            )

            refined = refine_peak_boundaries(
                rt_arr,
                eic_arr,
                rt_hint,
                rtmin_hint=(None if pd.isna(row["RTmin"]) else float(row["RTmin"])),
                rtmax_hint=(None if pd.isna(row["RTmax"]) else float(row["RTmax"])),
                search_half_window_min=float(args.search_half_window_min),
                local_half_window_min=float(args.local_half_window_min),
                sigma_mult=float(args.sigma_mult),
                min_rel_height=float(args.min_rel_height),
                smooth_window_scans=int(args.smooth_window_scans),
                smooth_passes=int(args.smooth_passes),
                confirm_scans=int(args.boundary_confirm),
                max_expand_scans=int(args.max_expand_scans),
                max_expand_min=float(args.max_expand_min),
                rise_rel_tol=float(args.rise_rel_tol),
                rebound_rel=float(args.rebound_rel),
                rise_patience=int(args.rise_patience),
                oversize_factor=float(args.oversize_factor),
            )
            rt_outside_refined = not (float(refined.rtmin) <= float(rt_hint) <= float(refined.rtmax))
            if args.update_rt_mode == "always":
                center_rt = float(refined.apex_rt)
            elif args.update_rt_mode == "when_outside_bounds" and (
                (not bool(refined.old_rt_in_bounds)) or rt_outside_refined
            ):
                center_rt = float(refined.apex_rt)
            else:
                center_rt = float(rt_hint)

            attrs = _compute_one_feature_attributes(
                rt_arr,
                eic_arr,
                mass_arr,
                target_mz=mz,
                target_rt_min=center_rt,
                target_rtmin=float(refined.rtmin),
                target_rtmax=float(refined.rtmax),
                rt_tol_sec=float(args.rt_tol_sec),
                include_literature_top=True,
            )

            upd = {
                "RTmin": round(float(refined.rtmin), 6),
                "RTmax": round(float(refined.rtmax), 6),
            }
            if args.update_rt_mode == "always" or (
                args.update_rt_mode == "when_outside_bounds" and (
                    (not bool(refined.old_rt_in_bounds)) or rt_outside_refined
                )
            ):
                upd["RT"] = round(float(refined.apex_rt), 6)
            for c in LITERATURE_TOP_COLUMNS:
                upd[c] = attrs.get(c, np.nan)
            updates[int(idx)] = upd

            report_rows.append(
                {
                    "row_index": int(idx),
                    "Feature_ID": str(row["Feature_ID"]),
                    "source_file": str(row["source_file"]),
                    "mz": mz,
                    "old_RT": float(row["RT"]),
                    "new_RT": float(upd.get("RT", row["RT"])),
                    "old_RTmin": old_rtmin,
                    "new_RTmin": float(refined.rtmin),
                    "old_RTmax": old_rtmax,
                    "new_RTmax": float(refined.rtmax),
                    "new_width_sec": float(refined.width_sec),
                    "apex_RT": float(refined.apex_rt),
                    "apex_idx": int(refined.apex_idx),
                    "left_idx": int(refined.left_idx),
                    "right_idx": int(refined.right_idx),
                    "status": str(refined.status),
                    "bound_mode": str(refined.bound_mode),
                    "old_rt_in_bounds": bool(refined.old_rt_in_bounds),
                    "old_rt_outside_refined": bool(rt_outside_refined),
                    "rt_recentred": bool(refined.rt_recentred),
                    "baseline": float(refined.baseline),
                    "noise_sigma": float(refined.noise_sigma),
                    "threshold": float(refined.threshold),
                    "af": float(refined.af),
                    "ff": float(refined.ff),
                    "sf": float(refined.sf),
                    "left_expand_scans": int(refined.left_expand_scans),
                    "right_expand_scans": int(refined.right_expand_scans),
                    "left_rebound_stop": bool(refined.left_rebound_stop),
                    "right_rebound_stop": bool(refined.right_rebound_stop),
                    "oversized_shrink": bool(refined.oversized_shrink),
                }
            )

    if not updates:
        raise RuntimeError("no updates produced (check filters / source paths)")

    for i, upd in updates.items():
        for k, v in upd.items():
            df.at[i, k] = v

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    backup_path = _backup_final_csv_if_needed(output_csv, backup_dir)
    df.to_csv(output_csv, index=False)

    print("done")
    print(f"input_csv:      {input_csv}")
    print(f"output_csv:     {output_csv}")
    print(f"updated_rows:    {len(updates)}")

    if report_csv:
        rep = pd.DataFrame(report_rows)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(report_csv, index=False)
        print(f"report_csv:      {report_csv}")
    if backup_path is not None:
        print(f"backup_csv:      {backup_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Recompute RTmin/RTmax from mz+RT and recompute 13 literature peak attributes"
    )
    p.add_argument(
        "--input-csv",
        type=str,
        default="PeakTruthLab/datasets/feature_table_final_10000.csv",
        help="Input feature table CSV",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default="PeakTruthLab/datasets/feature_table_final_10000.csv",
        help="Output CSV (can be same as input for in-place overwrite)",
    )
    p.add_argument(
        "--report-csv",
        type=str,
        default="PeakTruthLab/results/rt_bounds_recompute_report.csv",
        help="Report CSV with before/after RT bounds",
    )
    p.add_argument(
        "--backup-dir",
        type=str,
        default="PeakTruthLab/datasets/backups",
        help="Backup directory used before overwriting feature_table_final_10000.csv",
    )
    p.add_argument(
        "--feature-id",
        type=str,
        default="",
        help="Optional single Feature_ID to process",
    )
    p.add_argument(
        "--only-source-files",
        type=str,
        default="",
        help="Optional comma list, e.g. 'Blood-30V.mzML,HepG2-30V.mzML'",
    )

    p.add_argument("--mz-tolerance", type=float, default=15.0)
    p.add_argument("--tolerance-unit", type=str, default="ppm", choices=["ppm", "Da"])
    p.add_argument("--method", type=str, default="nearest", choices=["nearest", "window_sum"])

    p.add_argument("--search-half-window-min", type=float, default=0.35)
    p.add_argument("--local-half-window-min", type=float, default=1.0)
    p.add_argument("--sigma-mult", type=float, default=3.0)
    p.add_argument("--min-rel-height", type=float, default=0.02)
    p.add_argument("--smooth-window-scans", type=int, default=5, help="LWMA smoothing window in scans")
    p.add_argument("--smooth-passes", type=int, default=2, help="How many smoothing passes to apply")
    p.add_argument(
        "--expand-scans",
        "--max-expand-scans",
        dest="max_expand_scans",
        type=int,
        default=18,
        help="Maximum scans used when extending a core boundary to a local minimum",
    )
    p.add_argument("--max-expand-min", type=float, default=0.35, help="Maximum RT extension from core boundary in minutes")
    p.add_argument("--rise-patience", type=int, default=2, help="Allow short spike-like rises before stopping expansion")
    p.add_argument("--rise-rel-tol", type=float, default=0.02, help="Treat tiny rises as still descending")
    p.add_argument("--rebound-rel", type=float, default=0.12, help="Rollback to a local minimum if a rebound appears farther away")
    p.add_argument("--boundary-confirm", type=int, default=2, help="Need this many baseline-like scans to confirm a core boundary")
    p.add_argument("--oversize-factor", type=float, default=1.8, help="Shrink to core boundary if extended width is too large")
    p.add_argument(
        "--update-rt-mode",
        type=str,
        default="when_outside_bounds",
        choices=["never", "when_outside_bounds", "always"],
        help="Whether to update RT to the refined apex RT",
    )
    p.add_argument(
        "--rt-tol-sec",
        type=float,
        default=30.0,
        help="Fallback only when RTmin/RTmax unavailable in attribute calculation",
    )

    args = p.parse_args()
    if args.mz_tolerance <= 0:
        raise ValueError("--mz-tolerance must be > 0")
    if args.search_half_window_min <= 0:
        raise ValueError("--search-half-window-min must be > 0")
    if args.local_half_window_min <= 0:
        raise ValueError("--local-half-window-min must be > 0")
    if args.smooth_window_scans <= 0:
        raise ValueError("--smooth-window-scans must be > 0")
    if args.smooth_passes <= 0:
        raise ValueError("--smooth-passes must be > 0")
    if args.max_expand_scans < 0:
        raise ValueError("--max-expand-scans must be >= 0")
    if args.max_expand_min < 0:
        raise ValueError("--max-expand-min must be >= 0")
    if args.rise_patience < 0:
        raise ValueError("--rise-patience must be >= 0")
    if args.rise_rel_tol < 0:
        raise ValueError("--rise-rel-tol must be >= 0")
    if args.rebound_rel < 0:
        raise ValueError("--rebound-rel must be >= 0")
    if args.boundary_confirm <= 0:
        raise ValueError("--boundary-confirm must be > 0")
    if args.oversize_factor <= 1.0:
        raise ValueError("--oversize-factor must be > 1.0")
    if args.min_rel_height <= 0 or args.min_rel_height >= 1:
        raise ValueError("--min-rel-height must be in (0,1)")
    return args


if __name__ == "__main__":
    recompute(parse_args())
