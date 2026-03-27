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


def _robust_baseline_sigma(x: np.ndarray) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 0.0
    baseline = float(np.median(x))
    mad = float(np.median(np.abs(x - baseline)))
    sigma = 1.4826 * mad
    return baseline, sigma


def _estimate_bounds(
    rt: np.ndarray,
    eic: np.ndarray,
    rt_hint: float,
    *,
    search_half_window_min: float,
    local_half_window_min: float,
    sigma_mult: float,
    min_rel_height: float,
    expand_scans: int,
) -> tuple[float, float, float, float, int, int]:
    if rt.size == 0 or eic.size == 0 or rt.size != eic.size:
        return float(rt_hint), float(rt_hint), float(rt_hint), 0.0, -1, -1

    y = np.asarray(eic, dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    y[y < 0] = 0.0

    in_search = (rt >= (rt_hint - search_half_window_min)) & (rt <= (rt_hint + search_half_window_min))
    if np.any(in_search):
        cand_idx = np.where(in_search)[0]
        apex_idx = int(cand_idx[np.argmax(y[cand_idx])])
    else:
        apex_idx = int(np.argmin(np.abs(rt - rt_hint)))

    apex_rt = float(rt[apex_idx])
    apex_int = float(y[apex_idx])

    in_local = (rt >= (apex_rt - local_half_window_min)) & (rt <= (apex_rt + local_half_window_min))
    local_y = y[in_local] if np.any(in_local) else y
    baseline, sigma = _robust_baseline_sigma(local_y)

    thr_noise = baseline + sigma_mult * sigma
    thr_rel = apex_int * float(min_rel_height)
    threshold = max(thr_noise, thr_rel)

    if apex_int <= 0:
        return apex_rt, apex_rt, apex_rt, 0.0, apex_idx, apex_idx

    if threshold >= apex_int:
        threshold = apex_int * 0.5

    left = apex_idx
    right = apex_idx

    while left > 0 and y[left - 1] >= threshold:
        left -= 1
    while right < (len(y) - 1) and y[right + 1] >= threshold:
        right += 1

    if expand_scans > 0:
        left = max(0, left - int(expand_scans))
        right = min(len(y) - 1, right + int(expand_scans))

    rtmin = float(rt[left])
    rtmax = float(rt[right])
    width_sec = float(max(rtmax - rtmin, 0.0) * 60.0)
    return rtmin, rtmax, apex_rt, width_sec, left, right


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

            new_rtmin, new_rtmax, apex_rt, width_sec, li, ri = _estimate_bounds(
                rt_arr,
                eic_arr,
                rt_hint,
                search_half_window_min=float(args.search_half_window_min),
                local_half_window_min=float(args.local_half_window_min),
                sigma_mult=float(args.sigma_mult),
                min_rel_height=float(args.min_rel_height),
                expand_scans=int(args.expand_scans),
            )

            center_rt = apex_rt if args.use_apex_as_rt else rt_hint

            attrs = _compute_one_feature_attributes(
                rt_arr,
                eic_arr,
                mass_arr,
                target_mz=mz,
                target_rt_min=center_rt,
                target_rtmin=new_rtmin,
                target_rtmax=new_rtmax,
                rt_tol_sec=float(args.rt_tol_sec),
                include_literature_top=True,
            )

            upd = {
                "RTmin": round(float(new_rtmin), 6),
                "RTmax": round(float(new_rtmax), 6),
            }
            if args.use_apex_as_rt:
                upd["RT"] = round(float(apex_rt), 6)
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
                    "new_RTmin": float(new_rtmin),
                    "old_RTmax": old_rtmax,
                    "new_RTmax": float(new_rtmax),
                    "new_width_sec": width_sec,
                    "left_idx": int(li),
                    "right_idx": int(ri),
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
    p.add_argument("--expand-scans", type=int, default=1)

    p.add_argument(
        "--use-apex-as-rt",
        action="store_true",
        help="Update RT to the detected apex RT",
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
    if args.expand_scans < 0:
        raise ValueError("--expand-scans must be >= 0")
    if args.min_rel_height <= 0 or args.min_rel_height >= 1:
        raise ValueError("--min-rel-height must be in (0,1)")
    return args


if __name__ == "__main__":
    recompute(parse_args())
