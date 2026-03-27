from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, ScalarFormatter

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


def _tick_formatter(v: float, _pos: Any) -> str:
    if abs(v) < 1e-12:
        v = 0.0
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.1f}"


def _nice_step(target_step: float) -> float:
    if target_step <= 0:
        return 0.5
    exp = np.floor(np.log10(target_step))
    base = target_step / (10 ** exp)
    if base <= 1:
        nice = 1
    elif base <= 2:
        nice = 2
    elif base <= 2.5:
        nice = 2.5
    elif base <= 5:
        nice = 5
    else:
        nice = 10
    return float(nice * (10 ** exp))


def _extract_window(rt: np.ndarray, eic: np.ndarray, center_rt: float, window_min: float) -> tuple[np.ndarray, np.ndarray]:
    half = float(window_min) / 2.0
    lo = center_rt - half
    hi = center_rt + half
    m = (rt >= lo) & (rt <= hi)
    if np.any(m):
        return rt[m], eic[m]
    return rt, eic


def _build_axis(rt_win: np.ndarray, eic_win: np.ndarray, center_rt: float, width_px: int, height_px: int, dpi: int):
    fig = plt.figure(figsize=(float(width_px) / float(dpi), float(height_px) / float(dpi)), dpi=int(dpi))
    ax = fig.add_subplot(111)

    y = np.asarray(eic_win, dtype=np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    y[y < 0] = 0.0
    ax.plot(rt_win, y, color="royalblue", linewidth=0.8)

    half = 1.0
    ax.set_xlim(center_rt - half, center_rt + half)

    y_max = float(np.max(y)) if y.size else 0.0
    y_top = max(y_max * 1.15, 0.5)
    y_bottom = -0.08 * y_top
    ax.set_ylim(y_bottom, y_top)

    y_range = max(y_top - y_bottom, 0.5)
    y_step = max(0.5, _nice_step(y_range / 5.0))
    _ = y_step

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(_tick_formatter))

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.tick_params(axis="both", labelsize=6)
    ax.yaxis.get_offset_text().set_size(6)

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.canvas.draw()
    return fig, ax


def _load_labelme_annotation(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    shapes = obj.get("shapes")
    if not isinstance(shapes, list):
        return {
            "obj": obj,
            "iw": int(obj.get("imageWidth", 480) or 480),
            "ih": int(obj.get("imageHeight", 480) or 480),
            "is_empty": True,
            "has_d": False,
            "rect_idx": None,
            "x_left": None,
            "x_right": None,
        }
    if len(shapes) == 0:
        return {
            "obj": obj,
            "iw": int(obj.get("imageWidth", 480) or 480),
            "ih": int(obj.get("imageHeight", 480) or 480),
            "is_empty": True,
            "has_d": False,
            "rect_idx": None,
            "x_left": None,
            "x_right": None,
        }

    has_d = False
    rect_idx = None
    rect = None
    for i, s in enumerate(shapes):
        if not isinstance(s, dict):
            continue
        label = str(s.get("label", "")).strip()
        if label == "D":
            has_d = True
            continue
        pts = s.get("points")
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        p1 = pts[0]
        p2 = pts[1]
        if not (isinstance(p1, list) and isinstance(p2, list) and len(p1) >= 2 and len(p2) >= 2):
            continue
        x1 = float(p1[0])
        x2 = float(p2[0])
        if abs(x2 - x1) <= 1e-12:
            continue
        rect = pts
        rect_idx = i
        break

    iw = int(obj.get("imageWidth", 480) or 480)
    ih = int(obj.get("imageHeight", 480) or 480)
    if rect is None:
        return {
            "obj": obj,
            "iw": iw,
            "ih": ih,
            "is_empty": False,
            "has_d": bool(has_d),
            "rect_idx": None,
            "x_left": None,
            "x_right": None,
        }

    p1 = rect[0]
    p2 = rect[1]
    x1 = float(p1[0])
    x2 = float(p2[0])
    return {
        "obj": obj,
        "iw": iw,
        "ih": ih,
        "is_empty": False,
        "has_d": bool(has_d),
        "rect_idx": int(rect_idx),
        "x_left": min(x1, x2),
        "x_right": max(x1, x2),
    }


def _pixel_x_to_rt(ax, x_px: float, y_px: float, img_h: float) -> float:
    y_disp = float(img_h - y_px)
    x_data, _ = ax.transData.inverted().transform((float(x_px), y_disp))
    return float(x_data)


def _rt_to_pixel_x(ax, rt_val: float, y_px: float, img_h: float) -> float:
    y_disp = float(img_h - y_px)
    x_disp, _ = ax.transData.transform((float(rt_val), y_disp))
    return float(x_disp)


def _nearest_idx(rt: np.ndarray, x: float) -> int:
    if rt.size == 0:
        return -1
    return int(np.argmin(np.abs(rt - float(x))))


def _smooth3(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y.copy()
    k = np.asarray([1.0, 1.0, 1.0], dtype=np.float64) / 3.0
    return np.convolve(y, k, mode="same")


def _smooth5(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y.copy()
    k = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64) / 5.0
    return np.convolve(y, k, mode="same")


def _robust_baseline_sigma(y: np.ndarray) -> tuple[float, float]:
    if y.size == 0:
        return 0.0, 0.0
    yy = np.asarray(y, dtype=np.float64)
    yy = yy[np.isfinite(yy)]
    if yy.size == 0:
        return 0.0, 0.0
    baseline = float(np.median(yy))
    mad = float(np.median(np.abs(yy - baseline)))
    sigma = 1.4826 * mad
    return baseline, sigma


def _expand_left_to_minimum(
    rt: np.ndarray,
    ys: np.ndarray,
    left_idx: int,
    *,
    max_expand_scans: int,
    max_expand_min: float,
    rise_rel_tol: float,
    rise_patience: int,
) -> int:
    if left_idx <= 0:
        return left_idx
    start_rt = float(rt[left_idx])
    idx = int(left_idx)
    best_idx = idx
    best_val = float(ys[idx])
    steps = 0
    rise_streak = 0
    while idx - 1 >= 0:
        if steps >= int(max_expand_scans):
            break
        if start_rt - float(rt[idx - 1]) > float(max_expand_min):
            break
        nxt = float(ys[idx - 1])
        if nxt <= best_val * (1.0 + float(rise_rel_tol)):
            idx -= 1
            steps += 1
            rise_streak = 0
            if nxt < best_val:
                best_val = nxt
                best_idx = idx
            continue
        rise_streak += 1
        if rise_streak <= int(max(0, rise_patience)):
            idx -= 1
            steps += 1
            continue
        break
    return int(best_idx)


def _expand_right_with_rollback(
    rt: np.ndarray,
    ys: np.ndarray,
    right_idx: int,
    *,
    max_expand_scans: int,
    max_expand_min: float,
    rise_rel_tol: float,
    rebound_rel: float,
    rise_patience: int,
) -> tuple[int, int, bool]:
    if right_idx < 0 or right_idx >= len(rt):
        return right_idx, right_idx, False

    start_rt = float(rt[right_idx])
    idx = int(right_idx)
    valley_idx = int(right_idx)
    valley_val = float(ys[right_idx])
    steps = 0
    saw_descent = False
    rise_streak = 0

    while idx + 1 < len(rt):
        if steps >= int(max_expand_scans):
            break
        if float(rt[idx + 1]) - start_rt > float(max_expand_min):
            break

        cur = float(ys[idx])
        nxt = float(ys[idx + 1])

        if nxt <= cur * (1.0 + float(rise_rel_tol)):
            saw_descent = True
            idx += 1
            steps += 1
            rise_streak = 0
            if nxt < valley_val:
                valley_val = nxt
                valley_idx = idx
            continue

        if saw_descent and nxt >= valley_val * (1.0 + float(rebound_rel)):
            return int(valley_idx), int(valley_idx), True

        rise_streak += 1
        if rise_streak > int(max(0, rise_patience)):
            break

        idx += 1
        steps += 1

    return int(idx), int(valley_idx), False


def _find_boundary_from_apex(
    rt: np.ndarray,
    ys: np.ndarray,
    apex_idx: int,
    *,
    direction: str,
    threshold: float,
    max_walk_scans: int,
    confirm_scans: int,
) -> int:
    n = len(ys)
    idx = int(apex_idx)
    steps = 0
    below_streak = 0
    need = int(max(1, confirm_scans))
    if direction == "left":
        while idx - 1 >= 0 and steps < int(max_walk_scans):
            idx -= 1
            steps += 1
            if float(ys[idx]) <= float(threshold):
                below_streak += 1
                if below_streak >= need:
                    break
            else:
                below_streak = 0
        return int(idx)
    while idx + 1 < n and steps < int(max_walk_scans):
        idx += 1
        steps += 1
        if float(ys[idx]) <= float(threshold):
            below_streak += 1
            if below_streak >= need:
                break
        else:
            below_streak = 0
    return int(idx)


def _expand_right_if_descending(
    rt: np.ndarray,
    y: np.ndarray,
    right_idx: int,
    *,
    max_expand_scans: int,
    max_expand_min: float,
    rise_rel_tol: float,
) -> int:
    if right_idx < 0 or right_idx >= len(rt):
        return right_idx
    if max_expand_scans <= 0 or max_expand_min <= 0:
        return right_idx

    ys = _smooth3(np.asarray(y, dtype=np.float64))
    ys = np.where(np.isfinite(ys), ys, 0.0)
    ys[ys < 0] = 0.0

    start_rt = float(rt[right_idx])
    idx = int(right_idx)
    steps = 0
    while idx + 1 < len(rt):
        if steps >= int(max_expand_scans):
            break
        if float(rt[idx + 1]) - start_rt > float(max_expand_min):
            break

        cur = float(ys[idx])
        nxt = float(ys[idx + 1])
        if nxt <= cur * (1.0 + float(rise_rel_tol)):
            idx += 1
            steps += 1
            continue
        break
    return idx


def _parse_source_filter(v: str) -> set[str]:
    if not v.strip():
        return set()
    return {x.strip() for x in v.split(",") if x.strip()}


def recompute_from_labelme(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    report_csv = Path(args.report_csv).resolve() if args.report_csv else None
    images_root = Path(args.images_root).resolve()
    backup_dir = Path(args.backup_dir).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")
    if not images_root.exists():
        raise FileNotFoundError(f"images root not found: {images_root}")

    df = pd.read_csv(input_csv)
    req = {"Feature_ID", "source_file", "source_path", "mz", "RT", "RTmin", "RTmax"}
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
    delete_indices: list[int] = []
    report_rows: list[dict[str, Any]] = []
    spectra_cache: dict[str, Any] = {}

    for idx, row in work.iterrows():
        feature_id = str(row["Feature_ID"])
        source_file = str(row["source_file"])
        source_path = Path(str(row["source_path"]))
        stem = source_file.replace(".mzML", "")
        json_path = images_root / stem / f"{feature_id}.json"

        old_rt = float(row["RT"])
        old_rtmin = float(row["RTmin"]) if pd.notna(row["RTmin"]) else np.nan
        old_rtmax = float(row["RTmax"]) if pd.notna(row["RTmax"]) else np.nan

        if not source_path.exists():
            report_rows.append(
                {
                    "Feature_ID": feature_id,
                    "source_file": source_file,
                    "status": "skip_missing_mzml",
                    "json_path": str(json_path),
                    "old_RT": old_rt,
                    "old_RTmin": old_rtmin,
                    "old_RTmax": old_rtmax,
                }
            )
            continue

        if not json_path.exists():
            report_rows.append(
                {
                    "Feature_ID": feature_id,
                    "source_file": source_file,
                    "status": "skip_missing_json",
                    "json_path": str(json_path),
                    "old_RT": old_rt,
                    "old_RTmin": old_rtmin,
                    "old_RTmax": old_rtmax,
                }
            )
            continue

        sp_key = str(source_path)
        if sp_key not in spectra_cache:
            spectra_cache[sp_key] = load_ms1_spectra(source_path)
        spectra = spectra_cache[sp_key]
        if not spectra:
            report_rows.append(
                {
                    "Feature_ID": feature_id,
                    "source_file": source_file,
                    "status": "skip_no_ms1",
                    "json_path": str(json_path),
                    "old_RT": old_rt,
                    "old_RTmin": old_rtmin,
                    "old_RTmax": old_rtmax,
                }
            )
            continue

        try:
            ann = _load_labelme_annotation(json_path)
            iw = int(ann["iw"])
            ih = int(ann["ih"])

            if bool(ann.get("has_d", False)):
                delete_indices.append(int(idx))
                report_rows.append(
                    {
                        "Feature_ID": feature_id,
                        "source_file": source_file,
                        "status": "deleted_D",
                        "json_path": str(json_path),
                        "old_RT": old_rt,
                        "old_RTmin": old_rtmin,
                        "old_RTmax": old_rtmax,
                    }
                )
                continue

            if bool(ann.get("is_empty", False)):
                report_rows.append(
                    {
                        "Feature_ID": feature_id,
                        "source_file": source_file,
                        "status": "skip_empty_negative",
                        "json_path": str(json_path),
                        "old_RT": old_rt,
                        "old_RTmin": old_rtmin,
                        "old_RTmax": old_rtmax,
                    }
                )
                continue

            if ann.get("rect_idx") is None:
                report_rows.append(
                    {
                        "Feature_ID": feature_id,
                        "source_file": source_file,
                        "status": "skip_invalid_shape",
                        "json_path": str(json_path),
                        "old_RT": old_rt,
                        "old_RTmin": old_rtmin,
                        "old_RTmax": old_rtmax,
                    }
                )
                continue

            x_left = float(ann["x_left"])
            x_right = float(ann["x_right"])

            rt, eic, mass_track = _extract_trace(
                spectra,
                target_mz=float(row["mz"]),
                tolerance=float(args.mz_tolerance),
                unit=str(args.tolerance_unit),
                method=str(args.method),
            )
            if rt.size == 0:
                raise RuntimeError("empty trace")

            rt_win, eic_win = _extract_window(rt, eic, center_rt=old_rt, window_min=float(args.window_min))
            fig, ax = _build_axis(
                rt_win,
                eic_win,
                center_rt=old_rt,
                width_px=int(iw),
                height_px=int(ih),
                dpi=int(args.image_dpi),
            )
            try:
                rt_l = _pixel_x_to_rt(ax, x_left, ih * 0.5, ih)
                rt_r = _pixel_x_to_rt(ax, x_right, ih * 0.5, ih)

                lo = min(rt_l, rt_r)
                hi = max(rt_l, rt_r)
                win_lo = old_rt - float(args.window_min) / 2.0
                win_hi = old_rt + float(args.window_min) / 2.0
                lo = max(lo, win_lo)
                hi = min(hi, win_hi)

                li = _nearest_idx(rt, lo)
                ri = _nearest_idx(rt, hi)
                if li < 0 or ri < 0:
                    raise RuntimeError("invalid nearest index")
                if li > ri:
                    li, ri = ri, li

                rtmin_from_box = float(rt[li])
                rtmax_from_box = float(rt[ri])

                ys = _smooth5(_smooth3(np.asarray(eic, dtype=np.float64)))
                ys = np.where(np.isfinite(ys), ys, 0.0)
                ys[ys < 0] = 0.0

                box_apex_idx = int(np.argmax(ys[li : ri + 1])) + int(li)
                box_apex_rt = float(rt[box_apex_idx])

                old_rt_in_box = bool(rtmin_from_box <= old_rt <= rtmax_from_box)
                if old_rt_in_box:
                    new_rt = old_rt
                else:
                    new_rt = box_apex_rt

                # Step A: from your current box, extend outward to local minima.
                li_ext = _expand_left_to_minimum(
                    rt,
                    ys,
                    li,
                    max_expand_scans=int(args.max_expand_scans),
                    max_expand_min=float(args.max_expand_min),
                    rise_rel_tol=float(args.rise_rel_tol),
                    rise_patience=int(args.rise_patience),
                )

                if bool(args.expand_right_descending):
                    ri_ext, valley_idx, rolled_back = _expand_right_with_rollback(
                        rt,
                        ys,
                        ri,
                        max_expand_scans=int(args.max_expand_scans),
                        max_expand_min=float(args.max_expand_min),
                        rise_rel_tol=float(args.rise_rel_tol),
                        rebound_rel=float(args.rebound_rel),
                        rise_patience=int(args.rise_patience),
                    )
                else:
                    ri_ext = int(ri)
                    valley_idx = int(ri)
                    rolled_back = False

                # Step B: if boundary is severely oversized (near-zero/noise at edges),
                # shrink by descending from apex to threshold on both sides.
                baseline, sigma = _robust_baseline_sigma(ys)
                apex_val = float(ys[box_apex_idx])
                low_thr = max(
                    float(baseline + float(args.noise_sigma_mult) * sigma),
                    float(apex_val * float(args.min_rel_height)),
                )

                max_walk_scans = max(20, int(args.max_expand_scans) * 4)
                li_core = _find_boundary_from_apex(
                    rt,
                    ys,
                    box_apex_idx,
                    direction="left",
                    threshold=low_thr,
                    max_walk_scans=max_walk_scans,
                    confirm_scans=int(args.boundary_confirm),
                )
                ri_core = _find_boundary_from_apex(
                    rt,
                    ys,
                    box_apex_idx,
                    direction="right",
                    threshold=low_thr,
                    max_walk_scans=max_walk_scans,
                    confirm_scans=int(args.boundary_confirm),
                )

                width_ext = float(max(rt[ri_ext] - rt[li_ext], 0.0))
                width_core = float(max(rt[ri_core] - rt[li_core], 1e-12))
                edge_low = bool((ys[li_ext] <= low_thr) and (ys[ri_ext] <= low_thr))
                oversized = bool(edge_low and (width_ext > width_core * float(args.oversize_factor)))

                if oversized:
                    li_final = int(li_core)
                    ri_final = int(ri_core)
                    bound_mode = "core_shrink"
                else:
                    li_final = int(li_ext)
                    ri_final = int(ri_ext)
                    bound_mode = "box_extend"

                if li_final > ri_final:
                    li_final, ri_final = ri_final, li_final

                new_rtmin = float(rt[li_final])
                new_rtmax = float(rt[ri_final])

                if new_rtmax < new_rtmin:
                    new_rtmin, new_rtmax = new_rtmax, new_rtmin

                if bool(args.update_rt_from_box_mid) and old_rt_in_box:
                    new_rt = float((new_rtmin + new_rtmax) / 2.0)

                # Sync the rectangle X bounds back to LabelMe JSON for manual review.
                mid_y_px = ih * 0.5
                x_new_l = _rt_to_pixel_x(ax, new_rtmin, mid_y_px, ih)
                x_new_r = _rt_to_pixel_x(ax, new_rtmax, mid_y_px, ih)
                rect_idx = int(ann["rect_idx"])
                pts = ann["obj"]["shapes"][rect_idx]["points"]
                p1 = pts[0]
                p2 = pts[1]
                p1[0] = float(min(x_new_l, x_new_r))
                p2[0] = float(max(x_new_l, x_new_r))
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(ann["obj"], f, ensure_ascii=False, indent=2)
            finally:
                plt.close(fig)

            attrs = _compute_one_feature_attributes(
                rt,
                eic,
                mass_track,
                target_mz=float(row["mz"]),
                target_rt_min=float(new_rt),
                target_rtmin=float(new_rtmin),
                target_rtmax=float(new_rtmax),
                rt_tol_sec=float(args.rt_tol_sec),
                include_literature_top=True,
            )

            upd = {
                "RTmin": round(float(new_rtmin), 6),
                "RTmax": round(float(new_rtmax), 6),
            }
            if bool(args.update_rt_from_box_mid):
                upd["RT"] = round(float(new_rt), 6)
            for c in LITERATURE_TOP_COLUMNS:
                upd[c] = attrs.get(c, np.nan)
            updates[int(idx)] = upd

            report_rows.append(
                {
                    "Feature_ID": feature_id,
                    "source_file": source_file,
                    "status": "updated",
                    "json_path": str(json_path),
                    "old_RT": old_rt,
                    "new_RT": float(new_rt),
                    "old_rt_in_box": bool(old_rt_in_box),
                    "box_apex_RT": float(box_apex_rt),
                    "old_RTmin": old_rtmin,
                    "box_RTmin": float(rtmin_from_box),
                    "new_RTmin": float(new_rtmin),
                    "old_RTmax": old_rtmax,
                    "box_RTmax": float(rtmax_from_box),
                    "new_RTmax": float(new_rtmax),
                    "left_expand_scans": int(max(0, li - li_ext)),
                    "right_expand_scans": int(max(0, ri_ext - ri)),
                    "rollback_to_valley": bool(rolled_back),
                    "rollback_valley_RT": float(rt[valley_idx]),
                    "bound_mode": str(bound_mode),
                    "low_threshold": float(low_thr),
                    "oversized_shrink": bool(oversized),
                    "json_box_synced": True,
                }
            )
        except Exception as e:
            report_rows.append(
                {
                    "Feature_ID": feature_id,
                    "source_file": source_file,
                    "status": f"error:{e}",
                    "json_path": str(json_path),
                    "old_RT": old_rt,
                    "old_RTmin": old_rtmin,
                    "old_RTmax": old_rtmax,
                }
            )

    for i, upd in updates.items():
        for k, v in upd.items():
            df.at[i, k] = v

    if delete_indices:
        df = df.drop(index=sorted(set(delete_indices))).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    backup_path = _backup_final_csv_if_needed(output_csv, backup_dir)
    df.to_csv(output_csv, index=False)

    rep = pd.DataFrame(report_rows)
    if report_csv:
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        rep.to_csv(report_csv, index=False)

    n_err = int((rep["status"].astype(str).str.startswith("error:")).sum()) if not rep.empty else 0
    n_skip = int((rep["status"].astype(str).str.startswith("skip_")).sum()) if not rep.empty else 0
    n_del = int((rep["status"].astype(str) == "deleted_D").sum()) if not rep.empty else 0

    print("done")
    print(f"input_csv:       {input_csv}")
    print(f"output_csv:      {output_csv}")
    print(f"images_root:     {images_root}")
    print(f"selected_rows:   {len(work)}")
    print(f"updated_rows:    {len(updates)}")
    print(f"deleted_rows:    {n_del}")
    print(f"skipped_rows:    {n_skip}")
    print(f"error_rows:      {n_err}")
    if report_csv:
        print(f"report_csv:      {report_csv}")
    if backup_path is not None:
        print(f"backup_csv:      {backup_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Recompute RTmin/RTmax from LabelMe rectangle boxes and recompute 13 attributes")
    p.add_argument("--input-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--output-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--report-csv", type=str, default="PeakTruthLab/results/rt_bounds_from_labelme_report.csv")
    p.add_argument("--images-root", type=str, default="PeakTruthLab/datasets/eic_images_flat")
    p.add_argument("--backup-dir", type=str, default="PeakTruthLab/datasets/backups")

    p.add_argument(
        "--only-source-files",
        type=str,
        default="",
        help="Optional comma list, e.g. '060-0145-005_017.mzML,060-0145-006_018.mzML'",
    )
    p.add_argument("--feature-id", type=str, default="", help="Optional single Feature_ID")

    p.add_argument("--mz-tolerance", type=float, default=10.0)
    p.add_argument("--tolerance-unit", type=str, default="ppm", choices=["ppm", "Da"])
    p.add_argument("--method", type=str, default="nearest", choices=["nearest", "window_sum"])

    p.add_argument("--window-min", type=float, default=2.0, help="EIC image X-window in minutes")
    p.add_argument("--image-dpi", type=int, default=150)

    p.add_argument("--expand-right-descending", action="store_true")
    p.add_argument("--max-expand-scans", type=int, default=18)
    p.add_argument("--max-expand-min", type=float, default=0.35)
    p.add_argument("--rise-patience", type=int, default=2, help="Allow short spike-like rises before stopping boundary expansion")
    p.add_argument(
        "--rise-rel-tol",
        type=float,
        default=0.02,
        help="Allow tiny rise (2%% default) while still treating right tail as descending",
    )
    p.add_argument(
        "--rebound-rel",
        type=float,
        default=0.12,
        help="If right-side rebound exceeds this ratio after a local minimum, roll back to that minimum",
    )
    p.add_argument("--noise-sigma-mult", type=float, default=3.0)
    p.add_argument("--min-rel-height", type=float, default=0.01)
    p.add_argument("--oversize-factor", type=float, default=1.8)
    p.add_argument("--boundary-confirm", type=int, default=2, help="Need this many consecutive points below threshold to confirm boundary")

    p.add_argument("--update-rt-from-box-mid", action="store_true")
    p.add_argument("--rt-tol-sec", type=float, default=30.0)

    args = p.parse_args()
    if args.mz_tolerance <= 0:
        raise ValueError("--mz-tolerance must be > 0")
    if args.window_min <= 0:
        raise ValueError("--window-min must be > 0")
    if args.image_dpi <= 0:
        raise ValueError("--image-dpi must be > 0")
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
    if args.noise_sigma_mult < 0:
        raise ValueError("--noise-sigma-mult must be >= 0")
    if args.min_rel_height <= 0 or args.min_rel_height >= 1:
        raise ValueError("--min-rel-height must be in (0,1)")
    if args.oversize_factor <= 1.0:
        raise ValueError("--oversize-factor must be > 1")
    if args.boundary_confirm <= 0:
        raise ValueError("--boundary-confirm must be > 0")
    return args


if __name__ == "__main__":
    recompute_from_labelme(parse_args())
