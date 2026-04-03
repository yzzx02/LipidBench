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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.utils.peak_attributes import (  # noqa: E402
    LITERATURE_TOP_COLUMNS,
    _compute_one_feature_attributes,
    _extract_trace,
    load_ms1_spectra,
)
from PeakTruthLab.scripts.annotation.recompute_rt_bounds_from_labelme import (  # noqa: E402
    _build_axis,
    _expand_left_to_minimum,
    _expand_right_with_rollback,
    _extract_window,
    _find_boundary_from_apex,
    _load_labelme_annotation,
    _nearest_idx,
    _pixel_x_to_rt,
    _robust_baseline_sigma,
    _rt_to_pixel_x,
    _smooth3,
    _smooth5,
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


def _parse_source_filter(v: str) -> set[str]:
    return {x.strip() for x in str(v).split(",") if x.strip()}


def _parse_exact_label_set(v: str) -> set[str]:
    return {x.strip() for x in str(v).split(",") if x.strip()}


def _clean_label(v: str) -> str:
    return str(v).strip()


def _norm_label(v: str) -> str:
    return str(v).strip().lower().replace("-", "_")


def _load_feature_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Feature_ID", "source_file", "source_path", "mz", "RT", "RTmin", "RTmax", "is_true_peak"}
    miss = sorted(required - set(df.columns))
    if miss:
        raise ValueError(f"input csv missing columns: {miss}")
    if df["Feature_ID"].astype(str).duplicated().any():
        dup = df.loc[df["Feature_ID"].astype(str).duplicated(), "Feature_ID"].astype(str).iloc[0]
        raise ValueError(f"duplicate Feature_ID detected in input csv: {dup}")
    return df


def _build_source_file_lookup(search_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if not search_root.exists():
        return out
    for p in search_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".mzml":
            continue
        out.setdefault(p.name.lower(), []).append(p.resolve())
    return out


def _source_candidate_score(path: Path) -> tuple[int, int, str]:
    parts = {str(part).lower() for part in path.parts}
    score = 0
    if "results" in parts:
        score += 100
    if "backups" in parts:
        score += 50
    if "datasets" in parts and ("质谱" in parts or "peaktruthlab" in parts):
        score -= 20
    if "data" in parts:
        score -= 10
    if any(any(ord(ch) > 127 for ch in str(part)) for part in path.parts):
        score += 40
    return (score, len(path.parts), str(path).lower())


def _resolve_source_path(source_path: Path, source_file: str, lookup: dict[str, list[Path]]) -> Path:
    if source_path.exists():
        return source_path.resolve()
    candidates = lookup.get(str(source_file).lower(), [])
    if not candidates:
        return source_path
    return sorted(candidates, key=_source_candidate_score)[0]


def _build_file_index(root: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in suffixes:
            continue
        rel_parts = p.relative_to(root).parts
        if any(str(part).startswith(".") for part in rel_parts):
            continue
        out.setdefault(p.stem, p)
    return out


def _get_annotation_id_sets(annotation_root: Path) -> tuple[set[str], set[str]]:
    true_dir = annotation_root / "true_peak"
    false_dir = annotation_root / "false_peak"
    true_ids = {p.stem for p in true_dir.glob("*.png")} if true_dir.exists() else set()
    false_ids = {p.stem for p in false_dir.glob("*.png")} if false_dir.exists() else set()
    return true_ids, false_ids


def _recompute_attrs_with_bounds(
    rt: np.ndarray,
    eic: np.ndarray,
    mass_track: np.ndarray,
    *,
    mz: float,
    rt_center: float,
    rtmin: float,
    rtmax: float,
    rt_tol_sec: float,
) -> dict[str, float]:
    return _compute_one_feature_attributes(
        rt,
        eic,
        mass_track,
        target_mz=float(mz),
        target_rt_min=float(rt_center),
        target_rtmin=float(rtmin),
        target_rtmax=float(rtmax),
        rt_tol_sec=float(rt_tol_sec),
        include_literature_top=True,
    )


def _archive_file(src: Path, dst_root: Path, rel_parts: list[str]) -> str:
    dst = dst_root.joinpath(*rel_parts)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return str(dst)


def _label_names_from_obj(obj: dict[str, Any]) -> list[str]:
    shapes = obj.get("shapes")
    if not isinstance(shapes, list):
        return []
    out: list[str] = []
    for s in shapes:
        if isinstance(s, dict) and isinstance(s.get("label"), str):
            out.append(str(s["label"]))
    return out


def _label_flags(
    labels: list[str],
    *,
    delete_labels: set[str],
    uncertain_labels: set[str],
) -> tuple[bool, bool]:
    cleaned = {_clean_label(x) for x in labels if _clean_label(x)}
    return bool(cleaned & delete_labels), bool(cleaned & uncertain_labels)


def _json_has_any_exact_label(json_path: Path, labels: set[str]) -> bool:
    if not labels:
        return False
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return any(_clean_label(x) in labels for x in _label_names_from_obj(obj))


def _render_eic_png(
    png_path: Path,
    *,
    rt: np.ndarray,
    eic: np.ndarray,
    center_rt: float,
    window_min: float,
    width_px: int,
    height_px: int,
    dpi: int,
) -> None:
    rt_win, eic_win = _extract_window(rt, eic, center_rt=float(center_rt), window_min=float(window_min))
    fig, _ax = _build_axis(
        rt_win,
        eic_win,
        center_rt=float(center_rt),
        width_px=int(width_px),
        height_px=int(height_px),
        dpi=int(dpi),
    )
    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=int(dpi))
    finally:
        plt.close(fig)


def _is_tiny_box(
    *,
    width_px: float,
    rtmin_from_box: float,
    rtmax_from_box: float,
    li: int,
    ri: int,
    tiny_box_min_width_px: float,
    tiny_box_min_width_sec: float,
    tiny_box_min_scans: int,
) -> bool:
    box_scans = int(max(0, ri - li + 1))
    box_width_sec = float(max(rtmax_from_box - rtmin_from_box, 0.0) * 60.0)
    return bool(
        float(width_px) <= float(tiny_box_min_width_px)
        or box_width_sec <= float(tiny_box_min_width_sec)
        or box_scans < int(tiny_box_min_scans)
    )


def finalize(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    annotation_root = Path(args.annotation_root).resolve()
    active_root = Path(args.images_root).resolve()
    reserved_root = Path(args.reserved_images_root).resolve()
    report_csv = Path(args.report_csv).resolve()
    backup_dir = Path(args.backup_dir).resolve()
    enable_backup = bool(args.enable_backup)

    if not input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {input_csv}")
    if not annotation_root.exists():
        raise FileNotFoundError(f"annotation root not found: {annotation_root}")
    if not active_root.exists():
        raise FileNotFoundError(f"active images root not found: {active_root}")

    df = _load_feature_table(input_csv)
    df["Feature_ID"] = df["Feature_ID"].astype(str)
    df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce")
    df["RTmin"] = pd.to_numeric(df["RTmin"], errors="coerce")
    df["RTmax"] = pd.to_numeric(df["RTmax"], errors="coerce")

    true_ids, false_ids = _get_annotation_id_sets(annotation_root)
    active_png_index = _build_file_index(active_root, (".png",))
    active_json_index = _build_file_index(active_root, (".json",))
    reserved_png_index = _build_file_index(reserved_root, (".png",)) if reserved_root.exists() else {}
    reserved_json_index = _build_file_index(reserved_root, (".json",)) if reserved_root.exists() else {}
    label_source = str(args.label_source)
    source_filter = _parse_source_filter(args.only_source_files)
    delete_labels = _parse_exact_label_set(args.delete_labels)
    uncertain_labels = _parse_exact_label_set(args.uncertain_labels)
    overlap_labels = sorted(delete_labels & uncertain_labels)
    if overlap_labels:
        raise ValueError(f"delete-labels and uncertain-labels overlap: {overlap_labels}")

    csv_ids = set(df["Feature_ID"].astype(str))
    all_annot_ids = sorted(true_ids | false_ids)
    if label_source == "active_json":
        active_ids = sorted(fid for fid in csv_ids if fid in active_png_index and fid in active_json_index)
        reserved_ids = sorted(fid for fid in csv_ids if fid in reserved_png_index and fid in reserved_json_index)
    else:
        active_ids = [fid for fid in all_annot_ids if fid in csv_ids and fid in active_png_index and fid in active_json_index]
        reserved_ids = [fid for fid in all_annot_ids if fid in csv_ids and fid in reserved_png_index and fid in reserved_json_index]
    stale_ids = [fid for fid in all_annot_ids if fid not in csv_ids and fid not in active_png_index and fid not in reserved_png_index]
    missing_asset_ids = [
        fid
        for fid in all_annot_ids
        if fid in csv_ids and fid not in active_png_index and fid not in reserved_png_index
    ]
    if label_source == "active_json":
        extra_active_delete_ids: list[str] = []
    else:
        extra_active_delete_ids = sorted(
            fid
            for fid, json_path in active_json_index.items()
            if fid in csv_ids and fid not in all_annot_ids and _json_has_any_exact_label(json_path, delete_labels)
        )

    row_index = {str(fid): int(i) for i, fid in df["Feature_ID"].items()}
    spectra_cache: dict[str, Any] = {}
    updates: dict[int, dict[str, Any]] = {}
    delete_indices: set[int] = set()
    delete_asset_specs: list[tuple[Path, Path]] = []
    report_rows: list[dict[str, Any]] = []
    source_lookup = _build_source_file_lookup(Path(args.source_search_root).resolve())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_root = backup_dir / f"annotation_finalize_assets_{ts}"
    archive_active_root = archive_root / "eic_images_flat"
    archive_annotation_root = archive_root / "annotation_by_image"

    for fid in reserved_ids:
        report_rows.append(
            {
                "Feature_ID": fid,
                "status": "skip_reserved_root",
                "annotation_folder": "true_peak" if fid in true_ids else "false_peak",
                "json_path": str(reserved_json_index.get(fid, "")),
            }
        )

    for fid in stale_ids:
        report_rows.append(
            {
                "Feature_ID": fid,
                "status": "skip_stale_annotation_only",
                "annotation_folder": "true_peak" if fid in true_ids else "false_peak",
            }
        )

    for fid in missing_asset_ids:
        report_rows.append(
            {
                "Feature_ID": fid,
                "status": "skip_missing_root_asset",
                "annotation_folder": "true_peak" if fid in true_ids else "false_peak",
            }
        )

    for fid in extra_active_delete_ids:
        idx = row_index[fid]
        json_path = active_json_index[fid]
        png_path = active_png_index.get(fid)
        delete_indices.add(idx)
        if png_path is not None:
            delete_asset_specs.append((png_path, archive_active_root / png_path.relative_to(active_root)))
        delete_asset_specs.append((json_path, archive_active_root / json_path.relative_to(active_root)))
        row = df.loc[idx]
        report_rows.append(
            {
                "Feature_ID": fid,
                "status": "deleted_delete_label_active_only",
                "annotation_folder": "",
                "json_path": str(json_path),
                "old_RT": float(row["RT"]) if pd.notna(row["RT"]) else np.nan,
                "old_RTmin": float(row["RTmin"]) if pd.notna(row["RTmin"]) else np.nan,
                "old_RTmax": float(row["RTmax"]) if pd.notna(row["RTmax"]) else np.nan,
                "labels": "|".join(sorted(delete_labels)),
            }
        )

    for fid in active_ids:
        idx = row_index[fid]
        row = df.loc[idx]
        json_path = active_json_index[fid]
        png_path = active_png_index[fid]
        source_file = str(row["source_file"])
        if source_filter and source_file not in source_filter:
            continue
        source_path_raw = Path(str(row["source_path"]))
        source_path = _resolve_source_path(source_path_raw, source_file, source_lookup)
        old_rt = float(row["RT"]) if pd.notna(row["RT"]) else np.nan
        old_rtmin = float(row["RTmin"]) if pd.notna(row["RTmin"]) else np.nan
        old_rtmax = float(row["RTmax"]) if pd.notna(row["RTmax"]) else np.nan
        annotation_folder = "true_peak" if fid in true_ids else "false_peak" if fid in false_ids else ""
        default_label = 1 if fid in true_ids else 0 if fid in false_ids else np.nan

        if not source_path.exists():
            report_rows.append(
                {
                    "Feature_ID": fid,
                    "status": "skip_missing_mzml",
                    "annotation_folder": annotation_folder,
                    "json_path": str(json_path),
                    "source_path": str(source_path_raw),
                    "resolved_source_path": str(source_path),
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
                    "Feature_ID": fid,
                    "status": "skip_no_ms1",
                    "annotation_folder": annotation_folder,
                    "json_path": str(json_path),
                    "source_path": str(source_path),
                }
            )
            continue

        rt, eic, mass_track = _extract_trace(
            spectra,
            target_mz=float(row["mz"]),
            tolerance=float(args.mz_tolerance),
            unit=str(args.tolerance_unit),
            method=str(args.method),
        )
        if rt.size == 0:
            report_rows.append(
                {
                    "Feature_ID": fid,
                    "status": "skip_empty_trace",
                    "annotation_folder": annotation_folder,
                    "json_path": str(json_path),
                }
            )
            continue

        ann = _load_labelme_annotation(json_path)
        labels = [_clean_label(x) for x in _label_names_from_obj(ann["obj"]) if _clean_label(x)] if isinstance(ann.get("obj"), dict) else []
        has_delete_label, has_uncertain_label = _label_flags(
            labels,
            delete_labels=delete_labels,
            uncertain_labels=uncertain_labels,
        )

        if has_delete_label:
            delete_indices.add(idx)
            delete_asset_specs.append((png_path, archive_active_root / png_path.relative_to(active_root)))
            delete_asset_specs.append((json_path, archive_active_root / json_path.relative_to(active_root)))
            ann_png = (annotation_root / "true_peak" / f"{fid}.png") if fid in true_ids else (annotation_root / "false_peak" / f"{fid}.png")
            if ann_png.exists():
                delete_asset_specs.append((ann_png, archive_annotation_root / ann_png.relative_to(annotation_root)))
            report_rows.append(
                {
                    "Feature_ID": fid,
                    "status": "deleted_delete_label",
                    "annotation_folder": annotation_folder,
                    "json_path": str(json_path),
                    "old_RT": old_rt,
                    "old_RTmin": old_rtmin,
                    "old_RTmax": old_rtmax,
                    "labels": "|".join(labels),
                }
            )
            continue

        if label_source == "active_json":
            if has_uncertain_label:
                new_label = np.nan
            else:
                new_label = 0 if (bool(ann.get("is_empty", False)) or ann.get("rect_idx") is None) else 1
        else:
            # Default all non-delete annotated samples to positive unless explicitly put in false_peak folder later.
            if has_uncertain_label:
                new_label = np.nan
            else:
                new_label = int(default_label) if pd.notna(default_label) else 1

        if bool(ann.get("is_empty", False)) or ann.get("rect_idx") is None:
            attrs = _recompute_attrs_with_bounds(
                rt,
                eic,
                mass_track,
                mz=float(row["mz"]),
                rt_center=float(old_rt),
                rtmin=float(old_rtmin),
                rtmax=float(old_rtmax),
                rt_tol_sec=float(args.rt_tol_sec),
            )
            upd = {"is_true_peak": new_label}
            if source_path != source_path_raw:
                upd["source_path"] = str(source_path)
            for c in LITERATURE_TOP_COLUMNS:
                upd[c] = attrs.get(c, np.nan)
            updates[idx] = upd
            report_rows.append(
                {
                    "Feature_ID": fid,
                    "status": "uncertain_no_box_keep_original" if has_uncertain_label else "fallback_original_no_box",
                    "annotation_folder": annotation_folder,
                    "json_path": str(json_path),
                    "old_RT": old_rt,
                    "old_RTmin": old_rtmin,
                    "old_RTmax": old_rtmax,
                    "new_RT": old_rt,
                    "new_RTmin": old_rtmin,
                    "new_RTmax": old_rtmax,
                    "is_true_peak": new_label,
                    "labels": "|".join(labels),
                }
            )
            continue

        iw = int(ann["iw"])
        ih = int(ann["ih"])
        x_left = float(ann["x_left"])
        x_right = float(ann["x_right"])
        width_px = float(abs(x_right - x_left))

        rt_win, eic_win = _extract_window(rt, eic, center_rt=old_rt, window_min=float(args.window_min))
        fig, ax = _build_axis(
            rt_win,
            eic_win,
            center_rt=old_rt,
            width_px=iw,
            height_px=ih,
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

            if _is_tiny_box(
                width_px=width_px,
                rtmin_from_box=rtmin_from_box,
                rtmax_from_box=rtmax_from_box,
                li=li,
                ri=ri,
                tiny_box_min_width_px=float(args.tiny_box_min_width_px),
                tiny_box_min_width_sec=float(args.tiny_box_min_width_sec),
                tiny_box_min_scans=int(args.tiny_box_min_scans),
            ):
                attrs = _recompute_attrs_with_bounds(
                    rt,
                    eic,
                    mass_track,
                    mz=float(row["mz"]),
                    rt_center=float(old_rt),
                    rtmin=float(old_rtmin),
                    rtmax=float(old_rtmax),
                    rt_tol_sec=float(args.rt_tol_sec),
                )
                upd = {"is_true_peak": new_label}
                for c in LITERATURE_TOP_COLUMNS:
                    upd[c] = attrs.get(c, np.nan)
                updates[idx] = upd
                if source_path != source_path_raw:
                    updates[idx]["source_path"] = str(source_path)
                report_rows.append(
                    {
                        "Feature_ID": fid,
                        "status": "uncertain_tiny_box_keep_original" if has_uncertain_label else "skip_tiny_box_keep_original",
                        "annotation_folder": annotation_folder,
                        "json_path": str(json_path),
                        "old_RT": old_rt,
                        "old_RTmin": old_rtmin,
                        "old_RTmax": old_rtmax,
                        "new_RT": old_rt,
                        "new_RTmin": old_rtmin,
                        "new_RTmax": old_rtmax,
                        "box_RTmin": rtmin_from_box,
                        "box_RTmax": rtmax_from_box,
                        "box_width_px": width_px,
                        "box_scan_count": int(ri - li + 1),
                        "box_width_sec": float(max(rtmax_from_box - rtmin_from_box, 0.0) * 60.0),
                        "is_true_peak": new_label,
                        "labels": "|".join(labels),
                    }
                )
                continue

            ys = _smooth5(_smooth3(np.asarray(eic, dtype=np.float64)))
            ys = np.where(np.isfinite(ys), ys, 0.0)
            ys[ys < 0] = 0.0

            box_apex_idx = int(np.argmax(ys[li : ri + 1])) + int(li)
            box_apex_rt = float(rt[box_apex_idx])

            old_rt_in_box = bool(rtmin_from_box <= old_rt <= rtmax_from_box)
            new_rt = float(old_rt) if old_rt_in_box else float(box_apex_rt)
            box_center_px = float((x_left + x_right) * 0.5)
            box_center_offset_px = float(abs(box_center_px - (iw * 0.5)))

            li_ext = _expand_left_to_minimum(
                rt,
                ys,
                li,
                max_expand_scans=int(args.max_expand_scans),
                max_expand_min=float(args.max_expand_min),
                rise_rel_tol=float(args.rise_rel_tol),
                rise_patience=int(args.rise_patience),
            )
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

            rect_idx = int(ann["rect_idx"])
            pts = ann["obj"]["shapes"][rect_idx]["points"]
            rerendered_image = False
            rerender_reason = ""
            rerender_mode = str(args.rerender_box_image)
            if rerender_mode == "always":
                rerendered_image = True
                rerender_reason = "always"
            elif rerender_mode == "recenter_only":
                if bool(args.recenter_changed_peak) and (
                    (not old_rt_in_box) or (box_center_offset_px >= float(args.recenter_box_offset_px))
                ):
                    rerendered_image = True
                    rerender_reason = "rt_outside_box" if not old_rt_in_box else "box_far_from_center"

            ann["obj"]["imagePath"] = png_path.name
            ann["obj"]["imageWidth"] = int(iw)
            ann["obj"]["imageHeight"] = int(ih)
            mid_y_px = ih * 0.5
            if rerendered_image:
                _render_eic_png(
                    png_path,
                    rt=rt,
                    eic=eic,
                    center_rt=float(new_rt),
                    window_min=float(args.window_min),
                    width_px=int(iw),
                    height_px=int(ih),
                    dpi=int(args.image_dpi),
                )
                rt_win_new, eic_win_new = _extract_window(rt, eic, center_rt=float(new_rt), window_min=float(args.window_min))
                fig2, ax2 = _build_axis(
                    rt_win_new,
                    eic_win_new,
                    center_rt=float(new_rt),
                    width_px=iw,
                    height_px=ih,
                    dpi=int(args.image_dpi),
                )
                try:
                    x_new_l = _rt_to_pixel_x(ax2, new_rtmin, mid_y_px, ih)
                    x_new_r = _rt_to_pixel_x(ax2, new_rtmax, mid_y_px, ih)
                finally:
                    plt.close(fig2)
            else:
                x_new_l = _rt_to_pixel_x(ax, new_rtmin, mid_y_px, ih)
                x_new_r = _rt_to_pixel_x(ax, new_rtmax, mid_y_px, ih)
            pts[0][0] = float(min(x_new_l, x_new_r))
            pts[1][0] = float(max(x_new_l, x_new_r))
            if not args.dry_run:
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(ann["obj"], f, ensure_ascii=False, indent=2)

        finally:
            plt.close(fig)

        attrs = _recompute_attrs_with_bounds(
            rt,
            eic,
            mass_track,
            mz=float(row["mz"]),
            rt_center=float(new_rt),
            rtmin=float(new_rtmin),
            rtmax=float(new_rtmax),
            rt_tol_sec=float(args.rt_tol_sec),
        )
        upd = {
            "RTmin": round(float(new_rtmin), 6),
            "RTmax": round(float(new_rtmax), 6),
            "RT": round(float(new_rt), 6),
            "is_true_peak": new_label,
        }
        if source_path != source_path_raw:
            upd["source_path"] = str(source_path)
        for c in LITERATURE_TOP_COLUMNS:
            upd[c] = attrs.get(c, np.nan)
        updates[idx] = upd

        report_rows.append(
            {
                "Feature_ID": fid,
                "status": "uncertain_updated_from_box" if has_uncertain_label else "updated_from_box",
                "annotation_folder": annotation_folder,
                "json_path": str(json_path),
                "old_RT": old_rt,
                "new_RT": float(new_rt),
                "old_rt_in_box": bool(old_rt_in_box),
                "box_apex_RT": float(box_apex_rt),
                "box_center_offset_px": float(box_center_offset_px),
                "old_RTmin": old_rtmin,
                "box_RTmin": float(rtmin_from_box),
                "new_RTmin": float(new_rtmin),
                "old_RTmax": old_rtmax,
                "box_RTmax": float(rtmax_from_box),
                "new_RTmax": float(new_rtmax),
                "box_width_px": width_px,
                "box_scan_count": int(ri - li + 1),
                "box_width_sec": float(max(rtmax_from_box - rtmin_from_box, 0.0) * 60.0),
                "left_expand_scans": int(max(0, li - li_ext)),
                "right_expand_scans": int(max(0, ri_ext - ri)),
                "rollback_to_valley": bool(rolled_back),
                "rollback_valley_RT": float(rt[valley_idx]),
                "bound_mode": str(bound_mode),
                "low_threshold": float(low_thr),
                "oversized_shrink": bool(oversized),
                "rerendered_image": bool(rerendered_image),
                "rerender_reason": str(rerender_reason),
                "recentered_image": bool(rerendered_image),
                "recenter_reason": str(rerender_reason),
                "is_true_peak": new_label,
                "labels": "|".join(labels),
            }
        )

    out_df = df.copy()
    for i, upd in updates.items():
        for k, v in upd.items():
            out_df.at[i, k] = v

    if delete_indices:
        out_df = out_df.drop(index=sorted(delete_indices)).reset_index(drop=True)

    if not args.dry_run:
        backup_path = _backup_final_csv_if_needed(output_csv, backup_dir) if enable_backup else None
        for src, dst in delete_asset_specs:
            if src.exists():
                if enable_backup:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                else:
                    src.unlink()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)
    else:
        backup_path = None
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)

    report_csv.parent.mkdir(parents=True, exist_ok=True)
    rep = pd.DataFrame(report_rows)
    rep.to_csv(report_csv, index=False)

    status_counts = rep["status"].value_counts().to_dict() if not rep.empty else {}
    print("done")
    print(f"input_csv:              {input_csv}")
    print(f"output_csv:             {output_csv}")
    print(f"annotation_root:        {annotation_root}")
    print(f"active_images_root:     {active_root}")
    print(f"reserved_images_root:   {reserved_root}")
    print(f"label_source:           {label_source}")
    print(f"delete_labels:          {sorted(delete_labels)}")
    print(f"uncertain_labels:       {sorted(uncertain_labels)}")
    print(f"enable_backup:          {enable_backup}")
    print(f"rerender_box_image:     {args.rerender_box_image}")
    print(f"annot_true_pngs:        {len(true_ids)}")
    print(f"annot_false_pngs:       {len(false_ids)}")
    print(f"processable_active_ids: {len(active_ids)}")
    print(f"reserved_skipped_ids:   {len(reserved_ids)}")
    print(f"stale_annotation_ids:   {len(stale_ids)}")
    print(f"missing_asset_ids:      {len(missing_asset_ids)}")
    print(f"extra_active_delete_ids:{len(extra_active_delete_ids)}")
    print(f"rows_updated:           {len(updates)}")
    print(f"rows_deleted_label:     {len(delete_indices)}")
    print(f"delete_asset_moves:     {len(delete_asset_specs)}")
    print(f"dry_run:                {bool(args.dry_run)}")
    print(f"report_csv:             {report_csv}")
    print(f"status_counts:          {status_counts}")
    if backup_path is not None:
        print(f"backup_csv:             {backup_path}")
    if not args.dry_run and enable_backup:
        print(f"archive_root:           {archive_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Finalize annotation_by_image labels into feature_table_final_10000.csv"
    )
    p.add_argument("--input-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--output-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--annotation-root", type=str, default="PeakTruthLab/datasets/annotation_by_image")
    p.add_argument("--images-root", type=str, default="PeakTruthLab/datasets/eic_images_flat")
    p.add_argument("--reserved-images-root", type=str, default="PeakTruthLab/datasets/eic_images_flat_2")
    p.add_argument("--report-csv", type=str, default="PeakTruthLab/results/finalize_annotation_by_image_report.csv")
    p.add_argument("--backup-dir", type=str, default="PeakTruthLab/datasets/backups")
    p.add_argument("--enable-backup", action="store_true", help="Save CSV/assets backup before overwriting or deleting files")
    p.add_argument("--source-search-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--only-source-files", type=str, default="", help="Optional comma-separated source_file names to process")
    p.add_argument(
        "--label-source",
        type=str,
        default="annotation_folder",
        choices=["annotation_folder", "active_json"],
        help="Use annotation_by_image folder membership or active JSON box/no-box state as labels",
    )
    p.add_argument("--delete-labels", type=str, default="D", help="Exact label names that remove a sample from the dataset")
    p.add_argument("--uncertain-labels", type=str, default="d", help="Exact label names that keep a sample but set is_true_peak empty")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--mz-tolerance", type=float, default=15.0)
    p.add_argument("--tolerance-unit", type=str, default="ppm", choices=["ppm", "Da"])
    p.add_argument("--method", type=str, default="nearest", choices=["nearest", "window_sum"])
    p.add_argument("--window-min", type=float, default=2.0)
    p.add_argument("--image-dpi", type=int, default=150)
    p.add_argument("--max-expand-scans", type=int, default=18)
    p.add_argument("--max-expand-min", type=float, default=0.35)
    p.add_argument("--rise-patience", type=int, default=2)
    p.add_argument("--rise-rel-tol", type=float, default=0.02)
    p.add_argument("--rebound-rel", type=float, default=0.12)
    p.add_argument("--noise-sigma-mult", type=float, default=3.0)
    p.add_argument("--min-rel-height", type=float, default=0.01)
    p.add_argument("--oversize-factor", type=float, default=1.8)
    p.add_argument("--boundary-confirm", type=int, default=2)
    p.add_argument("--rt-tol-sec", type=float, default=30.0)
    p.add_argument("--tiny-box-min-width-px", type=float, default=4.5)
    p.add_argument("--tiny-box-min-width-sec", type=float, default=1.0)
    p.add_argument("--tiny-box-min-scans", type=int, default=3)
    p.add_argument(
        "--rerender-box-image",
        type=str,
        default="recenter_only",
        choices=["never", "recenter_only", "always"],
        help="When to redraw PNGs for box-labeled samples after recomputing bounds",
    )
    p.add_argument("--recenter-changed-peak", action="store_true")
    p.add_argument("--recenter-box-offset-px", type=float, default=60.0)

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
    if args.tiny_box_min_width_px <= 0:
        raise ValueError("--tiny-box-min-width-px must be > 0")
    if args.tiny_box_min_width_sec <= 0:
        raise ValueError("--tiny-box-min-width-sec must be > 0")
    if args.tiny_box_min_scans <= 0:
        raise ValueError("--tiny-box-min-scans must be > 0")
    if args.recenter_box_offset_px < 0:
        raise ValueError("--recenter-box-offset-px must be >= 0")
    return args


if __name__ == "__main__":
    finalize(parse_args())
