from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import shutil
import sys
import json
import os
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.eic.extract_eic_pyopenms import build
from lipidbench.utils.peak_attributes import (
    compute_peak_attributes,
)

SOURCE_CSV = PROJECT_ROOT / "results/xcms_three_targets/features_merged_filtered.csv"
FINAL_CSV = PROJECT_ROOT / "PeakTruthLab/datasets/feature_table_final_10000.csv"
IMAGES_ROOT = PROJECT_ROOT / "PeakTruthLab/datasets/eic_images_flat"
TOP_AREA_CSV = PROJECT_ROOT / "results/xcms_three_targets/features_top550_per_file_area.csv"
ALIGN_CSV = PROJECT_ROOT / "results/xcms_three_targets/alignment_report_top550_area.csv"
SUMMARY_JSON = PROJECT_ROOT / "results/xcms_three_targets/summary_top550_area_finalize.json"
LOCK_FILE = PROJECT_ROOT / "results/xcms_three_targets/.refresh_top550_area_and_finalize.lock"
BACKUP_DIR = PROJECT_ROOT / "PeakTruthLab/datasets/backups"

TARGETS = [
    ("Blood-30V", "Blood-30V.mzML"),
    ("Urine-30V", "Urine-30V.mzML"),
    ("HepG2-30V", "HepG2-30V.mzML"),
]
TOP_N = 550


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


def _resolve_mzml_path(row: pd.Series) -> Path:
    source_path = str(row.get("source_path", "")).strip()
    source_file = str(row.get("source_file", "")).strip()
    stem = str(row.get("stem", "")).strip()

    candidates: list[Path] = []
    if source_path:
        candidates.append(Path(source_path))
    if source_file:
        candidates.append(PROJECT_ROOT / "data/ceshiji" / source_file)
    if source_file and stem:
        candidates.append(PROJECT_ROOT / "data/xcms_target" / stem / source_file)

    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        f"Cannot resolve mzML path for source_file={source_file}, source_path={source_path}, stem={stem}"
    )


def _select_top_by_area(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = ["source_file", "source_path", "stem", "Feature_ID", "mz", "RTmin", "RT", "RTmax", "maxo", "area"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in source csv: {missing}")

    out_parts: list[pd.DataFrame] = []
    for stem, source_file in TARGETS:
        sub = df[df["source_file"] == source_file].copy()
        if sub.empty:
            raise ValueError(f"No rows in source csv for {source_file}")

        sub["area"] = pd.to_numeric(sub["area"], errors="coerce")
        sub["maxo"] = pd.to_numeric(sub["maxo"], errors="coerce")
        for c in ["mz", "RTmin", "RT", "RTmax"]:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        sub = sub.dropna(subset=["Feature_ID", "mz", "RTmin", "RT", "RTmax", "area"]).copy()
        sub = sub.sort_values(["area", "maxo", "Feature_ID"], ascending=[False, False, True]).head(TOP_N).reset_index(drop=True)

        if len(sub) != TOP_N:
            raise ValueError(f"{source_file} selected {len(sub)} rows, expected {TOP_N}")

        # Fix stem/source_file to expected values for safety.
        sub["stem"] = stem
        sub["source_file"] = source_file

        out_parts.append(sub[need_cols].copy())

    out = pd.concat(out_parts, ignore_index=True)
    out = out.drop_duplicates(subset=["source_file", "Feature_ID"], keep="first").reset_index(drop=True)
    return out


def _rebuild_images(top_df: pd.DataFrame) -> None:
    IMAGES_ROOT.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        processes_number=1,
        method="nearest",
        unit="ppm",
        tolerance=15.0,
        images_path=str(IMAGES_ROOT),
        smooth_sigma=0.0,
    )

    for stem, source_file in TARGETS:
        target_dir = IMAGES_ROOT / stem
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        sub = top_df[top_df["source_file"] == source_file].copy().reset_index(drop=True)
        mzml_path = _resolve_mzml_path(sub.iloc[0])
        info = sub[["Feature_ID", "mz", "RT", "RTmin", "RTmax"]].copy().reset_index(drop=True)

        print(f"[EIC] {source_file}: exporting {len(info)}")
        ok = 0
        failed = 0
        chunk_size = 50

        for start in range(0, len(info), chunk_size):
            chunk = info.iloc[start : start + chunk_size].reset_index(drop=True)
            try:
                build(paths=[mzml_path], info=chunk, plot=True, args=args)
                ok += len(chunk)
                continue
            except Exception:
                pass

            for i in range(len(chunk)):
                one = chunk.iloc[[i]].reset_index(drop=True)
                try:
                    build(paths=[mzml_path], info=one, plot=True, args=args)
                    ok += 1
                except Exception as e:
                    failed += 1
                    if failed <= 10:
                        print(f"[EIC] {source_file}: skip {one.iloc[0]['Feature_ID']} -> {e}")

        print(f"[EIC] {source_file}: ok={ok}, failed={failed}")


def _compute_all_features(top_df: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []

    for stem, source_file in TARGETS:
        sub = top_df[top_df["source_file"] == source_file].copy().reset_index(drop=True)
        mzml_path = _resolve_mzml_path(sub.iloc[0])
        info = sub[["Feature_ID", "mz", "RT", "RTmin", "RTmax"]].copy().reset_index(drop=True)

        print(f"[ATTR] {source_file}: computing {len(info)}")
        calc = compute_peak_attributes(
            info,
            mzml_path=mzml_path,
            mz_tolerance=15.0,
            tolerance_unit="ppm",
            method="nearest",
            rt_tol_sec=30.0,
            include_literature_top=True,
        )

        calc = calc.drop_duplicates(subset=["Feature_ID"], keep="first")
        calc_cols = [
            c for c in calc.columns
            if c in {"Feature_ID", "SNR", "CV", "GS", "TPAS", "H2B", "ZZ", "DZZ", "PCC", "SKEW", "DENT", "DM", "ENT", "JAG"}
        ]
        merged = sub.merge(calc[calc_cols], on="Feature_ID", how="left")
        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True)
    return out


def _build_alignment_report(top_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, r in top_df.iterrows():
        stem = str(r["stem"])
        fid = str(r["Feature_ID"])
        d = IMAGES_ROOT / stem
        png = d / f"{fid}.png"
        js = d / f"{fid}.json"
        rows.append(
            {
                "source_file": str(r["source_file"]),
                "stem": stem,
                "Feature_ID": fid,
                "png_exists": png.exists(),
                "json_exists": js.exists(),
                "aligned": bool(png.exists() and js.exists()),
            }
        )
    return pd.DataFrame(rows)


def _replace_in_final(new_df: pd.DataFrame) -> tuple[int, int, int]:
    final = pd.read_csv(FINAL_CSV)
    
    # Drop unwanted columns leaving only specified columns
    COLS_TO_DROP = [
        'peak_apex_intensity', 'peak_area_auc', 'peak_snr_robust', 'peak_fwhm_min', 
        'peak_asymmetry_factor_10', 'peak_tailing_factor_5', 'peak_rt_skewness',
        'peak_rt_excess_kurtosis', 'peak_jaggedness', 'peak_gaussian_similarity',
        'peak_local_max_count', 'peak_mz_error_ppm_at_apex'
    ]
    drop_cols = [c for c in COLS_TO_DROP if c in final.columns]
    final = final.drop(columns=drop_cols)
    print(f"Columns remaining in final after dropping redundant: {final.columns.tolist()}")

    before_rows = len(final)

    target_files = {sf for _, sf in TARGETS}
    keep = final[~final["source_file"].isin(target_files)].copy().reset_index(drop=True)

    # Keep the original 10k rows unchanged; replace only previous 1650 rows.
    replaced_old = before_rows - len(keep)

    add = pd.DataFrame(index=range(len(new_df)), columns=final.columns)

    for c in ["source_file", "source_path", "Feature_ID", "mz", "RTmin", "RT", "RTmax"]:
        if c in add.columns and c in new_df.columns:
            add[c] = new_df[c].values

    base_cols = {"source_file", "source_path", "Feature_ID", "mz", "RTmin", "RT", "RTmax", "is_true_peak"}
    calc_cols = [c for c in new_df.columns if c in add.columns and c not in base_cols]
    for c in calc_cols:
        add[c] = new_df[c].values

    if "is_true_peak" in add.columns:
        add["is_true_peak"] = pd.NA

    out = pd.concat([keep, add], ignore_index=True)
    out = out.drop_duplicates(subset=["source_file", "Feature_ID"], keep="first").reset_index(drop=True)
    backup_path = _backup_final_csv_if_needed(FINAL_CSV, BACKUP_DIR)
    out.to_csv(FINAL_CSV, index=False)
    print(f"Final CSV saved with {len(out)} rows and {len(out.columns)} columns.")
    if backup_path is not None:
        print(f"Final CSV backup saved: {backup_path}")

    return before_rows, replaced_old, len(out)


def main() -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = None
    try:
        lock_fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except FileExistsError:
        raise RuntimeError(f"Another refresh run is already in progress: {LOCK_FILE}")

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(SOURCE_CSV)
    if not FINAL_CSV.exists():
        raise FileNotFoundError(FINAL_CSV)

    try:
        src = pd.read_csv(SOURCE_CSV)

        top = _select_top_by_area(src)
        TOP_AREA_CSV.parent.mkdir(parents=True, exist_ok=True)
        top.to_csv(TOP_AREA_CSV, index=False)

        top_with_attrs = _compute_all_features(top)

        align = _build_alignment_report(top)
        align.to_csv(ALIGN_CSV, index=False)

        before_rows, replaced_old, after_rows = _replace_in_final(top_with_attrs)

        # Build images LAST so that if pyopenms hangs, the CSV is already saved.
        _rebuild_images(top)

        selected_per_file = {k: int(v) for k, v in top.groupby("source_file").size().to_dict().items()}
        aligned_per_file = {k: int(v) for k, v in align.groupby("source_file")["aligned"].sum().to_dict().items()}

    # completeness check on new 1650 rows
        base_cols = {"source_file", "source_path", "stem", "Feature_ID", "mz", "RTmin", "RT", "RTmax", "maxo", "area", "is_true_peak"}
        present_cols = [c for c in top_with_attrs.columns if c not in base_cols]
        null_counts = {c: int(top_with_attrs[c].isna().sum()) for c in present_cols}

        summary = {
            "selection_rule": "top550_per_mzml_by_area_desc",
            "selected_rows": int(len(top)),
            "selected_per_file": selected_per_file,
            "alignment_rows": int(len(align)),
            "aligned_rows": int(align["aligned"].sum()),
            "misaligned_rows": int((~align["aligned"]).sum()),
            "aligned_per_file": aligned_per_file,
            "final_table_before_rows": int(before_rows),
            "replaced_old_rows": int(replaced_old),
            "final_table_after_rows": int(after_rows),
            "new_rows_feature_null_counts": null_counts,
            "outputs": {
                "final_csv": str(FINAL_CSV.resolve()),
                "top_area_csv": str(TOP_AREA_CSV.resolve()),
                "alignment_csv": str(ALIGN_CSV.resolve()),
                "images_root": str(IMAGES_ROOT.resolve()),
            },
        }

        SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("done")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()


if __name__ == "__main__":
    main()
