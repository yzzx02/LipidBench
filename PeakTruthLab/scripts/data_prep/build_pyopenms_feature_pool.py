from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.utils.data_io import load_pyopenms_results


DEFAULT_PEAK_ATTR_COLUMNS = [
    "peak_apex_intensity",
    "peak_area_auc",
    "peak_snr_robust",
    "peak_fwhm_min",
    "peak_asymmetry_factor_10",
    "peak_tailing_factor_5",
    "peak_jaggedness",
    "peak_gaussian_similarity",
    "peak_local_max_count",
    "peak_mz_error_ppm_at_apex",
]


def _normalize_single_table(df_raw: pd.DataFrame, source_file: str, source_path: str) -> pd.DataFrame:
    out = df_raw.copy()

    # Ensure core columns are present and numeric.
    for col in ["mz", "RT", "RTmin", "RTmax"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "RT" not in out.columns and "RTmin" in out.columns and "RTmax" in out.columns:
        out["RT"] = (out["RTmin"] + out["RTmax"]) / 2.0

    if "RTmin" not in out.columns and "RT" in out.columns:
        out["RTmin"] = out["RT"] - 0.1
    if "RTmax" not in out.columns and "RT" in out.columns:
        out["RTmax"] = out["RT"] + 0.1

    # Keep only valid rows for downstream EIC plotting.
    out = out.dropna(subset=["Feature_ID", "mz", "RT", "RTmin", "RTmax"]).copy()

    out["Feature_ID"] = out["Feature_ID"].astype(str).str.strip()
    out = out[out["Feature_ID"] != ""].copy()

    # Make Feature_ID globally unique after multi-file merge.
    stem = Path(source_file).stem
    out["Feature_ID"] = out["Feature_ID"].map(lambda x: f"{stem}__{x}")

    out["mz"] = pd.to_numeric(out["mz"], errors="coerce").round(4)
    out["RTmin"] = pd.to_numeric(out["RTmin"], errors="coerce").round(3)
    out["RT"] = pd.to_numeric(out["RT"], errors="coerce").round(3)
    out["RTmax"] = pd.to_numeric(out["RTmax"], errors="coerce").round(3)

    out["source_file"] = source_file
    out["source_path"] = source_path

    # Unified optional columns: reserve space for peak attributes and labels.
    for c in DEFAULT_PEAK_ATTR_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    if "is_true_peak" not in out.columns:
        out["is_true_peak"] = pd.NA

    ordered = [
        "source_file",
        "source_path",
        "Feature_ID",
        "mz",
        "RTmin",
        "RT",
        "RTmax",
        *DEFAULT_PEAK_ATTR_COLUMNS,
        "is_true_peak",
    ]
    return out[ordered].reset_index(drop=True)


def _extract_single_mzml(
    mzml_path: Path,
    mz_tol: float,
    min_fwhm: float,
    max_fwhm: float,
    noise: float,
    sn: float,
) -> pd.DataFrame:
    import pyopenms as oms  # local import to keep script import-light

    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml_path), exp)

    mass_traces = []
    mtd = oms.MassTraceDetection()
    mtd_par = mtd.getDefaults()
    mtd_par.setValue(b"mass_error_ppm", float(mz_tol))
    mtd_par.setValue(b"noise_threshold_int", float(noise))
    mtd_par.setValue(b"chrom_peak_snr", float(sn))
    mtd.setParameters(mtd_par)
    mtd.run(exp, mass_traces, 0)

    mass_traces_deconvol = []
    epd = oms.ElutionPeakDetection()
    epd_par = epd.getDefaults()
    epd_par.setValue(b"min_fwhm", float(min_fwhm))
    epd_par.setValue(b"max_fwhm", float(max_fwhm))
    epd_par.setValue(b"chrom_peak_snr", float(sn))
    epd.setParameters(epd_par)
    epd.detectPeaks(mass_traces, mass_traces_deconvol)

    feature_map = oms.FeatureMap()
    ffm = oms.FeatureFindingMetabo()
    ffm_par = ffm.getDefaults()
    ffm_par.setValue(b"local_rt_range", 8.0)
    ffm_par.setValue(b"local_mz_range", 3.5)
    ffm_par.setValue(b"mz_scoring_13C", b"true")
    ffm_par.setValue(b"report_convex_hulls", b"true")
    ffm_par.setValue(b"charge_upper_bound", 2)
    ffm.setParameters(ffm_par)
    ffm.run(mass_traces_deconvol, feature_map, [])

    df = feature_map.get_df()
    if df is None or df.empty:
        return pd.DataFrame()

    # pyOpenMS versions may emit lowercase underscore columns (rt_start/mz_end).
    # Map them to the canonical names expected by lipidbench.utils.data_io loader.
    rename_map = {
        "rt": "RT",
        "mz": "mz",
        "rt_start": "RTstart",
        "rt_end": "RTend",
        "mz_start": "MZstart",
        "mz_end": "MZend",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Reuse existing project normalization logic from lipidbench.utils.data_io.
    with tempfile.TemporaryDirectory(prefix="pyopenms_norm_") as tmp_dir:
        tmp_csv = Path(tmp_dir) / "raw_pyopenms.csv"
        df.to_csv(tmp_csv, index=False)
        std = load_pyopenms_results(
            file_path=tmp_csv,
            mz_tol=mz_tol,
            min_fwhm=min_fwhm,
            max_fwhm=max_fwhm,
            noise=noise,
            sn=sn,
            force_recompute_bounds=False,
        )

    if std is None or std.empty:
        return pd.DataFrame()
    return _normalize_single_table(std, source_file=mzml_path.name, source_path=str(mzml_path))


def _distribute_quotas(group_sizes: dict[str, int], target_n: int, min_per_file: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if total <= target_n:
        return dict(group_sizes)

    quotas = {k: 0 for k in group_sizes}

    # First pass: proportional allocation.
    for k, size in group_sizes.items():
        quotas[k] = min(size, int(round(target_n * (size / total))))

    # Ensure minimum per file where possible.
    if min_per_file > 0:
        for k, size in group_sizes.items():
            if size > 0 and quotas[k] < min_per_file:
                quotas[k] = min(size, min_per_file)

    # If over-allocated, trim from largest quotas first.
    over = sum(quotas.values()) - target_n
    if over > 0:
        keys = sorted(quotas, key=lambda x: quotas[x], reverse=True)
        i = 0
        while over > 0 and keys:
            k = keys[i % len(keys)]
            floor = min(min_per_file, group_sizes[k]) if min_per_file > 0 else 0
            if quotas[k] > floor:
                quotas[k] -= 1
                over -= 1
            i += 1
            if i > len(keys) * (target_n + 10):
                break

    # If under-allocated, add to groups with remaining capacity.
    under = target_n - sum(quotas.values())
    if under > 0:
        keys = sorted(group_sizes, key=lambda x: group_sizes[x] - quotas[x], reverse=True)
        i = 0
        while under > 0 and keys:
            k = keys[i % len(keys)]
            if quotas[k] < group_sizes[k]:
                quotas[k] += 1
                under -= 1
            i += 1
            if i > len(keys) * (target_n + 10):
                break

    return quotas


def _sample_stratified(df_all: pd.DataFrame, target_n: int, seed: int, min_per_file: int) -> pd.DataFrame:
    if len(df_all) <= target_n:
        return df_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    sizes = df_all.groupby("source_file").size().to_dict()
    quotas = _distribute_quotas(sizes, target_n=target_n, min_per_file=min_per_file)

    parts: list[pd.DataFrame] = []
    for src, q in quotas.items():
        g = df_all[df_all["source_file"] == src]
        if q <= 0:
            continue
        q = min(q, len(g))
        parts.append(g.sample(n=q, random_state=seed))

    sampled = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=df_all.columns)
    if len(sampled) > target_n:
        sampled = sampled.sample(n=target_n, random_state=seed)
    elif len(sampled) < target_n:
        remain = df_all.drop(sampled.index, errors="ignore")
        need = target_n - len(sampled)
        if need > 0 and len(remain) > 0:
            sampled = pd.concat(
                [sampled, remain.sample(n=min(need, len(remain)), random_state=seed)],
                ignore_index=True,
            )

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_pool(args: argparse.Namespace) -> None:
    mzml_root = Path(args.mzml_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mzml_root.exists():
        raise FileNotFoundError(f"mzml_root not found: {mzml_root}")

    mzml_files = sorted(mzml_root.rglob("*.mzML"))
    if not mzml_files:
        raise RuntimeError(f"No .mzML files found under: {mzml_root}")

    all_frames: list[pd.DataFrame] = []
    per_file_rows: list[dict[str, str | int]] = []

    for i, mzml in enumerate(mzml_files, start=1):
        print(f"[{i}/{len(mzml_files)}] extracting: {mzml.name}")
        try:
            df = _extract_single_mzml(
                mzml_path=mzml,
                mz_tol=args.mz_tol,
                min_fwhm=args.min_fwhm,
                max_fwhm=args.max_fwhm,
                noise=args.noise,
                sn=args.sn,
            )
            n = int(len(df))
            per_file_rows.append({"source_file": mzml.name, "rows": n})
            if n > 0:
                all_frames.append(df)
        except Exception as e:
            per_file_rows.append({"source_file": mzml.name, "rows": 0})
            print(f"  failed: {e}")

    if not all_frames:
        raise RuntimeError("No features were extracted from all mzML files.")

    df_all = pd.concat(all_frames, ignore_index=True)

    # Strict dedup on core identity fields after cross-file merge.
    core_key = ["source_file", "Feature_ID", "mz", "RT"]
    df_all = df_all.drop_duplicates(subset=[c for c in core_key if c in df_all.columns]).reset_index(drop=True)

    full_csv = out_dir / "feature_pool_full.csv"
    df_all.to_csv(full_csv, index=False)

    sampled = _sample_stratified(
        df_all=df_all,
        target_n=int(args.target_n),
        seed=int(args.seed),
        min_per_file=int(args.min_per_file),
    )
    sampled_csv = out_dir / f"feature_pool_sampled_{int(args.target_n)}.csv"
    sampled.to_csv(sampled_csv, index=False)

    summary = pd.DataFrame(per_file_rows)
    summary_csv = out_dir / "pyopenms_extraction_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("done")
    print(f"files:           {len(mzml_files)}")
    print(f"merged_rows:     {len(df_all)}")
    print(f"sampled_rows:    {len(sampled)}")
    print(f"merged_csv:      {full_csv}")
    print(f"sampled_csv:     {sampled_csv}")
    print(f"summary_csv:     {summary_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build feature pool from multiple mzML by pyOpenMS")
    p.add_argument("--mzml-root", type=str, required=True, help="Root directory containing mzML files")
    p.add_argument("--out-dir", type=str, default="PeakTruthLab/datasets", help="Output directory for CSV files")

    p.add_argument("--target-n", type=int, default=10000, help="Target sampled row count")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    p.add_argument("--min-per-file", type=int, default=200, help="Minimum sampled rows per file when possible")

    p.add_argument("--mz-tol", type=float, default=10.0)
    p.add_argument("--min-fwhm", type=float, default=2.5)
    p.add_argument("--max-fwhm", type=float, default=60.0)
    p.add_argument("--noise", type=float, default=1000.0)
    p.add_argument("--sn", type=float, default=3.0)

    args = p.parse_args()
    if args.target_n <= 0:
        raise ValueError("--target-n must be > 0")
    if args.min_per_file < 0:
        raise ValueError("--min-per-file must be >= 0")
    return args


if __name__ == "__main__":
    build_pool(parse_args())
