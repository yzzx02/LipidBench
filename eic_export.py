import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EIC plots without GUI")
    p.add_argument("--mzml", required=True, help="Path to a single .mzML file")
    p.add_argument("--algo", required=True, choices=["asari", "pyopenms", "xcms", "msdial"], help="Which algorithm's feature table to use")
    p.add_argument("--results-dir", required=True, help="Results directory containing xcms/pyopenms/asari/msdial subfolders")
    p.add_argument("--ppm", type=float, default=10.0, help="m/z tolerance in ppm for nearest-point EIC")
    p.add_argument("--out-dir", required=True, help="Output directory for EIC images")
    p.add_argument("--max-features", type=int, default=200, help="Limit number of features exported")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mzml_path = Path(args.mzml).resolve()
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not mzml_path.exists():
        raise FileNotFoundError(mzml_path)
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)

    from lipidbench.utils.feature_table_io import (
        find_feature_table,
        load_feature_table,
        standardize_rt_columns_for_display,
        suggest_peak_column,
    )
    from lipidbench.eic.extract_eic_pyopenms import extract_eic_nearest_ppm

    if args.algo == "pyopenms" and sys.version_info >= (3, 13):
        raise RuntimeError(
            "pyopenms 在 Windows 上通常不支持 Python 3.13（会出现 DLL load failed）。"
            "请改用 Python 3.9/3.10/3.11 的虚拟环境来运行。"
        )

    try:
        import pyopenms as oms
    except Exception as e:
        raise RuntimeError(f"pyopenms not available: {e}")

    feature_path = find_feature_table(results_dir, args.algo)
    df_raw = load_feature_table(feature_path, args.algo)
    df = standardize_rt_columns_for_display(df_raw, args.algo)

    peak_col = suggest_peak_column(df_raw, args.algo)
    if peak_col is None:
        # best effort: choose first numeric-ish column
        for c in df_raw.columns:
            if c in ("Feature_ID", "mz", "RT", "RTmin", "RTmax", "rtime"):
                continue
            if str(c).endswith(".mzML") or c in ("peak_area", "Area", "intensity"):
                peak_col = str(c)
                break

    out_dir.mkdir(parents=True, exist_ok=True)

    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(mzml_path), exp)

    try:
        from matplotlib.figure import Figure
    except Exception as e:
        raise RuntimeError(f"matplotlib not available: {e}")

    n = min(int(args.max_features), len(df))
    for i in range(n):
        row = df.iloc[i]
        mz = row.get("mz")
        if mz is None or pd.isna(mz):
            continue
        mz = float(mz)

        rtmin = row.get("RTmin")
        rtmax = row.get("RTmax")
        rtmin = float(rtmin) if pd.notna(rtmin) else None
        rtmax = float(rtmax) if pd.notna(rtmax) else None

        feature_id = str(row.get("Feature_ID", f"F{i+1}"))

        trace = extract_eic_nearest_ppm(
            exp,
            target_mz=mz,
            ppm=float(args.ppm),
            rt_min_limit=(rtmin - 0.2) if rtmin is not None else None,
            rt_max_limit=(rtmax + 0.2) if rtmax is not None else None,
            ms_level=1,
        )

        fig = Figure(figsize=(4, 3), dpi=100)  # 400x300
        ax = fig.add_subplot(111)
        ax.plot(trace.rt_min, trace.intensity, linewidth=1.0)
        ax.set_xlabel("RT (min)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"{mzml_path.stem} | {feature_id} | m/z={mz:.4f}")
        fig.tight_layout()

        out_path = out_dir / f"{feature_id}.png"
        fig.savefig(out_path, dpi=100)

    print(f"Exported EIC images to: {out_dir}")
    if peak_col:
        print(f"Default PeakArea column suggestion: {peak_col}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
