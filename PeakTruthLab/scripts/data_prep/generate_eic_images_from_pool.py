from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.eic.extract_eic_pyopenms import build


REQUIRED_COLS = ["Feature_ID", "mz", "RT", "RTmin", "RTmax", "source_file", "source_path"]


def _rt_range_minmax(mzml_path: Path) -> tuple[float, float] | None:
    try:
        import pymzml  # type: ignore
    except Exception:
        return None

    rt_min = None
    rt_max = None
    try:
        reader = pymzml.run.Reader(str(mzml_path))
        for spec in reader:
            ms_level = int(getattr(spec, "ms_level", 1) or 1)
            if ms_level != 1:
                continue
            rt = float(spec.scan_time_in_minutes())
            if rt_min is None or rt < rt_min:
                rt_min = rt
            if rt_max is None or rt > rt_max:
                rt_max = rt
    except Exception:
        return None

    if rt_min is None or rt_max is None:
        return None
    return float(rt_min), float(rt_max)


def _validate_pool(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"feature pool missing columns: {missing}")

    out = df.copy()
    for c in ["mz", "RT", "RTmin", "RTmax"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["Feature_ID"] = out["Feature_ID"].astype(str).str.strip()
    out["source_file"] = out["source_file"].astype(str).str.strip()
    out["source_path"] = out["source_path"].astype(str).str.strip()

    out = out.dropna(subset=["Feature_ID", "mz", "RT", "RTmin", "RTmax", "source_file", "source_path"]).copy()
    out = out[out["Feature_ID"] != ""].copy()
    return out.reset_index(drop=True)


def generate(args: argparse.Namespace) -> None:
    pool_csv = Path(args.pool_csv).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not pool_csv.exists():
        raise FileNotFoundError(f"pool csv not found: {pool_csv}")

    df = pd.read_csv(pool_csv)
    df = _validate_pool(df)

    only_sources = [s.strip() for s in str(args.only_source_files).split(",") if s.strip()]
    if only_sources:
        df = df[df["source_file"].isin(only_sources)].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(f"No rows left after --only-source-files filter: {only_sources}")

    if args.max_total > 0 and len(df) > args.max_total:
        df = df.sample(n=args.max_total, random_state=args.seed).reset_index(drop=True)

    groups = []
    for src, g in df.groupby("source_path", sort=True):
        gg = g.copy()
        if args.max_per_file > 0 and len(gg) > args.max_per_file:
            gg = gg.sample(n=args.max_per_file, random_state=args.seed)
        groups.append((Path(src), gg[["Feature_ID", "mz", "RT", "RTmin", "RTmax"]].reset_index(drop=True)))

    done_files = 0
    done_rows = 0
    skipped_files = 0
    failed_files = 0
    fail_rows: list[dict[str, str]] = []
    rt_range_cache: dict[Path, tuple[float, float] | None] = {}

    for i, (mzml_path, info_df) in enumerate(groups, start=1):
        print(f"[{i}/{len(groups)}] preparing: {mzml_path.name} rows={len(info_df)}")
        if not mzml_path.exists():
            skipped_files += 1
            print(f"[{i}/{len(groups)}] skip missing mzML: {mzml_path}")
            continue
        if info_df.empty:
            skipped_files += 1
            print(f"[{i}/{len(groups)}] skip empty group: {mzml_path.name}")
            continue

        rt_range = rt_range_cache.get(mzml_path)
        if mzml_path not in rt_range_cache:
            rt_range = _rt_range_minmax(mzml_path)
            rt_range_cache[mzml_path] = rt_range
        if rt_range is not None:
            lo, hi = rt_range
            before = len(info_df)
            info_df = info_df[(info_df["RT"] > lo) & (info_df["RT"] < hi)].reset_index(drop=True)
            if len(info_df) == 0:
                skipped_files += 1
                print(f"[{i}/{len(groups)}] skip out-of-range RT: {mzml_path.name} (before={before}, after=0)")
                continue

        print(f"[{i}/{len(groups)}] exporting EICs: {mzml_path.name} rows={len(info_df)}")

        run_args = SimpleNamespace(
            processes_number=max(1, int(args.processes_number)),
            method=args.method,
            unit=args.unit,
            tolerance=float(args.tolerance),
            images_path=str(out_root),
            smooth_sigma=float(args.smooth_sigma),
        )

        chunk_size = max(1, int(args.chunk_size))
        file_failed = False
        for start in range(0, len(info_df), chunk_size):
            part = info_df.iloc[start : start + chunk_size].reset_index(drop=True)
            try:
                build(paths=[mzml_path], info=part, plot=True, args=run_args)
                done_rows += len(part)
            except Exception as e:
                file_failed = True
                msg = str(e).replace("\n", " ")
                fail_rows.append(
                    {
                        "source_file": mzml_path.name,
                        "source_path": str(mzml_path),
                        "error": msg,
                    }
                )
                print(f"[{i}/{len(groups)}] chunk failed: {mzml_path.name} start={start} -> {msg}")
                break

        if file_failed:
            failed_files += 1
            continue

        done_files += 1

    print("done")
    print(f"pool_csv:       {pool_csv}")
    print(f"output_root:    {out_root}")
    print(f"files_exported: {done_files}")
    print(f"files_skipped:  {skipped_files}")
    print(f"files_failed:   {failed_files}")
    print(f"rows_exported:  {done_rows}")

    if fail_rows:
        fail_csv = out_root / "generation_failures.csv"
        pd.DataFrame(fail_rows).to_csv(fail_csv, index=False)
        print(f"failures_csv:   {fail_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate EIC images and initial LabelMe boxes from feature pool")
    p.add_argument(
        "--pool-csv",
        type=str,
        default="PeakTruthLab/datasets/feature_pool_sampled_10000.csv",
        help="Feature pool CSV with Feature_ID/mz/RT/RTmin/RTmax/source_path",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="PeakTruthLab/datasets/eic_images_pool",
        help="Output root for EIC images and LabelMe json files",
    )
    p.add_argument("--method", type=str, default="nearest", choices=["nearest", "window_sum"])
    p.add_argument("--unit", type=str, default="ppm", choices=["ppm", "Da", "da"])
    p.add_argument("--tolerance", type=float, default=10.0)
    p.add_argument("--smooth-sigma", type=float, default=0.0)
    p.add_argument("--processes-number", type=int, default=1)

    p.add_argument("--max-total", type=int, default=0, help="Debug limit on total rows (0 means all)")
    p.add_argument("--max-per-file", type=int, default=0, help="Debug limit per source file (0 means all)")
    p.add_argument("--chunk-size", type=int, default=300, help="Rows per extraction chunk to reduce memory usage")
    p.add_argument(
        "--only-source-files",
        type=str,
        default="",
        help="Optional comma-separated source_file names to process, e.g. 'A.mzML,B.mzML'",
    )
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.tolerance <= 0:
        raise ValueError("--tolerance must be > 0")
    if args.processes_number < 1:
        raise ValueError("--processes-number must be >= 1")
    if args.max_total < 0 or args.max_per_file < 0:
        raise ValueError("--max-total and --max-per-file must be >= 0")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    return args


if __name__ == "__main__":
    generate(parse_args())
