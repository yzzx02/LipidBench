from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lipidbench.eic.extract_eic_pyopenms import build
from lipidbench.utils.peak_attributes import compute_peak_attributes


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


@dataclass
class XcmsParams:
    polarity: str
    mz_tol: float
    minwidth: float
    maxwidth: float
    noise: float
    sn: float
    prefilter: int
    mzdiff: float


def _load_config_xcms_params(config_path: Path) -> XcmsParams:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    xcms = cfg.get("parameters", {}).get("xcms", {})
    peak = xcms.get("peak_picking", {})
    common = cfg.get("common_params", {})

    peakwidth = peak.get("peakwidth", [5, 50])
    if not isinstance(peakwidth, list) or len(peakwidth) != 2:
        peakwidth = [5, 50]

    return XcmsParams(
        polarity=str(xcms.get("polarity", "positive")),
        mz_tol=float(peak.get("ppm", common.get("mz_tolerance_ppm", 10.0))),
        minwidth=float(peakwidth[0]),
        maxwidth=float(peakwidth[1]),
        noise=float(peak.get("noise", 1000.0)),
        sn=float(peak.get("snthresh", 3.0)),
        prefilter=int(peak.get("prefilter_val", 3)),
        mzdiff=float(peak.get("mzdiff", 0.001)),
    )


def _parse_targets(v: str) -> list[str]:
    out = [x.strip() for x in v.split(",") if x.strip()]
    if not out:
        raise ValueError("--targets is empty")
    return out


def _run_single_xcms(mzml_path: Path, out_csv: Path, p: XcmsParams, tmp_root: Path) -> None:
    r_script = PROJECT_ROOT / "lipidbench" / "runners" / "xcms.R"
    run_dir = tmp_root / f"xcms_single_{mzml_path.stem}"
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    staged = run_dir / mzml_path.name
    try:
        staged.hardlink_to(mzml_path)
    except Exception:
        shutil.copy2(mzml_path, staged)

    cmd = [
        "Rscript",
        str(r_script),
        "--dir",
        str(run_dir),
        "--output",
        str(out_csv),
        "--polarity",
        str(p.polarity),
        "--mz_tol",
        str(p.mz_tol),
        "--minwidth",
        str(p.minwidth),
        "--maxwidth",
        str(p.maxwidth),
        "--noise",
        str(p.noise),
        "--sn",
        str(p.sn),
        "--prefilter",
        str(p.prefilter),
        "--mzdiff",
        str(p.mzdiff),
    ]
    subprocess.run(cmd, check=True)

    shutil.rmtree(run_dir, ignore_errors=True)


def _extract_single_file_table(raw_csv: Path, source_file: str, source_path: str, top_n: int) -> pd.DataFrame:
    x = pd.read_csv(raw_csv)
    need = ["Row.names", "mzmed", "rtmed", "rtmin", "rtmax", "maxo", "into"]
    miss = [c for c in need if c not in x.columns]
    if miss:
        raise ValueError(f"xcms output missing columns {miss}: {raw_csv}")

    stem = source_file.replace(".mzML", "")

    x["mz"] = pd.to_numeric(x["mzmed"], errors="coerce")
    x["RT"] = pd.to_numeric(x["rtmed"], errors="coerce") / 60.0
    x["RTmin"] = pd.to_numeric(x["rtmin"], errors="coerce") / 60.0
    x["RTmax"] = pd.to_numeric(x["rtmax"], errors="coerce") / 60.0
    x["maxo"] = pd.to_numeric(x["maxo"], errors="coerce")
    x["area"] = pd.to_numeric(x["into"], errors="coerce")

    x = x.dropna(subset=["mz", "RT", "RTmin", "RTmax", "maxo", "area"]).copy()
    x = x.sort_values(["area", "maxo"], ascending=[False, False], kind="mergesort")

    if top_n > 0:
        x = x.head(top_n).copy()

    x["row_id"] = pd.to_numeric(x["Row.names"], errors="coerce")
    x["row_id"] = x["row_id"].fillna(pd.Series(range(1, len(x) + 1), index=x.index)).astype(int)
    x["Feature_ID"] = x["row_id"].map(lambda i: f"{stem}__F{i}")
    x["source_file"] = source_file
    x["source_path"] = source_path

    out = x[["source_file", "source_path", "Feature_ID", "mz", "RTmin", "RT", "RTmax", "maxo", "area"]].copy()
    out = out.drop_duplicates(subset=["Feature_ID"], keep="first")
    out = out.reset_index(drop=True)
    return out


def _compute_attrs(df_one: pd.DataFrame, mzml_path: Path, mz_tolerance: float) -> pd.DataFrame:
    info = df_one[["Feature_ID", "mz", "RT", "RTmin", "RTmax"]].copy()
    calc = compute_peak_attributes(
        info,
        mzml_path=mzml_path,
        mz_tolerance=float(mz_tolerance),
        tolerance_unit="ppm",
        method="nearest",
        rt_tol_sec=30.0,
        include_literature_top=True,
    )

    attr_cols = [
        "Feature_ID",
        "SNR",
        "CV",
        "GS",
        "TPAS",
        "H2B",
        "ZZ",
        "DZZ",
        "PCC",
        "SKEW",
        "DENT",
        "DM",
        "ENT",
        "JAG",
    ]
    calc = calc[[c for c in attr_cols if c in calc.columns]].copy()
    merged = df_one.merge(calc, on="Feature_ID", how="left")
    return merged


def _draw_images(df_one: pd.DataFrame, mzml_path: Path, images_root: Path, mz_tolerance: float) -> None:
    stem = mzml_path.stem
    stem_dir = images_root / stem
    if stem_dir.exists():
        shutil.rmtree(stem_dir, ignore_errors=True)

    args = SimpleNamespace(
        processes_number=1,
        method="nearest",
        unit="ppm",
        tolerance=float(mz_tolerance),
        images_path=str(images_root),
        smooth_sigma=0.0,
    )

    info = df_one[["Feature_ID", "mz", "RT", "RTmin", "RTmax"]].copy().reset_index(drop=True)

    ok = 0
    chunk_size = 120
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
                fid = str(one.iloc[0]["Feature_ID"])
                print(f"[WARN] draw fail: {mzml_path.name} {fid} -> {e}")

    png_ids = {p.stem for p in stem_dir.glob("*.png")}
    expected_ids = set(info["Feature_ID"].astype(str))
    missing = expected_ids - png_ids
    if missing:
        raise RuntimeError(f"image generation incomplete for {mzml_path.name}: missing_png={len(missing)}")

    print(f"[EIC] {mzml_path.name}: generated={ok}/{len(info)}")


def _verify_alignment(df_subset: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    rows = []
    for _, r in df_subset.iterrows():
        source_file = str(r["source_file"])
        stem = source_file.replace(".mzML", "")
        fid = str(r["Feature_ID"])
        png = images_root / stem / f"{fid}.png"
        js = images_root / stem / f"{fid}.json"
        rows.append(
            {
                "source_file": source_file,
                "Feature_ID": fid,
                "png_exists": png.exists(),
                "json_exists": js.exists(),
                "aligned": bool(png.exists() and js.exists()),
            }
        )
    return pd.DataFrame(rows)


def rebuild(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    final_csv = Path(args.final_csv).resolve()
    images_root = Path(args.images_root).resolve()
    xcms_out_root = Path(args.xcms_out_root).resolve()
    data_root = Path(args.data_root).resolve()
    report_csv = Path(args.report_csv).resolve()
    backup_dir = Path(args.backup_dir).resolve()

    targets = _parse_targets(args.targets)

    if not final_csv.exists():
        raise FileNotFoundError(f"final csv not found: {final_csv}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    params = _load_config_xcms_params(config_path)
    final_df = pd.read_csv(final_csv)

    old_counts = (
        final_df[final_df["source_file"].isin(targets)]
        .groupby("source_file")
        .size()
        .to_dict()
    )

    xcms_out_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    rebuilt_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for source_file in targets:
        mzml_path = data_root / source_file
        if not mzml_path.exists():
            raise FileNotFoundError(f"missing target mzML: {mzml_path}")

        keep_n = int(old_counts.get(source_file, 0))
        if keep_n <= 0:
            keep_n = int(args.default_top_n)

        stem = mzml_path.stem
        xcms_raw = xcms_out_root / f"{stem}_xcms_raw.csv"

        print(f"[XCMS] {source_file} keep_n={keep_n}")
        _run_single_xcms(mzml_path=mzml_path, out_csv=xcms_raw, p=params, tmp_root=xcms_out_root)

        one = _extract_single_file_table(
            raw_csv=xcms_raw,
            source_file=source_file,
            source_path=str(mzml_path),
            top_n=keep_n,
        )

        if len(one) < keep_n:
            print(f"[WARN] {source_file}: only {len(one)} features from XCMS, expected {keep_n}")

        one = _compute_attrs(one, mzml_path=mzml_path, mz_tolerance=params.mz_tol)
        _draw_images(one, mzml_path=mzml_path, images_root=images_root, mz_tolerance=params.mz_tol)

        rebuilt_parts.append(one)
        summary_rows.append(
            {
                "source_file": source_file,
                "old_count": int(old_counts.get(source_file, 0)),
                "new_count": int(len(one)),
                "xcms_csv": str(xcms_raw),
            }
        )

    rebuilt = pd.concat(rebuilt_parts, ignore_index=True)

    keep_old = final_df[~final_df["source_file"].isin(targets)].copy().reset_index(drop=True)

    out_cols = list(final_df.columns)
    add = pd.DataFrame(index=range(len(rebuilt)), columns=out_cols)
    for c in out_cols:
        if c in rebuilt.columns:
            add[c] = rebuilt[c].values
        elif c == "is_true_peak":
            add[c] = pd.NA

    new_final = pd.concat([keep_old, add], ignore_index=True)
    backup_path = _backup_final_csv_if_needed(final_csv, backup_dir)
    new_final.to_csv(final_csv, index=False)

    align = _verify_alignment(add, images_root=images_root)
    bad = align[~align["aligned"]]

    report_rows = summary_rows.copy()
    if len(bad) > 0:
        report_rows.append(
            {
                "source_file": "__ALIGNMENT__",
                "old_count": "",
                "new_count": "",
                "xcms_csv": f"bad_pairs={len(bad)}",
            }
        )
    pd.DataFrame(report_rows).to_csv(report_csv, index=False)

    print("done")
    print(f"final_csv:        {final_csv}")
    print(f"images_root:      {images_root}")
    print(f"targets:          {len(targets)}")
    print(f"replaced_rows:    {len(add)}")
    print(f"alignment_bad:    {len(bad)}")
    print(f"report_csv:       {report_csv}")
    if backup_path is not None:
        print(f"backup_csv:       {backup_path}")

    if len(bad) > 0:
        bad_csv = report_csv.with_name(report_csv.stem + "_alignment_bad.csv")
        bad.to_csv(bad_csv, index=False)
        print(f"alignment_bad_csv:{bad_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Rebuild subset dataset from XCMS and replace rows/images one-to-one")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--final-csv", type=str, default="PeakTruthLab/datasets/feature_table_final_10000.csv")
    p.add_argument("--images-root", type=str, default="PeakTruthLab/datasets/eic_images_flat")
    p.add_argument("--data-root", type=str, default="data/ceshiji")
    p.add_argument("--xcms-out-root", type=str, default="results/xcms_subset_rebuild")
    p.add_argument("--report-csv", type=str, default="PeakTruthLab/results/rebuild_subset_report.csv")
    p.add_argument("--backup-dir", type=str, default="PeakTruthLab/datasets/backups")
    p.add_argument(
        "--targets",
        type=str,
        required=True,
        help="Comma-separated source_file names with .mzML",
    )
    p.add_argument("--default-top-n", type=int, default=523)
    args = p.parse_args()
    if args.default_top_n <= 0:
        raise ValueError("--default-top-n must be > 0")
    return args


if __name__ == "__main__":
    rebuild(parse_args())
