from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PeakTruthLab.scripts.data_prep.rebuild_dataset_from_xcms_subset import (  # noqa: E402
    _compute_attrs,
    _draw_images,
    _extract_single_file_table,
    _load_config_xcms_params,
    _run_single_xcms,
)


def _normalize_target_name(v: str) -> str:
    name = str(v).strip()
    if not name:
        return ""
    if not name.lower().endswith(".mzml"):
        name = f"{name}.mzML"
    return name


def _parse_targets(v: str) -> list[str]:
    out = [_normalize_target_name(x) for x in str(v).split(",") if str(x).strip()]
    out = [x for x in out if x]
    if not out:
        raise ValueError("--targets is empty")
    return out


def _verify_alignment(df_subset: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    rows = []
    for _, r in df_subset.iterrows():
        source_file = str(r["source_file"])
        stem = Path(source_file).stem
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
                "png_path": str(png),
                "json_path": str(js),
            }
        )
    return pd.DataFrame(rows)


def _write_batch_readme(
    *,
    batch_root: Path,
    images_root: Path,
    features_csv: Path,
    manifest_csv: Path,
    summary_json: Path,
    targets: list[str],
    sample_n_per_file: int,
) -> None:
    text = "\n".join(
        [
            f"Batch root: {batch_root}",
            f"Images root: {images_root}",
            f"Features CSV: {features_csv}",
            f"Manifest CSV: {manifest_csv}",
            f"Summary JSON: {summary_json}",
            f"Targets ({len(targets)}): {', '.join(targets)}",
            f"Sampling rule: random {sample_n_per_file} per mzML after XCMS filtering",
            "This batch is isolated for annotation and is not merged into feature_table_final_10000.csv yet.",
        ]
    )
    (batch_root / "README.txt").write_text(text, encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    data_root = Path(args.data_root).resolve()
    batch_root = Path(args.batch_root).resolve()
    xcms_out_root = batch_root / "xcms_raw"
    images_root = batch_root / "eic_images"
    features_csv = batch_root / "sampled_features_random550_per_file.csv"
    manifest_csv = batch_root / "annotation_manifest.csv"
    align_csv = batch_root / "alignment_report.csv"
    summary_json = batch_root / "summary.json"

    targets = _parse_targets(args.targets)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    params = _load_config_xcms_params(config_path)
    batch_root.mkdir(parents=True, exist_ok=True)
    xcms_out_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    rebuilt_parts: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for source_file in targets:
        mzml_path = data_root / source_file
        if not mzml_path.exists():
            raise FileNotFoundError(f"missing target mzML: {mzml_path}")

        stem = mzml_path.stem
        xcms_raw = xcms_out_root / f"{stem}_xcms_raw.csv"

        print(f"[XCMS] {source_file}")
        if bool(args.reuse_xcms_raw) and xcms_raw.exists():
            print(f"[XCMS] reuse existing raw csv: {xcms_raw}")
        else:
            _run_single_xcms(mzml_path=mzml_path, out_csv=xcms_raw, p=params, tmp_root=xcms_out_root)

        one = _extract_single_file_table(
            raw_csv=xcms_raw,
            source_file=source_file,
            source_path=str(mzml_path),
            top_n=int(args.sample_n_per_file),
            sample_random=True,
            seed=int(args.seed),
        )
        one = _compute_attrs(one, mzml_path=mzml_path, mz_tolerance=float(args.eic_mz_tolerance))
        _draw_images(one, mzml_path=mzml_path, images_root=images_root, mz_tolerance=float(args.eic_mz_tolerance))

        one["image_rel"] = one["Feature_ID"].astype(str).map(lambda fid: str(Path(stem) / f"{fid}.png"))
        one["json_rel"] = one["Feature_ID"].astype(str).map(lambda fid: str(Path(stem) / f"{fid}.json"))
        one["batch_name"] = batch_root.name
        one["annotation_status"] = "pending"
        rebuilt_parts.append(one)

        summary_rows.append(
            {
                "source_file": source_file,
                "sampled_rows": int(len(one)),
                "xcms_raw_csv": str(xcms_raw),
            }
        )

    sampled = pd.concat(rebuilt_parts, ignore_index=True)
    sampled = sampled.drop_duplicates(subset=["source_file", "Feature_ID"], keep="first").reset_index(drop=True)
    sampled.to_csv(features_csv, index=False)

    align = _verify_alignment(sampled, images_root=images_root)
    align.to_csv(align_csv, index=False)

    manifest = sampled[
        [
            "source_file",
            "source_path",
            "Feature_ID",
            "mz",
            "RTmin",
            "RT",
            "RTmax",
            "maxo",
            "area",
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
            "image_rel",
            "json_rel",
            "annotation_status",
            "batch_name",
        ]
    ].copy()
    manifest["image_abs"] = manifest["image_rel"].map(lambda p: str((images_root / p).resolve()))
    manifest["json_abs"] = manifest["json_rel"].map(lambda p: str((images_root / p).resolve()))
    manifest.to_csv(manifest_csv, index=False)

    summary = {
        "batch_name": batch_root.name,
        "created_at": datetime.now().isoformat(),
        "targets": targets,
        "sample_n_per_file": int(args.sample_n_per_file),
        "seed": int(args.seed),
        "xcms_params": {
            "polarity": params.polarity,
            "ppm": params.mz_tol,
            "minwidth": params.minwidth,
            "maxwidth": params.maxwidth,
            "noise": params.noise,
            "snthresh": params.sn,
            "prefilter_val": params.prefilter,
            "mzdiff": params.mzdiff,
        },
        "eic_attr_ppm": float(args.eic_mz_tolerance),
        "selected_rows_total": int(len(sampled)),
        "selected_per_file": {str(k): int(v) for k, v in sampled.groupby("source_file").size().to_dict().items()},
        "aligned_rows": int(align["aligned"].sum()),
        "misaligned_rows": int((~align["aligned"]).sum()),
        "outputs": {
            "batch_root": str(batch_root),
            "features_csv": str(features_csv),
            "manifest_csv": str(manifest_csv),
            "images_root": str(images_root),
            "align_csv": str(align_csv),
        },
        "per_file_reports": summary_rows,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_batch_readme(
        batch_root=batch_root,
        images_root=images_root,
        features_csv=features_csv,
        manifest_csv=manifest_csv,
        summary_json=summary_json,
        targets=targets,
        sample_n_per_file=int(args.sample_n_per_file),
    )

    print("done")
    print(f"batch_root:      {batch_root}")
    print(f"targets:         {len(targets)}")
    print(f"selected_total:  {len(sampled)}")
    print(f"aligned_bad:     {int((~align['aligned']).sum())}")
    print(f"features_csv:    {features_csv}")
    print(f"manifest_csv:    {manifest_csv}")
    print(f"images_root:     {images_root}")
    print(f"summary_json:    {summary_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prepare a standalone XCMS annotation batch for new mzML files")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--data-root", type=str, default="data/ceshiji")
    p.add_argument(
        "--batch-root",
        type=str,
        default="PeakTruthLab/datasets/annotation_batches/sphingolipid_random550_20260330",
    )
    p.add_argument(
        "--targets",
        type=str,
        required=True,
        help="Comma-separated source_file names or stems, e.g. A.mzML,B",
    )
    p.add_argument("--sample-n-per-file", type=int, default=550)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eic-mz-tolerance", type=float, default=15.0)
    p.add_argument("--reuse-xcms-raw", action="store_true")
    args = p.parse_args()
    if args.sample_n_per_file <= 0:
        raise ValueError("--sample-n-per-file must be > 0")
    if args.eic_mz_tolerance <= 0:
        raise ValueError("--eic-mz-tolerance must be > 0")
    return args


if __name__ == "__main__":
    prepare(parse_args())
