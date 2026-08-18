from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lipidbench.utils.peak_attributes import (  # noqa: E402
    PEAK_ATTRIBUTE_COLUMNS,
    _compute_one_feature_attributes,
    _extract_trace,
    load_ms1_spectra,
)


ATTR13 = PEAK_ATTRIBUTE_COLUMNS[:13]
ATTR16 = PEAK_ATTRIBUTE_COLUMNS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_part_name(source_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("_")
    digest = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:10]
    return f"{stem}.{digest}.csv"


def index_mzml_files(roots: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for root in roots:
        root = root.resolve()
        if root.is_file():
            if root.suffix.lower() == ".mzml":
                index.setdefault(root.name.lower(), []).append(root)
            continue
        if not root.is_dir():
            raise FileNotFoundError(f"source root does not exist: {root}")
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(".mzml"):
                    path = Path(dirpath) / filename
                    index.setdefault(filename.lower(), []).append(path)
    return index


def resolve_sources(source_inventory: pd.DataFrame, roots: list[Path]) -> pd.DataFrame:
    index = index_mzml_files(roots)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    conflicting: list[dict[str, Any]] = []
    for _, source in source_inventory.iterrows():
        name = str(source["source_mzml_name"])
        matches = sorted(set(index.get(name.lower(), [])), key=lambda p: str(p).lower())
        if not matches:
            missing.append(name)
            continue
        hashes = {str(path): sha256_file(path) for path in matches}
        unique_hashes = sorted(set(hashes.values()))
        if len(unique_hashes) != 1:
            conflicting.append({"source": name, "matches": hashes})
            continue
        chosen = matches[0]
        rows.append(
            {
                "source_mzml_name": name,
                "resolved_path": str(chosen.resolve()),
                "sha256": unique_hashes[0],
                "size_bytes": chosen.stat().st_size,
                "duplicate_path_count": len(matches),
                "all_matching_paths": "|".join(str(path.resolve()) for path in matches),
                "seed_job_count": int(source["seed_job_count"]),
                "peak_job_count": int(source["peak_job_count"]),
                "eic_tolerance_ppm": float(source["eic_tolerance_ppm"]),
            }
        )
    if missing or conflicting:
        raise RuntimeError(
            json.dumps(
                {
                    "missing_sources": missing,
                    "conflicting_same_name_sources": conflicting,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    result = pd.DataFrame(rows).sort_values("source_mzml_name").reset_index(drop=True)
    if len(result) != len(source_inventory):
        raise RuntimeError("source resolution count mismatch")
    return result


def compute_source_part(
    jobs: pd.DataFrame,
    source_path: Path,
    source_hash: str,
    tolerance_ppm: float,
    backend: str,
) -> pd.DataFrame:
    spectra = load_ms1_spectra(source_path, backend=backend)
    if not spectra:
        raise RuntimeError(f"no MS1 spectra loaded from {source_path}")
    spectra = sorted(spectra, key=lambda spectrum: spectrum.rt_min)
    spectrum_rts = [float(spectrum.rt_min) for spectrum in spectra]
    output: list[dict[str, Any]] = []

    for _, job in jobs.iterrows():
        lo = float(min(job["RTmin"], job["RTmax"]))
        hi = float(max(job["RTmin"], job["RTmax"]))
        left = bisect.bisect_left(spectrum_rts, lo)
        right = bisect.bisect_right(spectrum_rts, hi)
        subset = spectra[left:right]
        result: dict[str, Any] = {
            "job_id": str(job["job_id"]),
            "entity_type": str(job["entity_type"]),
            "sample_id": str(job["sample_id"]),
            "source_mzml_name": str(job["source_mzml_name"]),
            "source_mzml_sha256": source_hash,
            "source_ms1_spectrum_count": len(spectra),
            "window_ms1_spectrum_count": len(subset),
            "error": "",
        }
        try:
            if not subset:
                raise RuntimeError(f"no spectra in RT window [{lo}, {hi}]")
            rt, eic, mass = _extract_trace(
                subset,
                target_mz=float(job["mz"]),
                tolerance=float(tolerance_ppm),
                unit="ppm",
                method="nearest",
            )
            attrs = _compute_one_feature_attributes(
                rt,
                eic,
                mass,
                target_mz=float(job["mz"]),
                target_rt_min=float(job["RT"]),
                target_rtmin=lo,
                target_rtmax=hi,
                rt_tol_sec=30.0,
                include_literature_top=True,
            )
            for name in ATTR16:
                result[name] = attrs.get(name, np.nan)
        except Exception as exc:  # keep a complete, auditable error table
            result["error"] = f"{type(exc).__name__}: {exc}"
            for name in ATTR16:
                result[name] = np.nan
        output.append(result)
    return pd.DataFrame(output)


def compare_legacy(
    jobs: pd.DataFrame,
    calculated: pd.DataFrame,
    *,
    exact_only: bool,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    merged = jobs.merge(
        calculated[["job_id", *ATTR13]],
        on="job_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_calc"),
    )
    if exact_only:
        merged = merged[merged["legacy_match_status"] == "exact"].copy()
    summary: dict[str, Any] = {"compared_rows": int(len(merged)), "attributes": {}}
    total_mismatches = 0
    for name in ATTR13:
        legacy = pd.to_numeric(merged[f"legacy_{name}"], errors="coerce").to_numpy(dtype=float)
        calc = pd.to_numeric(merged[name], errors="coerce").to_numpy(dtype=float)
        comparable = np.isfinite(legacy)
        equal = np.zeros(len(merged), dtype=bool)
        equal[comparable] = np.isclose(
            legacy[comparable], calc[comparable], rtol=rtol, atol=atol, equal_nan=False
        )
        mismatch = comparable & ~equal
        diffs = np.abs(legacy[comparable & np.isfinite(calc)] - calc[comparable & np.isfinite(calc)])
        mismatch_count = int(mismatch.sum())
        total_mismatches += mismatch_count
        summary["attributes"][name] = {
            "comparable": int(comparable.sum()),
            "mismatch_count": mismatch_count,
            "max_abs_diff": float(np.max(diffs)) if diffs.size else None,
        }
    summary["total_attribute_mismatches"] = total_mismatches
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RTX4070 handoff attributes on the workstation that holds the original mzML files."
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=REPO_ROOT / "PeakTruthLab/datasets/rtx4070_attribute_handoff_20260814",
    )
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("auto", "pyopenms", "pymzml"), default="auto")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()

    jobs_dir = args.jobs_dir.resolve()
    output_dir = args.output_dir.resolve()
    parts_dir = output_dir / "parts"
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    seed_jobs = pd.read_csv(jobs_dir / "old_final_seed_attribute_jobs.csv", encoding="utf-8-sig", low_memory=False)
    peak_jobs = pd.read_csv(jobs_dir / "old_final_peak_attribute_jobs.csv", encoding="utf-8-sig", low_memory=False)
    source_inventory = pd.read_csv(jobs_dir / "source_inventory.csv", encoding="utf-8-sig")
    if len(seed_jobs) != 15317 or seed_jobs["job_id"].nunique() != 15317:
        raise RuntimeError("Seed job input count/uniqueness mismatch")
    if len(peak_jobs) != 18580 or peak_jobs["job_id"].nunique() != 18580:
        raise RuntimeError("peak job input count/uniqueness mismatch")

    source_resolution = resolve_sources(source_inventory, args.source_root)
    source_resolution_path = output_dir / "source_resolution.csv"
    source_resolution.to_csv(
        source_resolution_path, index=False, encoding="utf-8-sig", lineterminator="\n"
    )

    all_jobs = pd.concat([seed_jobs, peak_jobs], ignore_index=True, sort=False)
    expected_job_ids = set(all_jobs["job_id"].astype(str))
    part_paths: list[Path] = []
    for _, source in source_resolution.iterrows():
        source_name = str(source["source_mzml_name"])
        source_hash = str(source["sha256"])
        source_path = Path(str(source["resolved_path"]))
        tolerance = float(source["eic_tolerance_ppm"])
        source_jobs = all_jobs[all_jobs["source_mzml_name"] == source_name].copy()
        part_path = parts_dir / safe_part_name(source_name)
        part_paths.append(part_path)

        reuse = False
        if part_path.exists():
            prior = pd.read_csv(part_path, encoding="utf-8-sig", low_memory=False)
            reuse = (
                len(prior) == len(source_jobs)
                and set(prior["job_id"].astype(str)) == set(source_jobs["job_id"].astype(str))
                and prior["source_mzml_sha256"].astype(str).nunique() == 1
                and str(prior["source_mzml_sha256"].iloc[0]) == source_hash
            )
        if reuse:
            print(f"reuse {source_name}: {len(source_jobs)} jobs")
            continue

        print(f"compute {source_name}: {len(source_jobs)} jobs")
        part = compute_source_part(
            source_jobs,
            source_path=source_path,
            source_hash=source_hash,
            tolerance_ppm=tolerance,
            backend=args.backend,
        )
        part.to_csv(part_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    calculated = pd.concat(
        [pd.read_csv(path, encoding="utf-8-sig", low_memory=False) for path in part_paths],
        ignore_index=True,
    )
    if len(calculated) != len(all_jobs) or set(calculated["job_id"].astype(str)) != expected_job_ids:
        raise RuntimeError("combined result count/job-ID mismatch")
    if calculated["job_id"].duplicated().any():
        raise RuntimeError("duplicate calculated job IDs")

    seed_results = calculated[calculated["entity_type"] == "seed"].copy().sort_values("job_id")
    peak_results = calculated[calculated["entity_type"] == "peak_instance"].copy().sort_values("job_id")
    seed_results_path = output_dir / "old_final_seed_calculated_16.csv"
    peak_results_path = output_dir / "old_final_peak_calculated_16.csv"
    seed_results.to_csv(seed_results_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    peak_results.to_csv(peak_results_path, index=False, encoding="utf-8-sig", lineterminator="\n")

    error_rows = calculated[calculated["error"].fillna("").astype(str) != ""]
    seed_regression = compare_legacy(
        seed_jobs,
        seed_results,
        exact_only=False,
        rtol=args.rtol,
        atol=args.atol,
    )
    peak_exact_regression = compare_legacy(
        peak_jobs,
        peak_results,
        exact_only=True,
        rtol=args.rtol,
        atol=args.atol,
    )

    nonfinite: dict[str, dict[str, int]] = {}
    for label, frame in (("seed", seed_results), ("peak", peak_results)):
        nonfinite[label] = {
            name: int((~np.isfinite(pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float))).sum())
            for name in ATTR16
        }

    status = "ok"
    if len(error_rows) > 0:
        status = "blocked_job_errors"
    elif seed_regression["total_attribute_mismatches"] > 0:
        status = "blocked_seed_13_regression_mismatch"
    elif peak_exact_regression["total_attribute_mismatches"] > 0:
        status = "blocked_peak_13_regression_mismatch"

    qc = {
        "status": status,
        "backend": args.backend,
        "source_roots": [str(path.resolve()) for path in args.source_root],
        "counts": {
            "source_files": int(len(source_resolution)),
            "seed_jobs": int(len(seed_jobs)),
            "seed_results": int(len(seed_results)),
            "peak_jobs": int(len(peak_jobs)),
            "peak_results": int(len(peak_results)),
            "job_errors": int(len(error_rows)),
        },
        "legacy_regression_tolerance": {"rtol": args.rtol, "atol": args.atol},
        "seed_legacy_13_regression": seed_regression,
        "peak_exact_legacy_13_regression": peak_exact_regression,
        "nonfinite_value_counts": nonfinite,
        "outputs": {
            source_resolution_path.name: sha256_file(source_resolution_path),
            seed_results_path.name: sha256_file(seed_results_path),
            peak_results_path.name: sha256_file(peak_results_path),
        },
    }
    qc_path = output_dir / "remote_attribute_qc.json"
    qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checksum_paths = [source_resolution_path, seed_results_path, peak_results_path, qc_path]
    checksums_path = output_dir / "SHA256SUMS.txt"
    checksums_path.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    if status != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
