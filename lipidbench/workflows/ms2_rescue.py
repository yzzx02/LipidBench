"""MS2-guided recovery of missing MS1 features; no differential analysis."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from lipidbench.utils.eic_methods import extract_intensity
from lipidbench.utils.peak_attributes import PEAK_ATTRIBUTE_COLUMNS, _compute_peak_features, load_ms1_spectra
from lipidbench.utils.plot_eic import plot_eic
from lipidbench.utils.rt_boundary_refiner import refine_peak_boundaries_guarded


class RescuePredictor(Protocol):
    def detect(self, image_path: Path) -> list[dict]:
        """Return image-coordinate boxes with score and (optionally) label."""

    def classify(self, image_path: Path, boxes: list[list[float]], attributes: np.ndarray) -> np.ndarray:
        """Return one true-peak probability per NEW box and attribute row."""


@dataclass(frozen=True)
class RescueOptions:
    detection_threshold: float
    classification_threshold: float
    rt_unit: str = "min"
    window_min: float = 2.0
    mz_tolerance_ppm: float = 10.0
    method: str = "nearest"
    ms2_rt_tolerance_min: float = 0.2
    min_scans: int = 3
    max_extension_scans: int = 1
    duplicate_apex_tolerance_min: float = 0.02

    def __post_init__(self):
        for name in ("detection_threshold", "classification_threshold"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("window_min", "mz_tolerance_ppm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("ms2_rt_tolerance_min", "duplicate_apex_tolerance_min"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rt_unit not in {"min", "sec"} or self.method not in {"nearest", "window_sum"}:
            raise ValueError("rt_unit must be min/sec; method must be nearest/window_sum")
        if not isinstance(self.min_scans, int) or self.min_scans < 3:
            raise ValueError("min_scans must be an integer >= 3")
        if not isinstance(self.max_extension_scans, int) or self.max_extension_scans < 0:
            raise ValueError("max_extension_scans must be a nonnegative integer")


@dataclass
class RescueResult:
    targets: pd.DataFrame
    candidates: pd.DataFrame
    peaks: pd.DataFrame
    summary: dict

    def write(self, output_dir: str | Path) -> None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("targets", "candidates", "peaks"):
            getattr(self, name).to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")
        self.peaks.loc[self.peaks.status.eq("rescued")].to_csv(
            folder / "rescued_peaks.csv", index=False, encoding="utf-8-sig")
        (folder / "summary.json").write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_identification_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("identification table must be CSV, TSV, or XLSX")
    return pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", encoding="utf-8-sig")


def _flag(value, name: str) -> bool:
    if pd.isna(value):
        raise ValueError(f"{name} must explicitly indicate presence/absence")
    text = str(value).strip().casefold()
    if text in {"true", "1", "1.0", "yes", "有", "present"}:
        return True
    if text in {"false", "0", "0.0", "no", "无", "absent"}:
        return False
    raise ValueError(f"invalid {name} value: {value!r}")


def _normalise_table(table: pd.DataFrame, base_dir: Path, options: RescueOptions,
                     column_map: dict[str, str] | None) -> list[dict]:
    mapping = column_map or {}
    missing_mapped = set(mapping.values()) - set(table.columns)
    if missing_mapped:
        raise ValueError(f"mapped columns missing from table: {sorted(missing_mapped)}")
    table = table.rename(columns={actual: canonical for canonical, actual in mapping.items()})
    if table.columns.duplicated().any():
        raise ValueError("column mapping creates duplicate column names")
    required = {"sample_id", "mzml_path", "precursor_mz", "ms2_rt", "has_ms1"}
    if required - set(table.columns):
        raise ValueError(f"identification table missing columns: {sorted(required - set(table.columns))}")
    rows = []
    for index, record in enumerate(table.to_dict("records"), start=1):
        try:
            for name in ("sample_id", "mzml_path"):
                if pd.isna(record[name]) or not str(record[name]).strip():
                    raise ValueError(f"{name} must be nonempty")
            mz = float(record["precursor_mz"])
            if not math.isfinite(mz) or mz <= 0:
                raise ValueError("precursor_mz must be finite and positive")
            has_ms2 = _flag(record.get("has_ms2", True), "has_ms2")
            rt = float(record["ms2_rt"]) if has_ms2 else None
            if rt is not None and (not math.isfinite(rt) or rt < 0):
                raise ValueError("ms2_rt must be finite and nonnegative for an MS2 target")
            path = Path(str(record["mzml_path"]).strip())
            path = (base_dir / path).resolve() if not path.is_absolute() else path.resolve()
            target = record.get("target_id")
            rows.append({**record, "row_id": index,
                         "target_id": str(target) if pd.notna(target) and str(target).strip() else f"target_{index}",
                         "sample_id": str(record["sample_id"]).strip(), "mzml_path": str(path),
                         "precursor_mz": mz, "ms2_rt_min": rt / 60 if rt is not None and options.rt_unit == "sec" else rt,
                         "has_ms1": _flag(record["has_ms1"], "has_ms1"), "has_ms2": has_ms2})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"table row {index}: {exc}") from exc
    sample_paths = {}
    for row in rows:
        if row["has_ms2"]:
            previous = sample_paths.setdefault(row["sample_id"], row["mzml_path"])
            if previous != row["mzml_path"]:
                raise ValueError(f"sample_id {row['sample_id']!r} refers to more than one mzML file")
    return rows


def _trace(spectra, mz, center, options):
    low, high = center - options.window_min / 2, center + options.window_min / 2
    selected = [s for s in spectra if low <= s.rt_min <= high]
    rt = np.asarray([s.rt_min for s in selected], dtype=float)
    intensity = np.asarray([extract_intensity(s.mz, s.intensity, target_mz=mz,
                           tolerance=options.mz_tolerance_ppm, unit="ppm", method=options.method)
                            for s in selected], dtype=float)
    if not np.isfinite(rt).all() or np.any(np.diff(rt) <= 0):
        raise ValueError("MS1 retention times must be finite and strictly increasing")
    if not np.isfinite(intensity).all() or np.any(intensity < 0):
        raise ValueError("MS1 intensities must be finite and nonnegative")
    return rt, intensity


def _process_target(row, spectra, predictor, options, images_dir):
    center = row["ms2_rt_min"]
    rt, intensity = _trace(spectra, row["precursor_mz"], center, options)
    if len(rt) < options.min_scans:
        return "insufficient_ms1_scans", []
    if intensity.max() <= 0:
        return "no_ms1_signal", []
    stem = f"target_{row['row_id']:06d}"
    geometry = plot_eic(rt, intensity, stem, images_dir,
                        xlim=(center - options.window_min / 2, center + options.window_min / 2))
    image_path = images_dir / f"{stem}.png"
    row["image_path"] = str(image_path.resolve())
    geometry_path = images_dir / f"{stem}.geometry.json"
    geometry_path.write_text(json.dumps(asdict(geometry), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    row["geometry_path"] = str(geometry_path.resolve())
    candidates, ready = [], []
    for index, detection in enumerate(predictor.detect(image_path), start=1):
        score = float(detection["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("detector returned an invalid score")
        if int(detection.get("label", 1)) != 1 or score < options.detection_threshold:
            continue
        candidate = {"candidate_id": f"{stem}_candidate_{index}", "row_id": row["row_id"],
                     "target_id": row["target_id"], "sample_id": row["sample_id"],
                     "precursor_mz": row["precursor_mz"], "detection_score": score,
                     "detection_box": json.dumps(list(map(float, detection["box"]))), "status": "boundary_qc_failed"}
        candidates.append(candidate)
        try:
            left, right = geometry.box_to_rt(detection["box"])
        except ValueError:
            candidate["status"] = "outside_plot"
            continue
        in_box = (rt >= left) & (rt <= right)
        if in_box.sum() < options.min_scans or intensity[in_box].max() <= 0:
            candidate["status"] = "insufficient_candidate_signal"
            continue
        if center < left - options.ms2_rt_tolerance_min or center > right + options.ms2_rt_tolerance_min:
            candidate["status"] = "outside_ms2_rt"
            continue
        apex_hint = float(rt[in_box][np.argmax(intensity[in_box])])
        refined = refine_peak_boundaries_guarded(rt, intensity, apex_hint, rtmin_hint=left,
                    rtmax_hint=right, max_extension_scans=options.max_extension_scans,
                    search_half_window_min=max(right - left, float(np.median(np.diff(rt)))))
        candidate.update(rtmin=refined.rtmin, rtmax=refined.rtmax, apex_rt=refined.apex_rt,
                         boundary_status=refined.status, guard_applied=refined.guard_applied)
        if refined.status != "ok":
            continue
        mask = (rt >= refined.rtmin) & (rt <= refined.rtmax)
        if mask.sum() < options.min_scans:
            candidate["status"] = "insufficient_candidate_signal"
            continue
        if center < refined.rtmin - options.ms2_rt_tolerance_min or center > refined.rtmax + options.ms2_rt_tolerance_min:
            candidate["status"] = "outside_ms2_rt"
            continue
        local_rt, local_y = rt[mask], intensity[mask]
        apex = int(np.argmax(local_y))
        candidate.update(apex_rt=float(local_rt[apex]), peak_height=float(local_y[apex]),
                         ms2_apex_distance_min=abs(float(local_rt[apex]) - center),
                         ms2_inside_peak=bool(refined.rtmin <= center <= refined.rtmax), n_scans=int(mask.sum()))
        # Continuous bounds are inserted explicitly; no missing-signal imputation.
        area_rt = np.concatenate(([refined.rtmin], rt[(rt > refined.rtmin) & (rt < refined.rtmax)], [refined.rtmax]))
        candidate["area_intensity_min"] = float(np.trapezoid(np.interp(area_rt, rt, intensity), area_rt))
        attrs = _compute_peak_features(local_rt, local_y, apex)
        candidate.update(attrs)
        candidate["candidate_box"] = json.dumps(geometry.candidate_box(refined.rtmin, refined.rtmax, float(local_y[apex])))
        ready.append(candidate)
    if not candidates:
        return "no_detection", candidates
    if not ready:
        return "no_valid_candidate", candidates
    probabilities = np.asarray(predictor.classify(image_path,
        [json.loads(c["candidate_box"]) for c in ready],
        np.asarray([[c[name] for name in PEAK_ATTRIBUTE_COLUMNS] for c in ready], dtype=float)), dtype=float)
    if probabilities.shape != (len(ready),) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("classifier must return one finite probability in [0, 1] per candidate")
    accepted = []
    for candidate, probability in zip(ready, probabilities, strict=True):
        candidate["classification_probability"] = float(probability)
        candidate["status"] = "accepted" if probability >= options.classification_threshold else "rejected"
        if candidate["status"] == "accepted":
            accepted.append(candidate)
    if not accepted:
        return "rejected", candidates
    chosen = min(accepted, key=lambda c: (not c["ms2_inside_peak"], c["ms2_apex_distance_min"],
                                          -c["classification_probability"], -c["detection_score"]))
    row.update(selected_candidate_id=chosen["candidate_id"], accepted_candidate_count=len(accepted))
    return ("existing_validated" if row["has_ms1"] else "rescued"), candidates


def _deduplicate(rows, candidates, options):
    by_id = {c["candidate_id"]: c for c in candidates}
    clusters = []
    for row in rows:
        if row["status"] not in {"rescued", "existing_validated"}:
            continue
        candidate = by_id[row["selected_candidate_id"]]
        matching = None
        for cluster in clusters:
            anchor = cluster[0]
            other = by_id[anchor["selected_candidate_id"]]
            overlap = min(candidate["rtmax"], other["rtmax"]) - max(candidate["rtmin"], other["rtmin"])
            union = max(candidate["rtmax"], other["rtmax"]) - min(candidate["rtmin"], other["rtmin"])
            if (row["sample_id"] == anchor["sample_id"] and row["mzml_path"] == anchor["mzml_path"]
                and abs(row["precursor_mz"] - anchor["precursor_mz"]) <= min(row["precursor_mz"], anchor["precursor_mz"]) * options.mz_tolerance_ppm * 1e-6
                and abs(candidate["apex_rt"] - other["apex_rt"]) <= options.duplicate_apex_tolerance_min
                and overlap / union >= 0.5):
                matching = cluster
                break
        if matching is None:
            clusters.append([row])
        else:
            matching.append(row)
    peaks = []
    for index, cluster in enumerate(clusters, start=1):
        existing = any(row["has_ms1"] for row in cluster)
        chosen = min(cluster, key=lambda row: (not row["has_ms1"],
                    -by_id[row["selected_candidate_id"]]["classification_probability"], row["row_id"]))
        candidate = by_id[chosen["selected_candidate_id"]]
        peak_id = f"peak_{index:06d}"
        for row in cluster:
            row["peak_id"] = peak_id
            if existing:
                row["status"] = "existing_validated" if row["has_ms1"] else "already_present"
            elif row is not chosen:
                row["status"] = "duplicate_peak"
        peaks.append({**candidate, "peak_id": peak_id, "mzml_path": chosen["mzml_path"],
                      "status": "existing_validated" if existing else "rescued",
                      "target_ids": json.dumps(sorted({row["target_id"] for row in cluster}), ensure_ascii=False),
                      "linked_target_rows": len(cluster)})
    return peaks


def rescue_ms2_peaks(table: pd.DataFrame, predictor: RescuePredictor, output_dir: str | Path,
                     options: RescueOptions, *, base_dir: str | Path = ".",
                     column_map: dict[str, str] | None = None, spectra_loader=load_ms1_spectra,
                     progress=None) -> RescueResult:
    """Recover peaks from identification targets without requiring an MS1 feature table.

    ``has_ms1`` means an existing EXTRACTED feature, not the presence of raw MS1 scans.
    Multiple MS2 targets for one physical peak are counted once. Errors are retained
    per target. The injectable predictor/loader also allow callers to reuse loaded models.
    """
    rows = _normalise_table(table, Path(base_dir).resolve(), options, column_map)
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder / name).exists() for name in ("targets.csv", "candidates.csv", "peaks.csv", "rescued_peaks.csv", "summary.json", "images")):
        raise FileExistsError("use a new output directory; existing rescue results will not be overwritten")
    images = folder / "images"
    all_candidates = []
    by_file = {}
    for row in rows:
        row["status"] = "skipped_no_ms2"
        if row["has_ms2"]:
            by_file.setdefault(row["mzml_path"], []).append(row)
    for filename, file_rows in by_file.items():
        try:
            spectra = spectra_loader(Path(filename))
        except Exception as exc:
            for row in file_rows:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            continue
        for index, row in enumerate(file_rows, start=1):
            try:
                row["status"], candidates = _process_target(row, spectra, predictor, options, images)
                all_candidates.extend(candidates)
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            if progress is not None:
                progress(f"{Path(filename).name}: {index}/{len(file_rows)} {row['status']}")
        del spectra
    peaks = _deduplicate(rows, all_candidates, options)
    summary = {"input_target_rows": len(rows), "eligible_missing_target_rows": sum(r["has_ms2"] and not r["has_ms1"] for r in rows),
               "rescued_target_rows": sum(r["status"] in {"rescued", "duplicate_peak"} for r in rows),
               "unique_rescued_peaks": sum(p["status"] == "rescued" for p in peaks),
               "unique_existing_validated_peaks": sum(p["status"] == "existing_validated" for p in peaks),
               "status_counts": dict(Counter(r["status"] for r in rows)), "options": asdict(options),
               "acceptance_basis": "detector + boundary/signal QC + candidate classifier; MS2 annotation supplied by user",
               "area_unit": "intensity * minute; raw MS1 EIC trapezoidal area", "differential_analysis": False}
    summary["per_sample"] = [{"sample_id": sample,
        "input_target_rows": sum(r["sample_id"] == sample for r in rows),
        "unique_rescued_peaks": sum(p["sample_id"] == sample and p["status"] == "rescued" for p in peaks),
        "unique_existing_validated_peaks": sum(p["sample_id"] == sample and p["status"] == "existing_validated" for p in peaks),
        "status_counts": dict(Counter(r["status"] for r in rows if r["sample_id"] == sample))}
        for sample in dict.fromkeys(r["sample_id"] for r in rows)]
    if hasattr(predictor, "provenance"):
        summary["model"] = predictor.provenance
    result = RescueResult(pd.DataFrame(rows, columns=list(dict.fromkeys([key for r in rows for key in r])) or ["status"]),
                          pd.DataFrame(all_candidates, columns=list(dict.fromkeys([key for c in all_candidates for key in c])) or ["status"]),
                          pd.DataFrame(peaks, columns=list(dict.fromkeys([key for p in peaks for key in p])) or ["status"]), summary)
    result.write(folder)
    return result
