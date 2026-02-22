from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lipidbench.utils.eic_methods import extract_intensity


@dataclass
class Spectrum1:
    rt_min: float
    mz: np.ndarray
    intensity: np.ndarray


def load_ms1_spectra(mzml_path: str | Path) -> list[Spectrum1]:
    import pymzml  # type: ignore

    reader = pymzml.run.Reader(str(mzml_path))
    out: list[Spectrum1] = []
    for spec in reader:
        ms_level = int(getattr(spec, "ms_level", 1) or 1)
        if ms_level != 1:
            continue

        try:
            rt_min = float(spec.scan_time_in_minutes())
        except Exception:
            continue

        try:
            peaks = spec.peaks("centroided")
        except Exception:
            peaks = spec.peaks("raw")

        if peaks is None or len(peaks) == 0:
            mz = np.asarray([], dtype=np.float64)
            inten = np.asarray([], dtype=np.float64)
        else:
            arr = np.asarray(peaks, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 2:
                mz = np.asarray([], dtype=np.float64)
                inten = np.asarray([], dtype=np.float64)
            else:
                mz = arr[:, 0]
                inten = arr[:, 1]

        out.append(Spectrum1(rt_min=rt_min, mz=mz, intensity=inten))

    return out


def _extract_trace(
    spectra: list[Spectrum1],
    target_mz: float,
    tolerance: float,
    unit: Literal["ppm", "Da"] = "ppm",
    method: Literal["nearest", "window_sum"] = "nearest",
):
    rt = np.asarray([s.rt_min for s in spectra], dtype=np.float64)
    eic = np.zeros_like(rt)
    mass_track = np.full_like(rt, np.nan)

    for i, s in enumerate(spectra):
        if s.mz.size == 0:
            continue

        val = extract_intensity(
            s.mz,
            s.intensity,
            target_mz=float(target_mz),
            tolerance=float(tolerance),
            unit=unit,
            method=method,
        )
        eic[i] = float(val)

        pos = int(np.searchsorted(s.mz, target_mz, side="left"))
        if pos >= s.mz.size:
            near = s.mz.size - 1
        elif pos <= 0:
            near = 0
        else:
            near = pos if abs(float(s.mz[pos]) - target_mz) < abs(target_mz - float(s.mz[pos - 1])) else (pos - 1)
        mass_track[i] = float(s.mz[near])

    return rt, eic, mass_track


def _compute_one_feature_attributes(
    rt: np.ndarray,
    eic: np.ndarray,
    mass_track: np.ndarray,
    target_mz: float,
    target_rt_min: float,
    rt_tol_sec: float,
) -> dict:
    rt_tol_min = float(rt_tol_sec) / 60.0
    in_win = (rt >= (target_rt_min - rt_tol_min)) & (rt <= (target_rt_min + rt_tol_min))
    if not np.any(in_win):
        return {
            "peak_slope": np.nan,
            "peak_sharpness": np.nan,
            "peak_height": np.nan,
            "peak_sn_ratio": np.nan,
            "peak_scan_number": 0,
            "peak_width": np.nan,
            "peak_mass_accuracy_ppm": np.nan,
            "peak_mass_accuracy_da": np.nan,
        }

    eic_win = eic[in_win]
    rt_win = rt[in_win]
    mass_win = mass_track[in_win]

    nonzero = np.sort(eic_win[eic_win > 0])
    if nonzero.size == 0:
        return {
            "peak_slope": 0.0,
            "peak_sharpness": 0.0,
            "peak_height": 0.0,
            "peak_sn_ratio": 0.0,
            "peak_scan_number": 0,
            "peak_width": 0.0,
            "peak_mass_accuracy_ppm": np.nan,
            "peak_mass_accuracy_da": np.nan,
        }

    if nonzero.size > 10:
        x = 10
        while x <= nonzero.size:
            blk = float(np.mean(nonzero[:x]))
            sd = float(np.std(nonzero[:x]))
            thres = blk + 3.0 * sd
            if x < nonzero.size and nonzero[x] >= thres:
                break
            x += 10
        cutoff = float(nonzero[min(x - 1, nonzero.size - 1)])
    else:
        blk = 0.0
        sd = 0.0
        cutoff = float(np.max(nonzero))

    above = np.where((eic_win > cutoff))[0]
    if above.size == 0:
        return {
            "peak_slope": 0.0,
            "peak_sharpness": 0.0,
            "peak_height": float(np.max(eic_win)),
            "peak_sn_ratio": 0.0,
            "peak_scan_number": 0,
            "peak_width": 0.0,
            "peak_mass_accuracy_ppm": np.nan,
            "peak_mass_accuracy_da": np.nan,
        }

    rel_peak = int(above[np.argmax(eic_win[above])])
    peak_height = float(eic_win[rel_peak])

    left = rel_peak
    while left > 0 and eic_win[left - 1] >= cutoff:
        left -= 1
    right = rel_peak
    while right < (len(eic_win) - 1) and eic_win[right + 1] >= cutoff:
        right += 1

    peak_scan_number = int(right - left + 1)
    peak_width = float(rt_win[right] - rt_win[left]) if right > left else 0.0

    if sd > 0:
        peak_sn = float((peak_height - blk) / sd)
    else:
        peak_sn = float(peak_height)

    left_span = rel_peak - left
    right_span = right - rel_peak
    left_slope = (peak_height - float(eic_win[left])) / max(left_span, 1)
    right_slope = (peak_height - float(eic_win[right])) / max(right_span, 1)
    denom = max(abs(left_slope) + abs(right_slope), 1e-12)
    peak_slope = float((left_slope + right_slope) / denom)

    sharp_vals = []
    for idx in range(left, right + 1):
        if idx == rel_peak:
            continue
        d = abs(idx - rel_peak)
        if d == 0:
            continue
        sharp_vals.append(abs(peak_height - float(eic_win[idx])) / (d * np.sqrt(max(peak_height, 1e-12))))
    peak_sharpness = float(np.max(sharp_vals)) if sharp_vals else 0.0

    mz_region = mass_win[left : right + 1]
    mz_region = mz_region[np.isfinite(mz_region)]
    if mz_region.size >= 2 and target_mz > 0:
        mass_acc_da = float(2.0 * np.std(mz_region))
        mass_acc_ppm = float(mass_acc_da / target_mz * 1e6)
    else:
        mass_acc_da = np.nan
        mass_acc_ppm = np.nan

    return {
        "peak_slope": peak_slope,
        "peak_sharpness": peak_sharpness,
        "peak_height": peak_height,
        "peak_sn_ratio": peak_sn,
        "peak_scan_number": peak_scan_number,
        "peak_width": peak_width,
        "peak_mass_accuracy_ppm": mass_acc_ppm,
        "peak_mass_accuracy_da": mass_acc_da,
    }


def compute_peak_attributes(
    features_df: pd.DataFrame,
    mzml_path: str | Path,
    *,
    mz_tolerance: float = 0.01,
    tolerance_unit: Literal["ppm", "Da"] = "Da",
    method: Literal["nearest", "window_sum"] = "nearest",
    rt_tol_sec: float = 30.0,
) -> pd.DataFrame:
    if "mz" not in features_df.columns or "RT" not in features_df.columns:
        raise ValueError("features_df 必须包含 mz 和 RT 列")

    spectra = load_ms1_spectra(mzml_path)
    if not spectra:
        raise ValueError("未从 mzML 读取到 MS1 光谱")

    rows = []
    for _, row in features_df.iterrows():
        mz = pd.to_numeric(row.get("mz"), errors="coerce")
        rt = pd.to_numeric(row.get("RT"), errors="coerce")
        if pd.isna(mz) or pd.isna(rt):
            continue

        rt_arr, eic_arr, mass_arr = _extract_trace(
            spectra,
            float(mz),
            float(mz_tolerance),
            unit=tolerance_unit,
            method=method,
        )
        attrs = _compute_one_feature_attributes(
            rt_arr,
            eic_arr,
            mass_arr,
            target_mz=float(mz),
            target_rt_min=float(rt),
            rt_tol_sec=float(rt_tol_sec),
        )

        out = dict(row)
        out.update(attrs)
        rows.append(out)

    return pd.DataFrame(rows)
