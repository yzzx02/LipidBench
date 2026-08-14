from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal, stats

from lipidbench.utils.eic_methods import extract_intensity


LITERATURE_TOP_COLUMNS = [
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

ADDITIONAL_PEAK_ATTRIBUTE_COLUMNS = [
    "SYM",
    "MOD",
    "EDGE",
]

PEAK_ATTRIBUTE_COLUMNS = [
    *LITERATURE_TOP_COLUMNS,
    *ADDITIONAL_PEAK_ATTRIBUTE_COLUMNS,
]


@dataclass
class Spectrum1:
    rt_min: float
    mz: np.ndarray
    intensity: np.ndarray


def _load_ms1_spectra_pyopenms(mzml_path: str | Path) -> list[Spectrum1]:
    import pyopenms  # type: ignore

    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(str(mzml_path), exp)
    picker = pyopenms.PeakPickerHiRes()
    out: list[Spectrum1] = []
    for spec in exp:
        ms_level = int(spec.getMSLevel())
        if ms_level != 1:
            continue

        try:
            rt_min = float(spec.getRT()) / 60.0
        except Exception:
            continue

        if spec.getType() == pyopenms.SpectrumSettings.SpectrumType.PROFILE:
            picked = pyopenms.MSSpectrum()
            picker.pick(spec, picked)
            mzs, intens = picked.get_peaks()
        else:
            mzs, intens = spec.get_peaks()

        if mzs is None or len(mzs) == 0:
            mz = np.asarray([], dtype=np.float64)
            inten = np.asarray([], dtype=np.float64)
        else:
            mz = np.asarray(mzs, dtype=np.float64)
            inten = np.asarray(intens, dtype=np.float64)

        out.append(Spectrum1(rt_min=rt_min, mz=mz, intensity=inten))

    return out


def _load_ms1_spectra_pymzml(mzml_path: str | Path) -> list[Spectrum1]:
    import pymzml  # type: ignore

    out: list[Spectrum1] = []
    reader = pymzml.run.Reader(str(mzml_path))
    for spec in reader:
        try:
            ms_level = int(getattr(spec, "ms_level", 1) or 1)
        except Exception:
            continue
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
                mz = np.asarray(arr[:, 0], dtype=np.float64)
                inten = np.asarray(arr[:, 1], dtype=np.float64)

        out.append(Spectrum1(rt_min=rt_min, mz=mz, intensity=inten))

    return out


def load_ms1_spectra(
    mzml_path: str | Path,
    backend: Literal["auto", "pyopenms", "pymzml"] = "auto",
) -> list[Spectrum1]:
    loaders = {
        "pyopenms": _load_ms1_spectra_pyopenms,
        "pymzml": _load_ms1_spectra_pymzml,
    }

    if backend == "auto":
        order = ["pyopenms", "pymzml"]
    else:
        order = [str(backend)]

    errors: list[str] = []
    for name in order:
        try:
            return loaders[name](mzml_path)
        except Exception as exc:
            errors.append(f"{name}:{type(exc).__name__}:{exc}")

    raise RuntimeError(f"failed to load mzML {mzml_path}: {' | '.join(errors)}")


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


def _interp_crossing(rt: np.ndarray, y: np.ndarray, i1: int, i2: int, level: float) -> float:
    x1, x2 = float(rt[i1]), float(rt[i2])
    y1, y2 = float(y[i1]), float(y[i2])
    if abs(y2 - y1) < 1e-12:
        return (x1 + x2) / 2.0
    t = (level - y1) / (y2 - y1)
    t = min(max(t, 0.0), 1.0)
    return x1 + t * (x2 - x1)


def _width_at_fraction(rt: np.ndarray, y: np.ndarray, apex_idx: int, frac: float) -> tuple[float, float, float] | None:
    if y.size < 3:
        return None
    apex = float(y[apex_idx])
    if not np.isfinite(apex) or apex <= 0:
        return None
    level = float(frac) * apex

    li = apex_idx
    while li > 0 and y[li] >= level:
        li -= 1
    if li == apex_idx:
        return None
    left_rt = _interp_crossing(rt, y, li, li + 1, level)

    ri = apex_idx
    while ri < (len(y) - 1) and y[ri] >= level:
        ri += 1
    if ri == apex_idx:
        return None
    right_rt = _interp_crossing(rt, y, ri - 1, ri, level)

    width = float(max(right_rt - left_rt, 0.0))
    return left_rt, right_rt, width


def _width_at_fraction_or_edges(rt: np.ndarray, y: np.ndarray, apex_idx: int, frac: float) -> tuple[float, float, float] | None:
    if y.size < 2:
        return None
    apex = float(y[apex_idx])
    if not np.isfinite(apex) or apex <= 0:
        return None
    level = float(frac) * apex

    li = int(apex_idx)
    while li > 0 and y[li] >= level:
        li -= 1
    if li == 0 and y[li] >= level:
        left_rt = float(rt[0])
    elif li < apex_idx:
        left_rt = _interp_crossing(rt, y, li, li + 1, level)
    else:
        return None

    ri = int(apex_idx)
    while ri < (len(y) - 1) and y[ri] >= level:
        ri += 1
    if ri == (len(y) - 1) and y[ri] >= level:
        right_rt = float(rt[-1])
    elif ri > apex_idx:
        right_rt = _interp_crossing(rt, y, ri - 1, ri, level)
    else:
        return None

    width = float(max(right_rt - left_rt, 0.0))
    return left_rt, right_rt, width


def _robust_noise_baseline(y: np.ndarray) -> tuple[float, float]:
    if y.size == 0:
        return 0.0, 0.0
    baseline = float(np.median(y))
    mad = float(np.median(np.abs(y - baseline)))
    sigma = 1.4826 * mad
    return baseline, sigma


def _count_local_maxima(y: np.ndarray, min_height: float) -> int:
    if y.size < 3:
        return 0
    cnt = 0
    for i in range(1, len(y) - 1):
        if y[i] >= min_height and y[i] > y[i - 1] and y[i] >= y[i + 1]:
            cnt += 1
    return int(cnt)


def _moving_average3(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    kernel = np.asarray([1.0, 1.0, 1.0], dtype=np.float64) / 3.0
    return np.convolve(x, kernel, mode="same")


def _safe_fisher_z(r: float) -> float:
    if not np.isfinite(r):
        return np.nan
    rc = float(np.clip(r, -0.999999, 0.999999))
    return float(0.5 * np.log((1.0 + rc) / (1.0 - rc)))


def _corr_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        return np.nan
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _shannon_entropy_from_hist(x: np.ndarray, bins: int = 256) -> float:
    if x.size == 0:
        return np.nan
    xmax = float(np.max(x))
    xmin = float(np.min(x))
    if not np.isfinite(xmax) or not np.isfinite(xmin) or xmax <= xmin:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, range=(xmin, xmax), density=False)
    p = hist.astype(np.float64)
    s = float(np.sum(p))
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _compute_additional_peak_features(eic_win: np.ndarray, apex_idx: int) -> dict[str, float]:
    """Compute the three extended Seed-window attributes without changing A.

    ``apex_idx`` is supplied by the existing extraction/windowing pipeline.  In
    particular, MOD intentionally evaluates the smoothed trace at the original
    apex index instead of finding a new smoothed apex.
    """

    out = {k: np.nan for k in ADDITIONAL_PEAK_ATTRIBUTE_COLUMNS}
    x = np.asarray(eic_win, dtype=np.float64)
    n = int(x.size)
    if n == 0 or apex_idx < 0 or apex_idx >= n:
        return out

    x = np.where(np.isfinite(x), x, 0.0)
    x[x < 0] = 0.0
    xA = float(x[apex_idx])
    eps = 1e-12

    # (14) SYM: compare equally long, apex-normalised left and right profiles.
    k = int(min(apex_idx, n - 1 - apex_idx))
    offsets = np.arange(1, k + 1, dtype=np.int64)
    left = x[apex_idx - offsets] / (xA + eps)
    right = x[apex_idx + offsets] / (xA + eps)
    numerator = float(np.sum(np.abs(left - right)))
    denominator = float(np.sum(left + right) + eps)
    out["SYM"] = float(np.clip(1.0 - numerator / denominator, 0.0, 1.0))

    # (15) MOD: strongest secondary-peak prominence on the same MA(3) trace
    # used by DZZ/PCC, excluding peaks within two scans of the original apex A.
    y = _moving_average3(x)
    peak_indices, properties = signal.find_peaks(y, prominence=0.0)
    if peak_indices.size:
        keep = np.abs(peak_indices.astype(np.int64) - int(apex_idx)) > 2
        prominences = np.asarray(properties.get("prominences", []), dtype=np.float64)
        secondary = prominences[keep]
        secondary = secondary[np.isfinite(secondary)]
        secondary_prominence = float(np.max(secondary)) if secondary.size else 0.0
    else:
        secondary_prominence = 0.0
    mod_denominator = float(y[apex_idx]) + eps
    out["MOD"] = float(np.clip(secondary_prominence / mod_denominator, 0.0, 1.0))

    # (16) EDGE: preserve the raw continuous edge/apex ratio.  Downstream
    # train-only preprocessing performs median imputation and z-scoring.
    edge_count = int(max(3, round(0.10 * n)))
    edge_count = min(edge_count, n)
    left_edge = float(np.median(x[:edge_count]))
    right_edge = float(np.median(x[-edge_count:]))
    out["EDGE"] = float(max(left_edge, right_edge) / (xA + eps))

    return out


def _compute_literature_top_features(rt_win: np.ndarray, eic_win: np.ndarray, apex_idx: int) -> dict[str, float]:
    out = {k: np.nan for k in LITERATURE_TOP_COLUMNS}

    x = np.asarray(eic_win, dtype=np.float64)
    n = int(x.size)
    if n < 3:
        return out

    x = np.where(np.isfinite(x), x, 0.0)
    x[x < 0] = 0.0
    xA = float(x[apex_idx])
    x_mean = float(np.mean(x))
    x_std = float(np.std(x))
    x_min = float(np.min(x))
    eps = 1e-12

    xs = _moving_average3(x)

    # (1) SNR
    if x_std > eps:
        out["SNR"] = float((xA - x_mean) / x_std)
    elif xA > 0:
        out["SNR"] = float(xA)

    # (2) CV
    if abs(x_mean) > eps:
        out["CV"] = float(x_std / x_mean)

    # (5) GS (Fisher-Z transformed correlation)
    if np.isfinite(xA) and xA > eps:
        fwhm_pack = _width_at_fraction(rt_win, x, apex_idx, 0.5)
        if fwhm_pack is None:
            span = float(rt_win.max() - rt_win.min()) if rt_win.size >= 2 else 1.0
            sigma_g = max(span / 6.0, 1e-6)
        else:
            _, _, fwhm = fwhm_pack
            sigma_g = max(float(fwhm) / 2.35482, 1e-6)
        mu = float(rt_win[apex_idx])
        xg = np.exp(-0.5 * ((rt_win - mu) / sigma_g) ** 2)
        r_gs = _corr_or_nan(x, xg)
        out["GS"] = _safe_fisher_z(r_gs)

    # (6) TPAS
    den_sum = float(np.sum(x))
    if den_sum > eps and xA > eps:
        out["TPAS"] = float(np.log10(max(0.5 * n * xA / den_sum, eps)))

    # (7) H2B
    fwhm_pack = _width_at_fraction_or_edges(rt_win, x, apex_idx, 0.5)
    if fwhm_pack is not None and rt_win.size >= 2:
        left_rt, right_rt, _ = fwhm_pack
        base = float(rt_win[-1] - rt_win[0])
        if base > eps:
            out["H2B"] = float((right_rt - left_rt) / base)

    # (8) ZZ
    zz_den = float(n * max((xA - x_min) ** 2, eps))
    zz_num = float(np.sum((2.0 * x[1:-1] - x[:-2] - x[2:]) ** 2))
    out["ZZ"] = float(np.log10(max(zz_num / zz_den, eps)))

    # (9) DZZ
    xsA = float(np.max(xs)) if xs.size else 0.0
    xs_min = float(np.min(xs)) if xs.size else 0.0
    zzs_den = float(n * max((xsA - xs_min) ** 2, eps))
    zzs_num = float(np.sum((2.0 * xs[1:-1] - xs[:-2] - xs[2:]) ** 2))
    zzs = float(np.log10(max(zzs_num / zzs_den, eps)))
    out["DZZ"] = float(out["ZZ"] - zzs) if np.isfinite(out["ZZ"]) else np.nan

    # (10) PCC (Fisher-Z transformed)
    r_pcc = _corr_or_nan(x, xs)
    out["PCC"] = _safe_fisher_z(r_pcc)

    # (12) SKEW on x/xA
    if xA > eps:
        xn = x / xA
        out["SKEW"] = float(stats.skew(xn, bias=False, nan_policy="omit"))

    # (27) DENT on x/sum(x)
    if den_sum > eps:
        p = x / den_sum
        p = p[p > 0]
        if p.size > 0:
            out["DENT"] = float(-np.sum(p * np.log2(p)))

    # (21) DM mean(abs(diff(x))) / mean(x)
    if n >= 2 and abs(x_mean) > eps:
        out["DM"] = float(np.mean(np.abs(np.diff(x))) / x_mean)

    # (23) ENT: entropy(x) 采用强度分布直方图熵近似 MATLAB entropy 行为
    out["ENT"] = _shannon_entropy_from_hist(x, bins=256)

    # (24) JAG
    if n >= 3:
        jag_num = float(np.sum(np.abs(np.diff(np.sign(np.diff(x))))))
        out["JAG"] = float(jag_num / max(n, 1))

    return out


def _compute_one_feature_attributes(
    rt: np.ndarray,
    eic: np.ndarray,
    mass_track: np.ndarray,
    target_mz: float,
    target_rt_min: float,
    target_rtmin: float | None,
    target_rtmax: float | None,
    rt_tol_sec: float,
    include_literature_top: bool = False,
) -> dict:
    if target_rtmin is not None and target_rtmax is not None and np.isfinite(target_rtmin) and np.isfinite(target_rtmax):
        lo = float(min(target_rtmin, target_rtmax))
        hi = float(max(target_rtmin, target_rtmax))
        in_win = (rt >= lo) & (rt <= hi)
    else:
        rt_tol_min = float(rt_tol_sec) / 60.0
        in_win = (rt >= (target_rt_min - rt_tol_min)) & (rt <= (target_rt_min + rt_tol_min))

    out = {c: np.nan for c in PEAK_ATTRIBUTE_COLUMNS}

    if not np.any(in_win):
        return out

    eic_win = eic[in_win]
    rt_win = rt[in_win]

    if eic_win.size == 0:
        return out

    eic_win = np.asarray(eic_win, dtype=np.float64)
    eic_win = np.where(np.isfinite(eic_win), eic_win, 0.0)
    eic_win[eic_win < 0] = 0.0

    apex_idx = int(np.argmax(eic_win))
    apex_int = float(eic_win[apex_idx])
    if not np.isfinite(apex_int) or apex_int <= 0:
        return out

    # Delegate all calculation to the literature top features method which is verified perfectly correct
    calc_out = _compute_literature_top_features(rt_win=rt_win, eic_win=eic_win, apex_idx=apex_idx)
    out.update(calc_out)
    out.update(_compute_additional_peak_features(eic_win=eic_win, apex_idx=apex_idx))

    return out


def compute_peak_attributes(
    features_df: pd.DataFrame,
    mzml_path: str | Path,
    *,
    mz_tolerance: float = 0.01,
    tolerance_unit: Literal["ppm", "Da"] = "Da",
    method: Literal["nearest", "window_sum"] = "nearest",
    rt_tol_sec: float = 30.0,
    include_literature_top: bool = False,
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
        rtmin = pd.to_numeric(row.get("RTmin"), errors="coerce") if "RTmin" in row else np.nan
        rtmax = pd.to_numeric(row.get("RTmax"), errors="coerce") if "RTmax" in row else np.nan

        attrs = _compute_one_feature_attributes(
            rt_arr,
            eic_arr,
            mass_arr,
            target_mz=float(mz),
            target_rt_min=float(rt),
            target_rtmin=(None if pd.isna(rtmin) else float(rtmin)),
            target_rtmax=(None if pd.isna(rtmax) else float(rtmax)),
            rt_tol_sec=float(rt_tol_sec),
            include_literature_top=bool(include_literature_top),
        )

        out = dict(row)
        out.update(attrs)
        rows.append(out)

    return pd.DataFrame(rows)
