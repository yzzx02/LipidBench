from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np


@dataclass(frozen=True)
class EICTrace:
    rt_min: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class MS1Cache:
    rt_min: np.ndarray
    mz_arrays: tuple[np.ndarray, ...]
    int_arrays: tuple[np.ndarray, ...]


def _nearest_intensity(mzs: np.ndarray, intensities: np.ndarray, target_mz: float, ppm: float) -> float:
    if mzs.size == 0:
        return 0.0

    idx = int(np.searchsorted(mzs, target_mz, side="left"))
    if idx <= 0:
        nearest_idx = 0
    elif idx >= mzs.size:
        nearest_idx = mzs.size - 1
    else:
        left = idx - 1
        right = idx
        nearest_idx = left if abs(mzs[left] - target_mz) <= abs(mzs[right] - target_mz) else right

    mz_nearest = float(mzs[nearest_idx])
    tol = target_mz * ppm * 1e-6
    if abs(mz_nearest - target_mz) <= tol:
        return float(intensities[nearest_idx])
    return 0.0


def _window_sum_intensity(mzs: np.ndarray, intensities: np.ndarray, target_mz: float, ppm: float) -> float:
    if mzs.size == 0:
        return 0.0

    tol = target_mz * ppm * 1e-6
    mz_min = target_mz - tol
    mz_max = target_mz + tol
    left = int(np.searchsorted(mzs, mz_min, side="left"))
    right = int(np.searchsorted(mzs, mz_max, side="right"))
    if right <= left:
        return 0.0
    return float(np.sum(intensities[left:right]))


def build_ms1_cache(exp, *, ms_level: int = 1) -> MS1Cache:
    """Build a reusable MS1 cache from a pyopenms experiment.

    Use this when extracting many EIC traces from the same mzML to avoid repeated
    per-spectrum Python object conversion overhead.
    """

    rt_list: list[float] = []
    mz_list: list[np.ndarray] = []
    int_list: list[np.ndarray] = []

    for spec in exp:
        if int(spec.getMSLevel()) != ms_level:
            continue
        rt_list.append(float(spec.getRT()) / 60.0)
        mzs, intensities = spec.get_peaks()
        mz_list.append(np.asarray(mzs, dtype=np.float64))
        int_list.append(np.asarray(intensities, dtype=np.float64))

    return MS1Cache(
        rt_min=np.asarray(rt_list, dtype=np.float64),
        mz_arrays=tuple(mz_list),
        int_arrays=tuple(int_list),
    )


def extract_eic_from_cache(
    cache: MS1Cache,
    *,
    target_mz: float,
    ppm: float,
    rt_min_limit: Optional[float] = None,
    rt_max_limit: Optional[float] = None,
    method: Literal["nearest", "window_sum"] = "nearest",
) -> EICTrace:
    """Extract EIC from prebuilt cache.

    method:
      - nearest: nearest point in ppm gate (current LipidBench behavior)
      - window_sum: sum intensity in ppm window (often better for DL peak-shape robustness)
    """

    if cache.rt_min.size == 0:
        return EICTrace(rt_min=np.asarray([], dtype=np.float64), intensity=np.asarray([], dtype=np.float64))

    mask = np.ones(cache.rt_min.shape, dtype=bool)
    if rt_min_limit is not None:
        mask &= cache.rt_min >= float(rt_min_limit)
    if rt_max_limit is not None:
        mask &= cache.rt_min <= float(rt_max_limit)

    rts = cache.rt_min[mask]
    idxs = np.flatnonzero(mask)
    ints = np.zeros(rts.shape, dtype=np.float64)

    use_window_sum = method == "window_sum"
    for out_i, src_i in enumerate(idxs):
        mzs = cache.mz_arrays[int(src_i)]
        spectrum_int = cache.int_arrays[int(src_i)]
        if use_window_sum:
            ints[out_i] = _window_sum_intensity(mzs, spectrum_int, float(target_mz), float(ppm))
        else:
            ints[out_i] = _nearest_intensity(mzs, spectrum_int, float(target_mz), float(ppm))

    return EICTrace(rt_min=rts, intensity=ints)


def extract_eic_nearest_ppm(
    exp,
    *,
    target_mz: float,
    ppm: float,
    rt_min_limit: Optional[float] = None,
    rt_max_limit: Optional[float] = None,
    ms_level: int = 1,
) -> EICTrace:
    """Extract EIC using nearest-point + ppm gate per spectrum.

    RT in returned trace is in minutes.
    """

    cache = build_ms1_cache(exp, ms_level=ms_level)
    return extract_eic_from_cache(
        cache,
        target_mz=target_mz,
        ppm=ppm,
        rt_min_limit=rt_min_limit,
        rt_max_limit=rt_max_limit,
        method="nearest",
    )
