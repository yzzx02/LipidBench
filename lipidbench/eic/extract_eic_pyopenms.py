from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class EICTrace:
    rt_min: np.ndarray
    intensity: np.ndarray


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

    rt_list: list[float] = []
    int_list: list[float] = []

    for spec in exp:
        if int(spec.getMSLevel()) != ms_level:
            continue

        rt_min = float(spec.getRT()) / 60.0
        if rt_min_limit is not None and rt_min < rt_min_limit:
            continue
        if rt_max_limit is not None and rt_min > rt_max_limit:
            continue

        mzs, intensities = spec.get_peaks()
        mzs_arr = np.asarray(mzs, dtype=np.float64)
        ints_arr = np.asarray(intensities, dtype=np.float64)

        rt_list.append(rt_min)
        int_list.append(_nearest_intensity(mzs_arr, ints_arr, float(target_mz), float(ppm)))

    return EICTrace(rt_min=np.asarray(rt_list, dtype=np.float64), intensity=np.asarray(int_list, dtype=np.float64))
