from __future__ import annotations

from typing import Literal

import numpy as np


def get_closest_index(mzs: np.ndarray, target_mz: float, pos: int) -> int:
    if pos >= mzs.size:
        return mzs.size - 1
    if pos <= 0:
        return 0
    return pos if (mzs[pos] - target_mz) < (target_mz - mzs[pos - 1]) else (pos - 1)


def calc_tolerance(target_mz: float, tolerance: float, unit: Literal["ppm", "Da"]) -> float:
    unit_norm = str(unit).strip().lower()
    if unit_norm == "ppm":
        return float(target_mz) * float(tolerance) * 1e-6
    if unit_norm == "da":
        return float(tolerance)
    raise ValueError("unit 仅支持 'ppm' 或 'Da'")


def nearest_intensity(
    mzs: np.ndarray,
    intensities: np.ndarray,
    target_mz: float,
    tolerance: float,
    unit: Literal["ppm", "Da"] = "ppm",
) -> float:
    if mzs.size == 0:
        return 0.0

    pos = int(np.searchsorted(mzs, target_mz, side="left"))
    closest = get_closest_index(mzs, float(target_mz), pos)
    tol = calc_tolerance(float(target_mz), float(tolerance), unit)
    if abs(float(mzs[closest]) - float(target_mz)) <= tol:
        return float(intensities[closest])
    return 0.0


def window_sum_intensity(
    mzs: np.ndarray,
    intensities: np.ndarray,
    target_mz: float,
    tolerance: float,
    unit: Literal["ppm", "Da"] = "ppm",
) -> float:
    if mzs.size == 0:
        return 0.0

    tol = calc_tolerance(float(target_mz), float(tolerance), unit)
    mz_min = float(target_mz) - tol
    mz_max = float(target_mz) + tol
    left = int(np.searchsorted(mzs, mz_min, side="left"))
    right = int(np.searchsorted(mzs, mz_max, side="right"))
    if right <= left:
        return 0.0
    return float(np.sum(intensities[left:right]))


def extract_intensity(
    mzs: np.ndarray,
    intensities: np.ndarray,
    target_mz: float,
    tolerance: float,
    unit: Literal["ppm", "Da"] = "ppm",
    method: Literal["nearest", "window_sum"] = "nearest",
) -> float:
    if method == "window_sum":
        return window_sum_intensity(mzs, intensities, target_mz, tolerance, unit)
    return nearest_intensity(mzs, intensities, target_mz, tolerance, unit)
