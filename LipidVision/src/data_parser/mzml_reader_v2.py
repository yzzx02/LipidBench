"""mzML 单次遍历多目标 EIC 提取模块（V2）。"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Tuple

import numpy as np

Array1D = np.ndarray
EICData = Tuple[Array1D, Array1D]


@dataclass(frozen=True)
class EICTarget:
    """单个 EIC 提取目标定义。

    Attributes:
        target_id: 目标唯一标识。
        ms_level: 质谱层级，1 或 2。
        target_mz: 目标 m/z。
        mz_tol_ppm: m/z 容差（ppm）。
        rt_start: 保留时间窗口起点（分钟）。
        rt_end: 保留时间窗口终点（分钟）。
    """

    target_id: str
    ms_level: int
    target_mz: float
    mz_tol_ppm: float
    rt_start: float
    rt_end: float


def _get_mzml_module() -> Any:
    """延迟加载 pyteomics.mzml 模块。"""
    try:
        return import_module("pyteomics.mzml")
    except ImportError as exc:
        raise ImportError("未检测到 pyteomics，请先安装依赖：pip install pyteomics") from exc


def _extract_rt_in_minutes(spectrum: dict) -> float:
    """从谱图对象中提取保留时间（分钟）。

    Args:
        spectrum: pyteomics 单个谱图字典。

    Returns:
        保留时间（分钟）；若无法解析返回 np.nan。
    """
    scan_list = spectrum.get("scanList", {})
    scans = scan_list.get("scan", [])
    if not scans:
        return float("nan")

    scan0 = scans[0]
    rt_raw = scan0.get("scan start time")
    if rt_raw is None:
        return float("nan")

    try:
        rt_value = float(rt_raw)
    except (TypeError, ValueError):
        return float("nan")

    unit_info = str(getattr(rt_raw, "unit_info", "")).lower()
    if "second" in unit_info or unit_info in {"s", "sec"}:
        return rt_value / 60.0

    unit_name = str(scan0.get("unitName", "")).lower()
    if "second" in unit_name:
        return rt_value / 60.0

    return rt_value


def _prepare_targets(targets: List[EICTarget]) -> Dict[int, Dict[str, np.ndarray]]:
    """将目标按 ms level 预编译为向量化结构。"""
    grouped: Dict[int, Dict[str, np.ndarray]] = {}
    for level in (1, 2):
        level_targets = [t for t in targets if t.ms_level == level]
        if not level_targets:
            continue

        target_ids = np.asarray([t.target_id for t in level_targets], dtype=object)
        rt_start = np.asarray([t.rt_start for t in level_targets], dtype=np.float64)
        rt_end = np.asarray([t.rt_end for t in level_targets], dtype=np.float64)
        mz_center = np.asarray([t.target_mz for t in level_targets], dtype=np.float64)
        mz_tol = np.asarray([t.mz_tol_ppm for t in level_targets], dtype=np.float64)
        mz_delta = mz_center * mz_tol * 1e-6

        grouped[level] = {
            "target_ids": target_ids,
            "rt_start": rt_start,
            "rt_end": rt_end,
            "mz_min": mz_center - mz_delta,
            "mz_max": mz_center + mz_delta,
        }
    return grouped


def extract_multiple_eic(mzml_path: str, targets: List[EICTarget]) -> Dict[str, EICData]:
    """单次遍历 mzML，批量提取多个目标的 EIC。

    设计要点：
        - 仅打开文件一次，流式遍历；
        - 每个谱图内对“当前 RT 且同层级”的目标做向量化窗口求和；
        - 不将整文件读入内存。

    Args:
        mzml_path: mzML 文件路径。
        targets: 目标列表。

    Returns:
        字典：target_id -> (rt_array, intensity_array)。
    """
    if not targets:
        return {}

    for t in targets:
        if t.ms_level not in (1, 2):
            raise ValueError(f"目标 {t.target_id} 的 ms_level 非法: {t.ms_level}")
        if not (np.isfinite(t.rt_start) and np.isfinite(t.rt_end) and t.rt_start < t.rt_end):
            raise ValueError(f"目标 {t.target_id} 的 RT 窗口非法")

    grouped = _prepare_targets(targets)

    rt_buffers: Dict[str, List[float]] = {t.target_id: [] for t in targets}
    int_buffers: Dict[str, List[float]] = {t.target_id: [] for t in targets}

    mzml_module = _get_mzml_module()

    with mzml_module.MzML(mzml_path, use_index=False, iterative=True) as reader:
        for spectrum in reader:
            ms_level = int(spectrum.get("ms level", 0))
            if ms_level not in grouped:
                continue

            rt_min = _extract_rt_in_minutes(spectrum)
            if not np.isfinite(rt_min):
                continue

            group = grouped[ms_level]
            active_mask = (rt_min >= group["rt_start"]) & (rt_min <= group["rt_end"])
            if not np.any(active_mask):
                continue

            active_idx = np.where(active_mask)[0]
            active_ids = group["target_ids"][active_idx]

            mz_array = np.asarray(spectrum.get("m/z array", []), dtype=np.float64)
            int_array = np.asarray(spectrum.get("intensity array", []), dtype=np.float64)

            if mz_array.size == 0 or int_array.size == 0:
                sums = np.zeros(active_idx.size, dtype=np.float64)
            else:
                mz_min = group["mz_min"][active_idx]
                mz_max = group["mz_max"][active_idx]

                left = np.searchsorted(mz_array, mz_min, side="left")
                right = np.searchsorted(mz_array, mz_max, side="right")

                csum = np.concatenate(([0.0], np.cumsum(int_array, dtype=np.float64)))
                sums = csum[right] - csum[left]

            for i, target_id in enumerate(active_ids):
                rt_buffers[str(target_id)].append(float(rt_min))
                int_buffers[str(target_id)].append(float(sums[i]))

    result: Dict[str, EICData] = {}
    for t in targets:
        rt_array = np.asarray(rt_buffers[t.target_id], dtype=np.float64)
        int_array = np.asarray(int_buffers[t.target_id], dtype=np.float64)
        if rt_array.size > 1:
            order = np.argsort(rt_array)
            rt_array = rt_array[order]
            int_array = int_array[order]
        result[t.target_id] = (rt_array, int_array)

    return result
