"""mzML 解析与 EIC 提取模块。"""

from importlib import import_module
from typing import Any, Optional, Tuple

import numpy as np


Array1D = np.ndarray


def _extract_rt_in_minutes(spectrum: dict) -> Optional[float]:
    """从谱图对象中提取保留时间（单位：分钟）。

    Args:
        spectrum: pyteomics 读取到的单个谱图字典。

    Returns:
        保留时间（分钟）。若缺失或无法解析则返回 None。
    """
    scan_list = spectrum.get("scanList", {})
    scans = scan_list.get("scan", [])
    if not scans:
        return None

    scan0 = scans[0]
    rt_raw = scan0.get("scan start time")
    if rt_raw is None:
        return None

    try:
        rt_value = float(rt_raw)
    except (TypeError, ValueError):
        return None

    unit_info = str(getattr(rt_raw, "unit_info", "")).lower()
    if "second" in unit_info or unit_info in {"s", "sec"}:
        return rt_value / 60.0

    unit_name = str(scan0.get("unitName", "")).lower()
    if "second" in unit_name:
        return rt_value / 60.0

    return rt_value


def _get_mzml_module() -> Any:
    """延迟加载 pyteomics.mzml 模块。

    Returns:
        pyteomics.mzml 模块对象。

    Raises:
        ImportError: 当环境中未安装 pyteomics 时抛出并附带清晰提示。
    """
    try:
        return import_module("pyteomics.mzml")
    except ImportError as exc:
        raise ImportError(
            "未检测到 pyteomics，请先安装依赖：pip install pyteomics"
        ) from exc


def extract_eic(
    mzml_path: str,
    target_mz: float,
    mz_tol_ppm: float,
    rt_range: Tuple[float, float],
    ms_level: int,
) -> Tuple[Array1D, Array1D]:
    """从 mzML 文件提取目标离子的 EIC 序列。

    说明：
        1) 依据 ppm 动态计算 m/z 窗口；
        2) 依据 ms level 与保留时间窗口过滤谱图；
        3) 在窗口内以 NumPy 向量化方式对强度求和。

    Args:
        mzml_path: mzML 文件路径。
        target_mz: 目标 m/z。
        mz_tol_ppm: m/z 容差（ppm）。
        rt_range: 保留时间窗口（分钟），格式为 (start_min, end_min)。
        ms_level: 质谱层级（1 或 2）。

    Returns:
        (rt_array, intensity_array)：
            - rt_array: 一维保留时间数组（分钟）
            - intensity_array: 与 rt 对齐的一维 EIC 强度数组
        若窗口内无匹配谱图，返回两个空数组。

    Raises:
        ValueError: 当 ms_level 非 1 或 2，或 rt_range 非法时抛出。
    """
    if ms_level not in (1, 2):
        raise ValueError("ms_level 必须为 1 或 2。")

    rt_start, rt_end = rt_range
    if rt_start >= rt_end:
        raise ValueError("rt_range 必须满足 start_min < end_min。")

    mz_delta = target_mz * mz_tol_ppm * 1e-6
    mz_min = target_mz - mz_delta
    mz_max = target_mz + mz_delta

    rt_values = []
    intensity_values = []

    mzml_module = _get_mzml_module()

    with mzml_module.MzML(mzml_path, use_index=False, iterative=True) as reader:
        for spectrum in reader:
            rt_min = _extract_rt_in_minutes(spectrum)
            if rt_min is None:
                continue
            if rt_min > rt_end:
                # 大多数 LC-MS 文件按 RT 递增存储，可提前结束提升性能。
                break
            if rt_min < rt_start:
                continue

            if int(spectrum.get("ms level", 0)) != ms_level:
                continue

            mz_array = np.asarray(spectrum.get("m/z array", []), dtype=np.float64)
            int_array = np.asarray(spectrum.get("intensity array", []), dtype=np.float64)
            if mz_array.size == 0 or int_array.size == 0:
                continue

            window_mask = (mz_array >= mz_min) & (mz_array <= mz_max)
            eic_intensity = float(np.sum(int_array[window_mask], dtype=np.float64))

            rt_values.append(rt_min)
            intensity_values.append(eic_intensity)

    if not rt_values:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )

    rt_array = np.asarray(rt_values, dtype=np.float64)
    intensity_array = np.asarray(intensity_values, dtype=np.float64)

    order = np.argsort(rt_array)
    return rt_array[order], intensity_array[order]
