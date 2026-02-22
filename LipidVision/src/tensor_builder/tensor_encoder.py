"""多通道高光谱张量编码模块。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d

Array1D = np.ndarray
Array3D = np.ndarray
ChannelData = Tuple[Array1D, Array1D]


def _safe_linear_interpolate(
    rt_array: Array1D,
    intensity_array: Array1D,
    standard_rt: Array1D,
) -> Array1D:
    """安全执行线性插值，异常情况下返回全零序列。

    Args:
        rt_array: 原始保留时间一维数组。
        intensity_array: 原始强度一维数组。
        standard_rt: 标准伪时间轴。

    Returns:
        插值后的强度数组，长度与 standard_rt 一致。
    """
    if rt_array.size == 0 or intensity_array.size == 0:
        return np.zeros_like(standard_rt, dtype=np.float64)

    rt = np.asarray(rt_array, dtype=np.float64)
    intensity = np.asarray(intensity_array, dtype=np.float64)

    finite_mask = np.isfinite(rt) & np.isfinite(intensity)
    rt = rt[finite_mask]
    intensity = intensity[finite_mask]
    if rt.size == 0:
        return np.zeros_like(standard_rt, dtype=np.float64)

    order = np.argsort(rt)
    rt = rt[order]
    intensity = intensity[order]

    unique_rt, inverse_idx = np.unique(rt, return_inverse=True)
    unique_intensity = np.zeros_like(unique_rt, dtype=np.float64)
    np.add.at(unique_intensity, inverse_idx, intensity)

    if unique_rt.size == 1:
        out = np.zeros_like(standard_rt, dtype=np.float64)
        nearest_idx = int(np.argmin(np.abs(standard_rt - unique_rt[0])))
        out[nearest_idx] = unique_intensity[0]
        return out

    interpolator = interp1d(
        unique_rt,
        unique_intensity,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    return np.asarray(interpolator(standard_rt), dtype=np.float64)


def _normalize_channel(channel: Array1D) -> Array1D:
    """对单通道执行独立 Min-Max 归一化（最小值固定为 0）。

    Args:
        channel: 插值后的通道一维数组。

    Returns:
        归一化后的通道数组。若最大值为 0，则返回全零数组。
    """
    max_val = float(np.max(channel)) if channel.size > 0 else 0.0
    if max_val <= 0.0:
        return np.zeros_like(channel, dtype=np.float64)
    return channel / max_val


def build_hyperspectral_tensor(
    rt_range: Tuple[float, float],
    ms1_data: ChannelData,
    ms2_data_list: List[ChannelData],
    num_pixels: int = 128,
) -> Tuple[Array1D, Array3D]:
    """将 MS1 与多个 MS2 碎片序列对齐并编码为高光谱张量。

    算法流程：
        1) 构建标准伪时间轴 `standard_rt`；
        2) 各通道分别线性插值到 `standard_rt`；
        3) 各通道独立归一化；
        4) 通道堆叠后沿 Y 轴广播，得到 (H, W, C) 张量。

    Args:
        rt_range: 保留时间窗口（分钟），格式为 (start_min, end_min)。
        ms1_data: MS1 通道数据 (rt_array, intensity_array)。
        ms2_data_list: MS2 碎片通道列表，每项为 (rt_array, intensity_array)。
        num_pixels: 图像宽高像素，默认 128。

    Returns:
        (standard_rt, tensor)：
            - standard_rt: 标准伪时间轴，一维数组，长度为 num_pixels
            - tensor: 高光谱张量，形状为 (num_pixels, num_pixels, 1 + N)

    Raises:
        ValueError: 当 rt_range 或 num_pixels 非法时抛出。
    """
    rt_start, rt_end = rt_range
    if rt_start >= rt_end:
        raise ValueError("rt_range 必须满足 start_min < end_min。")
    if num_pixels <= 0:
        raise ValueError("num_pixels 必须为正整数。")

    standard_rt = np.linspace(rt_start, rt_end, num_pixels, dtype=np.float64)

    channels_interp: List[Array1D] = [
        _safe_linear_interpolate(ms1_data[0], ms1_data[1], standard_rt)
    ]
    for ms2_data in ms2_data_list:
        channels_interp.append(
            _safe_linear_interpolate(ms2_data[0], ms2_data[1], standard_rt)
        )

    channels_norm: List[Array1D] = [_normalize_channel(ch) for ch in channels_interp]

    width_x_channels = np.stack(channels_norm, axis=1)
    tensor = np.tile(width_x_channels[np.newaxis, :, :], (num_pixels, 1, 1))

    return standard_rt, tensor
