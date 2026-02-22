"""多通道 EIC 对齐与 RGB 张量编码模块。"""

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


Array1D = np.ndarray
Array2D = np.ndarray
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


def build_rgb_tensor(
    rt_range: Tuple[float, float],
    ms1_data: ChannelData,
    ms2_channel_1_data: ChannelData,
    ms2_channel_2_data: ChannelData,
    num_pixels: int = 128,
) -> Tuple[Array1D, Array3D]:
    """将 MS1/MS2 异步散点序列对齐并编码为 RGB 图像张量。

    算法流程：
        1) 构建标准伪时间轴 standard_rt；
        2) 三通道分别线性插值到 standard_rt；
        3) 三通道独立归一化；
        4) 先堆叠为 (Width, 3)，再沿 Y 轴广播为 (Height, Width, 3)。

    Args:
        rt_range: 保留时间窗口（分钟），格式为 (start_min, end_min)。
        ms1_data: MS1 通道数据 (rt_array, intensity_array)。
        ms2_channel_1_data: MS2 通道1数据 (rt_array, intensity_array)。
        ms2_channel_2_data: MS2 通道2数据 (rt_array, intensity_array)。
        num_pixels: 图像宽度/时间采样点数，默认 128。

    Returns:
        (standard_rt, image_tensor)：
            - standard_rt: 标准伪时间轴，一维数组，长度为 num_pixels
            - image_tensor: 三通道图像张量，形状为 (num_pixels, num_pixels, 3)

    Raises:
        ValueError: 当 rt_range 或 num_pixels 非法时抛出。
    """
    rt_start, rt_end = rt_range
    if rt_start >= rt_end:
        raise ValueError("rt_range 必须满足 start_min < end_min。")
    if num_pixels <= 0:
        raise ValueError("num_pixels 必须为正整数。")

    standard_rt = np.linspace(rt_start, rt_end, num_pixels, dtype=np.float64)

    ms1_interp = _safe_linear_interpolate(ms1_data[0], ms1_data[1], standard_rt)
    ms2_c1_interp = _safe_linear_interpolate(
        ms2_channel_1_data[0], ms2_channel_1_data[1], standard_rt
    )
    ms2_c2_interp = _safe_linear_interpolate(
        ms2_channel_2_data[0], ms2_channel_2_data[1], standard_rt
    )

    ms1_norm = _normalize_channel(ms1_interp)
    ms2_c1_norm = _normalize_channel(ms2_c1_interp)
    ms2_c2_norm = _normalize_channel(ms2_c2_interp)

    width_x_channels = np.stack([ms1_norm, ms2_c1_norm, ms2_c2_norm], axis=1)
    image_tensor = np.tile(width_x_channels[np.newaxis, :, :], (num_pixels, 1, 1))

    return standard_rt, image_tensor


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)

    rt_window = (1.0, 9.0)

    ms1_rt = np.sort(rng.uniform(1.0, 9.0, size=75))
    ms1_int = (
        1200.0 * np.exp(-0.5 * ((ms1_rt - 5.0) / 0.55) ** 2)
        + rng.normal(0.0, 25.0, size=ms1_rt.size)
    )
    ms1_int = np.clip(ms1_int, a_min=0.0, a_max=None)

    ms2_c1_rt = np.sort(rng.uniform(1.0, 9.0, size=48))
    ms2_c1_int = (
        700.0 * np.exp(-0.5 * ((ms2_c1_rt - 5.15) / 0.75) ** 2)
        + rng.normal(0.0, 18.0, size=ms2_c1_rt.size)
    )
    ms2_c1_int = np.clip(ms2_c1_int, a_min=0.0, a_max=None)

    ms2_c2_rt = np.sort(rng.uniform(1.0, 9.0, size=41))
    ms2_c2_int = (
        500.0 * np.exp(-0.5 * ((ms2_c2_rt - 4.75) / 0.85) ** 2)
        + rng.normal(0.0, 15.0, size=ms2_c2_rt.size)
    )
    ms2_c2_int = np.clip(ms2_c2_int, a_min=0.0, a_max=None)

    standard_rt_axis, rgb_tensor = build_rgb_tensor(
        rt_range=rt_window,
        ms1_data=(ms1_rt, ms1_int),
        ms2_channel_1_data=(ms2_c1_rt, ms2_c1_int),
        ms2_channel_2_data=(ms2_c2_rt, ms2_c2_int),
        num_pixels=128,
    )

    print("standard_rt shape:", standard_rt_axis.shape)
    print("rgb_tensor shape:", rgb_tensor.shape)
    print("rgb_tensor dtype:", rgb_tensor.dtype)
    print("channel max values:", np.max(rgb_tensor, axis=(0, 1)))
