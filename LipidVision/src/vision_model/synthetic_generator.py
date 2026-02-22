"""合成高光谱样本生成器（用于分割预训练）。"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _gaussian_curve(x: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    """生成一维高斯曲线。"""
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def _build_one_sample(
    width: int = 128,
    height: int = 128,
    channels: int = 5,
    noise_std: float = 0.01,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成单个 (Tensor_X, Mask_Y)。

    约定：
        - Tensor_X 形状: (C, H, W)
        - Mask_Y 形状: (1, H, W)
        - 通道定义（默认5通道）：
            ch0: MS1（目标 + 干扰共流出）
            ch1/ch2: 目标物质 MS2 特征
            ch3/ch4: 干扰物质 MS2 特征
    """
    if channels < 5:
        raise ValueError("channels 至少为 5，以容纳目标与干扰的 MS2 特征")

    rng = np.random.default_rng(seed)
    x = np.arange(width, dtype=np.float32)

    # 目标峰参数
    target_center = float(rng.uniform(46.0, 84.0))
    target_sigma = float(rng.uniform(2.5, 5.5))
    target_amp_ms1 = float(rng.uniform(0.8, 1.2))

    # 干扰峰参数（故意靠近甚至重叠）
    offset = float(rng.uniform(-6.0, 6.0))
    if abs(offset) < 1.2:
        offset = 1.2 if offset >= 0 else -1.2
    inter_center = float(np.clip(target_center + offset, 8.0, width - 8.0))
    inter_sigma = float(rng.uniform(2.5, 5.5))
    inter_amp_ms1 = float(rng.uniform(0.7, 1.1))

    target_ms1 = _gaussian_curve(x, target_center, target_sigma, target_amp_ms1)
    inter_ms1 = _gaussian_curve(x, inter_center, inter_sigma, inter_amp_ms1)

    # 目标 MS2
    target_ms2_a = _gaussian_curve(x, target_center, target_sigma * 0.95, float(rng.uniform(0.6, 1.0)))
    target_ms2_b = _gaussian_curve(x, target_center, target_sigma * 1.05, float(rng.uniform(0.5, 0.9)))

    # 干扰 MS2（不同特征）
    inter_ms2_a = _gaussian_curve(x, inter_center, inter_sigma * 0.95, float(rng.uniform(0.55, 0.95)))
    inter_ms2_b = _gaussian_curve(x, inter_center, inter_sigma * 1.05, float(rng.uniform(0.45, 0.85)))

    lines = np.zeros((channels, width), dtype=np.float32)
    lines[0, :] = target_ms1 + inter_ms1
    lines[1, :] = target_ms2_a
    lines[2, :] = target_ms2_b
    lines[3, :] = inter_ms2_a
    lines[4, :] = inter_ms2_b

    if channels > 5:
        for ch in range(5, channels):
            aux_center = float(rng.uniform(10.0, width - 10.0))
            aux_sigma = float(rng.uniform(2.0, 7.0))
            aux_amp = float(rng.uniform(0.1, 0.4))
            lines[ch, :] = _gaussian_curve(x, aux_center, aux_sigma, aux_amp)

    # 加噪声并裁剪
    noise = rng.normal(loc=0.0, scale=noise_std, size=lines.shape).astype(np.float32)
    lines = np.clip(lines + noise, a_min=0.0, a_max=None)

    # 独立通道归一化
    max_per_ch = np.max(lines, axis=1, keepdims=True)
    max_per_ch[max_per_ch <= 0] = 1.0
    lines = lines / max_per_ch

    # 沿 Y 轴广播到 2D
    tensor_x = np.tile(lines[:, np.newaxis, :], (1, height, 1)).astype(np.float32)

    # 目标 mask：仅标注目标峰区域
    threshold = float(np.exp(-0.5 * (2.0**2)))  # ~0.1353 (约 ±2σ)
    target_region = (target_ms1 / max(target_amp_ms1, 1e-6)) >= threshold
    mask_line = target_region.astype(np.float32)
    mask_y = np.tile(mask_line[np.newaxis, np.newaxis, :], (1, height, 1)).astype(np.float32)

    return tensor_x, mask_y


def generate_synthetic_dataset(
    output_dir: str,
    n_samples: int = 10,
    width: int = 128,
    height: int = 128,
    channels: int = 5,
    noise_std: float = 0.01,
    seed: int = 42,
) -> None:
    """批量生成合成训练数据对并保存为 `.npy`。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for i in range(n_samples):
        tensor_x, mask_y = _build_one_sample(
            width=width,
            height=height,
            channels=channels,
            noise_std=noise_std,
            seed=seed + i,
        )
        np.save(out / f"image_{i:04d}.npy", tensor_x)
        np.save(out / f"mask_{i:04d}.npy", mask_y)

    print(f"[Done] generated {n_samples} pairs in {out}")


if __name__ == "__main__":
    generate_synthetic_dataset(
        output_dir="data/synthetic_train",
        n_samples=10,
        width=128,
        height=128,
        channels=5,
        noise_std=0.01,
        seed=2026,
    )
