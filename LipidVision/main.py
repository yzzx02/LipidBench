"""LipidVision-DIA 端到端示例入口。

将 mzML 读取模块与 RGB 张量构建模块串联：
1) 提取 MS1 与两个 MS2 通道的 EIC；
2) 插值对齐并构建 (H, W, C) 图像张量；
3) 输出张量基础信息并可选保存为 .npy。
"""

from pathlib import Path
from typing import Tuple
import argparse

import numpy as np
import matplotlib.pyplot as plt

from src.data_parser.mzml_reader import extract_eic
from src.tensor_builder.rgb_encoder import build_rgb_tensor


def run_pipeline(
    mzml_path: str,
    rt_range: Tuple[float, float],
    ms1_target_mz: float,
    ms2_channel_1_mz: float,
    ms2_channel_2_mz: float,
    mz_tol_ppm: float,
    num_pixels: int,
    output_npy: str,
    output_png: str,
) -> None:
    """执行从 mzML 到 RGB 张量的完整流程。

    Args:
        mzml_path: mzML 文件路径。
        rt_range: 保留时间窗口 (start_min, end_min)。
        ms1_target_mz: MS1 目标 m/z。
        ms2_channel_1_mz: MS2 通道1目标 m/z。
        ms2_channel_2_mz: MS2 通道2目标 m/z。
        mz_tol_ppm: 质量偏差容忍度（ppm）。
        num_pixels: 图像宽高像素（Width=Height）。
        output_npy: 输出张量的 .npy 保存路径；空字符串表示不保存。
        output_png: 输出预览图路径；空字符串表示不保存。
    """
    ms1_data = extract_eic(
        mzml_path=mzml_path,
        target_mz=ms1_target_mz,
        mz_tol_ppm=mz_tol_ppm,
        rt_range=rt_range,
        ms_level=1,
    )

    ms2_channel_1_data = extract_eic(
        mzml_path=mzml_path,
        target_mz=ms2_channel_1_mz,
        mz_tol_ppm=mz_tol_ppm,
        rt_range=rt_range,
        ms_level=2,
    )

    ms2_channel_2_data = extract_eic(
        mzml_path=mzml_path,
        target_mz=ms2_channel_2_mz,
        mz_tol_ppm=mz_tol_ppm,
        rt_range=rt_range,
        ms_level=2,
    )

    standard_rt, image_tensor = build_rgb_tensor(
        rt_range=rt_range,
        ms1_data=ms1_data,
        ms2_channel_1_data=ms2_channel_1_data,
        ms2_channel_2_data=ms2_channel_2_data,
        num_pixels=num_pixels,
    )

    print("标准伪时间轴 shape:", standard_rt.shape)
    print("RGB 张量 shape:", image_tensor.shape)
    print("RGB 张量 dtype:", image_tensor.dtype)
    print("各通道最大值:", np.max(image_tensor, axis=(0, 1)))

    if output_npy:
        out_path = Path(output_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, image_tensor)
        print("张量已保存到:", str(out_path.resolve()))

    if output_png:
        png_path = Path(output_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 8), dpi=120)
        plt.imshow(image_tensor)
        plt.title("LipidVision-DIA RGB Tensor Preview")
        plt.xlabel("Time (standard_rt index)")
        plt.ylabel("Broadcasted Y")
        plt.tight_layout()
        plt.savefig(png_path, dpi=120)
        plt.close()
        print("预览图已保存到:", str(png_path.resolve()))


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="LipidVision-DIA mzML -> RGB Tensor")
    parser.add_argument(
        "--mzml",
        type=str,
        default="hilic_swath_sample.mzML",
        help="输入 mzML 文件路径。",
    )
    parser.add_argument("--rt-start", type=float, default=0.0, help="保留时间窗口起点（分钟）。")
    parser.add_argument("--rt-end", type=float, default=30.0, help="保留时间窗口终点（分钟）。")
    parser.add_argument("--ms1-mz", type=float, default=760.585, help="MS1 目标 m/z。")
    parser.add_argument("--ms2-mz-1", type=float, default=184.073, help="MS2 通道1目标 m/z。")
    parser.add_argument("--ms2-mz-2", type=float, default=104.107, help="MS2 通道2目标 m/z。")
    parser.add_argument("--ppm", type=float, default=10.0, help="m/z 容忍度（ppm）。")
    parser.add_argument("--num-pixels", type=int, default=128, help="输出图像宽高像素。")
    parser.add_argument(
        "--output-npy",
        type=str,
        default="outputs/rgb_tensor.npy",
        help="输出 .npy 路径；如不想保存可传空字符串。",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="outputs/rgb_tensor_preview.png",
        help="输出 PNG 预览图路径；如不想保存可传空字符串。",
    )
    return parser


def main() -> None:
    """程序入口。"""
    parser = _build_arg_parser()
    args = parser.parse_args()

    mzml_file = Path(args.mzml)
    if not mzml_file.exists():
        raise FileNotFoundError(f"未找到 mzML 文件: {mzml_file}")

    run_pipeline(
        mzml_path=str(mzml_file),
        rt_range=(float(args.rt_start), float(args.rt_end)),
        ms1_target_mz=float(args.ms1_mz),
        ms2_channel_1_mz=float(args.ms2_mz_1),
        ms2_channel_2_mz=float(args.ms2_mz_2),
        mz_tol_ppm=float(args.ppm),
        num_pixels=int(args.num_pixels),
        output_npy=str(args.output_npy),
        output_png=str(args.output_png),
    )


if __name__ == "__main__":
    main()
