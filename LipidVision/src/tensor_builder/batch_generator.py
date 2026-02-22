"""两阶段高光谱张量批量生成模块。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_parser.mzml_reader import extract_eic
from src.tensor_builder.tensor_encoder import build_hyperspectral_tensor

Array1D = np.ndarray
Array3D = np.ndarray

REQUIRED_COLUMNS: List[str] = [
    "Lipid_Name",
    "Class",
    "Adduct",
    "MS1_mz",
    "Fragments_mz",
    "RT_Pred",
    "RT_Tol",
]


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """将脂质名称转换为安全文件名。

    Args:
        name: 原始脂质名称。
        max_len: 输出文件名最大长度。

    Returns:
        处理后的安全文件名（不含扩展名）。
    """
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", str(name)).strip()
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.strip("._")
    if not safe:
        safe = "unnamed_lipid"
    return safe[:max_len]


def _validate_target_table(df: pd.DataFrame) -> None:
    """校验目标列表必要列。

    Args:
        df: 目标列表 DataFrame。

    Raises:
        ValueError: 缺少必要列时抛出。
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"target_list.csv 缺少必要列: {missing}")


def _to_float(value: object, default: float = np.nan) -> float:
    """安全转换为浮点数。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_fragments_mz(raw: object) -> List[float]:
    """解析分号分隔的碎片 m/z 字符串。

    Args:
        raw: 原始字段，例如 "184.073; 264.268; 104.107"。

    Returns:
        可用碎片 m/z 浮点列表。
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(";")]
    out: List[float] = []
    for part in parts:
        if not part:
            continue
        value = _to_float(part)
        if np.isfinite(value):
            out.append(float(value))
    return out


def _is_zero_tensor(tensor: Array3D, eps: float = 0.0) -> bool:
    """判断张量是否全零（或近似全零）。

    Args:
        tensor: 输入 3D 张量。
        eps: 数值容忍阈值。

    Returns:
        若全零（或最大值 <= eps）返回 True。
    """
    if tensor.size == 0:
        return True
    return float(np.max(np.abs(tensor))) <= float(eps)


def generate_tensors_from_csv(
    mzml_path: str,
    target_list_csv: str,
    output_dir: str = "data/output_tensors",
    mz_tol_ppm: float = 10.0,
    num_pixels: int = 128,
    summary_json: str = "",
) -> Dict[str, int]:
    """从目标列表批量生成高光谱张量并保存。

    Args:
        mzml_path: mzML 文件路径。
        target_list_csv: 目标脂质列表 CSV 路径。
        output_dir: 输出目录，默认 data/output_tensors。
        mz_tol_ppm: 质量偏差容忍度（ppm）。
        num_pixels: 张量宽高像素。

    Returns:
        统计结果字典：
            {"saved": int, "skipped": int, "gate_dropped": int, "failed": int}。
    """
    mzml_file = Path(mzml_path)
    if not mzml_file.exists():
        raise FileNotFoundError(f"未找到 mzML 文件: {mzml_file}")

    csv_file = Path(target_list_csv)
    if not csv_file.exists():
        raise FileNotFoundError(f"未找到目标列表 CSV: {csv_file}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)
    _validate_target_table(df)

    # 仅保留必要列，避免意外列类型干扰
    table = df[REQUIRED_COLUMNS].copy()

    saved = 0
    skipped = 0
    gate_dropped = 0
    failed = 0
    used_names: Dict[str, int] = {}

    for _, row in table.iterrows():
        lipid_name = str(row["Lipid_Name"])
        ms1_mz = _to_float(row["MS1_mz"])
        fragments_mz = _parse_fragments_mz(row["Fragments_mz"])
        rt_pred = _to_float(row["RT_Pred"])
        rt_tol = _to_float(row["RT_Tol"])

        rt_start = float(rt_pred - rt_tol) if np.isfinite(rt_pred) and np.isfinite(rt_tol) else np.nan
        rt_end = float(rt_pred + rt_tol) if np.isfinite(rt_pred) and np.isfinite(rt_tol) else np.nan

        # 基础合法性检查
        if (
            not np.isfinite(ms1_mz)
            or not fragments_mz
            or not np.isfinite(rt_start)
            or not np.isfinite(rt_end)
            or rt_start >= rt_end
        ):
            failed += 1
            print(f"[Fail] {lipid_name} invalid row values.")
            continue

        rt_range: Tuple[float, float] = (float(rt_start), float(rt_end))

        try:
            # 阶段 A：MS1 极速粗筛
            ms1_data: Tuple[Array1D, Array1D] = extract_eic(
                mzml_path=str(mzml_file),
                target_mz=float(ms1_mz),
                mz_tol_ppm=float(mz_tol_ppm),
                rt_range=rt_range,
                ms_level=1,
            )
            rt_ms1, int_ms1 = ms1_data
            if rt_ms1.size == 0 or int_ms1.size == 0 or float(np.max(int_ms1)) < 500.0:
                gate_dropped += 1
                print(f"[Gate_Drop] {lipid_name} MS1 signal too weak.")
                continue

            # 阶段 B：高光谱精提（多个 MS2 碎片）
            ms2_data_list: List[Tuple[Array1D, Array1D]] = []
            for frag_mz in fragments_mz:
                frag_data = extract_eic(
                    mzml_path=str(mzml_file),
                    target_mz=float(frag_mz),
                    mz_tol_ppm=float(mz_tol_ppm),
                    rt_range=rt_range,
                    ms_level=2,
                )
                ms2_data_list.append(frag_data)

            _, tensor = build_hyperspectral_tensor(
                rt_range=rt_range,
                ms1_data=ms1_data,
                ms2_data_list=ms2_data_list,
                num_pixels=int(num_pixels),
            )

            if _is_zero_tensor(tensor):
                skipped += 1
                print(f"[Skip] {lipid_name} hyperspectral tensor is all-zero.")
                continue

            safe_name = sanitize_filename(lipid_name)
            if safe_name in used_names:
                used_names[safe_name] += 1
                safe_name = f"{safe_name}_{used_names[safe_name]}"
            else:
                used_names[safe_name] = 0

            out_path = out_dir / f"{safe_name}.npy"
            np.save(out_path, tensor)
            saved += 1
            print(f"[Save] {lipid_name} -> {out_path}")

        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"[Fail] {lipid_name} {type(exc).__name__}: {exc}")

    summary = {
        "saved": saved,
        "skipped": skipped,
        "gate_dropped": gate_dropped,
        "failed": failed,
    }
    print(
        "[Done] "
        f"saved={summary['saved']} "
        f"gate_dropped={summary['gate_dropped']} "
        f"skipped={summary['skipped']} "
        f"failed={summary['failed']}"
    )

    if summary_json:
        summary_path = Path(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数。"""
    parser = argparse.ArgumentParser(description="Batch tensor generator for LipidVision-DIA")
    parser.add_argument("--mzml", required=True, type=str, help="mzML 文件路径")
    parser.add_argument("--target-list", required=True, type=str, help="target_list.csv 路径")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output_tensors",
        help="输出目录，默认 data/output_tensors",
    )
    parser.add_argument("--ppm", type=float, default=10.0, help="m/z 容差（ppm）")
    parser.add_argument("--num-pixels", type=int, default=128, help="输出张量宽高像素")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="",
        help="可选：将统计结果写入 JSON 文件。",
    )
    return parser


def main() -> None:
    """命令行入口。"""
    args = _build_parser().parse_args()
    generate_tensors_from_csv(
        mzml_path=str(args.mzml),
        target_list_csv=str(args.target_list),
        output_dir=str(args.output_dir),
        mz_tol_ppm=float(args.ppm),
        num_pixels=int(args.num_pixels),
        summary_json=str(args.summary_json),
    )


if __name__ == "__main__":
    main()
