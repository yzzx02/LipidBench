"""两阶段 + 单次遍历缓存 + 并行张量构建批量引擎（V2）。"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data_parser.mzml_reader_v2 import EICTarget, extract_multiple_eic
from src.tensor_builder.tensor_encoder import build_hyperspectral_tensor

Array1D = np.ndarray
EICData = Tuple[Array1D, Array1D]

REQUIRED_COLUMNS: List[str] = [
    "Lipid_Name",
    "Class",
    "Adduct",
    "MS1_mz",
    "Fragments_mz",
    "RT_Pred",
    "RT_Tol",
]


@dataclass(frozen=True)
class LipidJob:
    """单个脂质任务信息。"""

    row_idx: int
    lipid_name: str
    lipid_class: str
    adduct: str
    ms1_mz: float
    fragments_mz: Tuple[float, ...]
    rt_range: Tuple[float, float]


def sanitize_filename(name: str, max_len: int = 120) -> str:
    """将脂质名称转换为安全文件名。"""
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", str(name)).strip()
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.strip("._")
    if not safe:
        safe = "unnamed_lipid"
    return safe[:max_len]


def _to_float(value: object, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_fragments_mz(raw: object) -> List[float]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    out: List[float] = []
    for part in text.split(";"):
        v = _to_float(part.strip())
        if np.isfinite(v):
            out.append(float(v))
    return out


def _validate_target_table(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"target_list_v2.csv 缺少必要列: {missing}")


def _build_jobs(df: pd.DataFrame) -> Tuple[List[LipidJob], List[str]]:
    jobs: List[LipidJob] = []
    errors: List[str] = []

    for row_idx, row in df.iterrows():
        lipid_name = str(row["Lipid_Name"])
        lipid_class = str(row["Class"])
        adduct = str(row["Adduct"])
        ms1_mz = _to_float(row["MS1_mz"])
        fragments = _parse_fragments_mz(row["Fragments_mz"])
        rt_pred = _to_float(row["RT_Pred"])
        rt_tol = _to_float(row["RT_Tol"])

        rt_start = rt_pred - rt_tol if np.isfinite(rt_pred) and np.isfinite(rt_tol) else np.nan
        rt_end = rt_pred + rt_tol if np.isfinite(rt_pred) and np.isfinite(rt_tol) else np.nan

        if (
            not np.isfinite(ms1_mz)
            or len(fragments) == 0
            or not np.isfinite(rt_start)
            or not np.isfinite(rt_end)
            or rt_start >= rt_end
        ):
            errors.append(f"[Fail] {lipid_name} invalid row values.")
            continue

        jobs.append(
            LipidJob(
                row_idx=int(row_idx),
                lipid_name=lipid_name,
                lipid_class=lipid_class,
                adduct=adduct,
                ms1_mz=float(ms1_mz),
                fragments_mz=tuple(float(x) for x in fragments),
                rt_range=(float(rt_start), float(rt_end)),
            )
        )

    return jobs, errors


def _target_id_ms1(job: LipidJob) -> str:
    return f"R{job.row_idx}::MS1"


def _target_id_frag(job: LipidJob, frag_idx: int) -> str:
    return f"R{job.row_idx}::F{frag_idx}"


def _build_eic_targets(jobs: List[LipidJob], mz_tol_ppm: float) -> List[EICTarget]:
    targets: List[EICTarget] = []
    for job in jobs:
        rt_start, rt_end = job.rt_range
        targets.append(
            EICTarget(
                target_id=_target_id_ms1(job),
                ms_level=1,
                target_mz=job.ms1_mz,
                mz_tol_ppm=float(mz_tol_ppm),
                rt_start=rt_start,
                rt_end=rt_end,
            )
        )
        for i, frag_mz in enumerate(job.fragments_mz):
            targets.append(
                EICTarget(
                    target_id=_target_id_frag(job, i),
                    ms_level=2,
                    target_mz=float(frag_mz),
                    mz_tol_ppm=float(mz_tol_ppm),
                    rt_start=rt_start,
                    rt_end=rt_end,
                )
            )
    return targets


def _is_zero_tensor(tensor: np.ndarray, eps: float = 0.0) -> bool:
    if tensor.size == 0:
        return True
    return float(np.max(np.abs(tensor))) <= float(eps)


def _encode_and_save_one(
    lipid_name: str,
    save_name: str,
    rt_range: Tuple[float, float],
    ms1_data: EICData,
    ms2_data_list: List[EICData],
    num_pixels: int,
    output_dir: str,
) -> Tuple[str, str]:
    """子进程任务：构建并保存高光谱张量。"""
    try:
        _, tensor = build_hyperspectral_tensor(
            rt_range=rt_range,
            ms1_data=ms1_data,
            ms2_data_list=ms2_data_list,
            num_pixels=int(num_pixels),
        )
        if _is_zero_tensor(tensor):
            return "skipped", f"[Skip] {lipid_name} hyperspectral tensor is all-zero."

        out_path = Path(output_dir) / f"{save_name}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, tensor)
        return "saved", f"[Save] {lipid_name} -> {out_path}"
    except Exception as exc:  # pragma: no cover
        return "failed", f"[Fail] {lipid_name} {type(exc).__name__}: {exc}"


def generate_tensors_from_csv_v2(
    mzml_path: str,
    target_list_csv: str,
    output_dir: str = "data/output_tensors_v2",
    mz_tol_ppm: float = 10.0,
    num_pixels: int = 128,
    ms1_gate_threshold: float = 500.0,
    max_workers: int = 0,
    summary_json: str = "",
) -> Dict[str, int]:
    """V2 批量引擎：单次遍历缓存 EIC + 并行构建张量。

    Args:
        mzml_path: mzML 文件路径。
        target_list_csv: target_list_v2.csv 路径。
        output_dir: 输出目录。
        mz_tol_ppm: ppm 容差。
        num_pixels: 张量宽高像素。
        ms1_gate_threshold: 阶段A门控阈值。
        max_workers: 进程数；<=0 时自动使用 CPU 核数-1。
        summary_json: 可选，写入统计 JSON。

    Returns:
        统计字典。
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
    jobs, row_errors = _build_jobs(df)

    failed = len(row_errors)
    for msg in row_errors:
        print(msg)

    if not jobs:
        summary = {"saved": 0, "skipped": 0, "gate_dropped": 0, "failed": failed}
        print("[Done] saved=0 gate_dropped=0 skipped=0 failed=0")
        return summary

    # 1) 单次遍历 mzML，缓存所有所需 EIC
    targets = _build_eic_targets(jobs, mz_tol_ppm=float(mz_tol_ppm))
    eic_cache = extract_multiple_eic(str(mzml_file), targets)

    # 2) 阶段 A：MS1 门控
    gate_dropped = 0
    tasks: List[Tuple[str, str, Tuple[float, float], EICData, List[EICData], int, str]] = []
    used_names: Dict[str, int] = {}

    for job in jobs:
        ms1_data = eic_cache.get(_target_id_ms1(job), (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)))
        _, ms1_int = ms1_data

        if ms1_int.size == 0 or float(np.max(ms1_int)) < float(ms1_gate_threshold):
            gate_dropped += 1
            print(f"[Gate_Drop] {job.lipid_name} MS1 signal too weak.")
            continue

        ms2_data_list: List[EICData] = []
        for i, _frag_mz in enumerate(job.fragments_mz):
            ms2_data_list.append(
                eic_cache.get(
                    _target_id_frag(job, i),
                    (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)),
                )
            )

        safe_name = sanitize_filename(job.lipid_name)
        if safe_name in used_names:
            used_names[safe_name] += 1
            safe_name = f"{safe_name}_{used_names[safe_name]}"
        else:
            used_names[safe_name] = 0

        tasks.append(
            (
                job.lipid_name,
                safe_name,
                job.rt_range,
                ms1_data,
                ms2_data_list,
                int(num_pixels),
                str(out_dir),
            )
        )

    # 3) 阶段 B：并行构建/存盘
    saved = 0
    skipped = 0

    if tasks:
        if max_workers <= 0:
            cpu = int((__import__("os").cpu_count() or 2))
            max_workers = max(1, cpu - 1)

        if max_workers == 1:
            for task in tasks:
                status, message = _encode_and_save_one(*task)
                print(message)
                if status == "saved":
                    saved += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_encode_and_save_one, *task) for task in tasks]
                for fut in as_completed(futures):
                    status, message = fut.result()
                    print(message)
                    if status == "saved":
                        saved += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        failed += 1

    summary = {
        "saved": int(saved),
        "gate_dropped": int(gate_dropped),
        "skipped": int(skipped),
        "failed": int(failed),
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
    parser = argparse.ArgumentParser(description="Phase-3 batch generator (single-pass + parallel)")
    parser.add_argument("--mzml", required=True, type=str, help="mzML 文件路径")
    parser.add_argument("--target-list", required=True, type=str, help="target_list_v2.csv 路径")
    parser.add_argument("--output-dir", type=str, default="data/output_tensors_v2", help="输出目录")
    parser.add_argument("--ppm", type=float, default=10.0, help="m/z 容差（ppm）")
    parser.add_argument("--num-pixels", type=int, default=128, help="输出张量宽高像素")
    parser.add_argument("--ms1-gate", type=float, default=500.0, help="阶段A门控阈值")
    parser.add_argument("--workers", type=int, default=0, help="并发进程数，0=自动")
    parser.add_argument("--summary-json", type=str, default="", help="可选：统计 JSON 输出路径")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    generate_tensors_from_csv_v2(
        mzml_path=str(args.mzml),
        target_list_csv=str(args.target_list),
        output_dir=str(args.output_dir),
        mz_tol_ppm=float(args.ppm),
        num_pixels=int(args.num_pixels),
        ms1_gate_threshold=float(args.ms1_gate),
        max_workers=int(args.workers),
        summary_json=str(args.summary_json),
    )


if __name__ == "__main__":
    main()
