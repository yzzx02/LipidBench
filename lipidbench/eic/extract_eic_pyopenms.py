from __future__ import annotations

from dataclasses import dataclass
import functools
import time
import multiprocessing as mp
from typing import Optional, Literal
from pathlib import Path
import numpy as np
import pandas as pd

from lipidbench.utils.eic_methods import extract_intensity
from lipidbench.utils.plot_eic import calc_coordinate, gussian_smooth, plot_eic


@dataclass(frozen=True)
class EICTrace:
    rt_min: np.ndarray
    intensity: np.ndarray


def extract_eic_trace(
    exp,
    *,
    target_mz: float,
    tolerance: float = 10.0,
    unit: Literal["ppm", "Da"] = "ppm",
    method: Literal["nearest", "window_sum"] = "nearest",
    rt_min_limit: Optional[float] = None,
    rt_max_limit: Optional[float] = None,
    ms_level: int = 1,
) -> EICTrace:
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
        val = extract_intensity(
            mzs_arr,
            ints_arr,
            target_mz=float(target_mz),
            tolerance=float(tolerance),
            unit=unit,
            method=method,
        )

        rt_list.append(rt_min)
        int_list.append(float(val))

    return EICTrace(rt_min=np.asarray(rt_list, dtype=np.float64), intensity=np.asarray(int_list, dtype=np.float64))


def extract_eic_nearest_ppm(
    exp,
    *,
    target_mz: float,
    ppm: float,
    rt_min_limit: Optional[float] = None,
    rt_max_limit: Optional[float] = None,
    ms_level: int = 1,
) -> EICTrace:
    return extract_eic_trace(
        exp,
        target_mz=target_mz,
        tolerance=ppm,
        unit="ppm",
        method="nearest",
        rt_min_limit=rt_min_limit,
        rt_max_limit=rt_max_limit,
        ms_level=ms_level,
    )


def extract_eic(
    path: str,
    df_info,
    tolerance: float = 10.0,
    unit: Literal["ppm", "Da"] = "ppm",
    method: Literal["nearest", "window_sum"] = "nearest",
) -> np.ndarray:
    """Low-memory 批量 EIC 提取。

    参数:
      - path: mzML 文件路径
            - df_info: 必须包含 `Feature_ID` 与 `mz` 列（DataFrame）
            - tolerance: 容差数值
            - unit: 容差单位，`ppm` 或 `Da`
            - method: `nearest` 或 `window_sum`（默认 `nearest`）

    返回:
      - matrix shape = (len(df_info)+1, n_ms1_scans)
      - 第 0 行为 RT(min)，其余行为各 feature 的 intensity trace
    """

    try:
        import pymzml  # type: ignore
    except ImportError as e:
        raise RuntimeError(f"pymzml is required for EIC extraction: {e}")

    if not (hasattr(df_info, "columns") and "Feature_ID" in df_info.columns and "mz" in df_info.columns):
        raise ValueError("df_info 必须是包含 'Feature_ID' 和 'mz' 列的 DataFrame")

    mz_targets = np.asarray(pd.to_numeric(df_info["mz"], errors="coerce"), dtype=np.float64)
    unit_norm = str(unit).strip().lower()
    if unit_norm not in {"ppm", "da"}:
        raise ValueError("unit 仅支持 'ppm' 或 'Da'")
    tol_value = float(tolerance)
    method_norm = str(method).strip().lower()
    if method_norm not in {"nearest", "window_sum"}:
        raise ValueError("method 仅支持 'nearest' 或 'window_sum'")

    rt_vals: list[float] = []
    traces: list[list[float]] = [[] for _ in range(len(mz_targets))]

    reader = pymzml.run.Reader(str(path))
    for spec in reader:
        ms_level = int(getattr(spec, "ms_level", 1) or 1)
        if ms_level != 1:
            continue

        try:
            rt_min = float(spec.scan_time_in_minutes())
        except Exception:
            continue

        try:
            peaks = spec.peaks("centroided")
        except Exception:
            peaks = spec.peaks("raw")

        if peaks is None or len(peaks) == 0:
            mzs_arr = np.asarray([], dtype=np.float64)
            ints_arr = np.asarray([], dtype=np.float64)
        else:
            arr = np.asarray(peaks, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 2:
                mzs_arr = np.asarray([], dtype=np.float64)
                ints_arr = np.asarray([], dtype=np.float64)
            else:
                mzs_arr = arr[:, 0]
                ints_arr = arr[:, 1]

        rt_vals.append(rt_min)
        if mzs_arr.size == 0:
            for k in range(len(mz_targets)):
                traces[k].append(0.0)
            continue

        for k, target_mz in enumerate(mz_targets):
            if not np.isfinite(target_mz) or target_mz <= 0:
                traces[k].append(0.0)
                continue

            val = extract_intensity(
                mzs_arr,
                ints_arr,
                target_mz=float(target_mz),
                tolerance=tol_value,
                unit=("ppm" if unit_norm == "ppm" else "Da"),
                method=("window_sum" if method_norm == "window_sum" else "nearest"),
            )
            traces[k].append(float(val))

    matrix = np.zeros((len(mz_targets) + 1, len(rt_vals)), dtype=np.float64)
    if rt_vals:
        matrix[0, :] = np.asarray(rt_vals, dtype=np.float64)
        for k, vals in enumerate(traces):
            matrix[k + 1, :] = np.asarray(vals, dtype=np.float64)
    return matrix


def draw_eic(index, paths, eic_list, df_info, image_path, sigma=0, window_min: float = 2.0, image_width_px: int = 400, image_height_px: int = 300, image_dpi: int = 150):
    eic = eic_list[index]
    rt = eic[0]
    feature_counts = len(eic) - 1
    if "RT" not in df_info.columns:
        raise ValueError("draw_eic 需要 df_info 包含 'RT' 列")
    table_rt_min = float(pd.to_numeric(df_info["RT"], errors="coerce").min())
    table_rt_max = float(pd.to_numeric(df_info["RT"], errors="coerce").max())
    assert table_rt_min > rt.min() and table_rt_max < rt.max(), \
        f"Feature table is not compatible with the EIC. The min acceptable RT is {rt.min()}, the max is {rt.max()}"

    name = paths[index].stem
    folder_name = f"{name}"
    current_path = Path.cwd()
    folder_path = current_path / image_path / folder_name
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    records = df_info[["Feature_ID", "mz", "RT"]].to_dict("records")

    for k in range(feature_counts):
        intensity = eic[k + 1]
        feature_id = str(df_info.iloc[k]["Feature_ID"])
        feature_rt = float(pd.to_numeric(df_info.iloc[k]["RT"], errors="coerce"))
        rtmin_raw = pd.to_numeric(df_info.iloc[k].get("RTmin", np.nan), errors="coerce")
        rtmax_raw = pd.to_numeric(df_info.iloc[k].get("RTmax", np.nan), errors="coerce")
        if pd.isna(rtmin_raw) or pd.isna(rtmax_raw):
            # fallback when bounds are unavailable
            rtmin_raw = feature_rt - 0.1
            rtmax_raw = feature_rt + 0.1
        calc_intensity, cal_rt = calc_coordinate(records, intensity, rt, k, windows_size=float(window_min))
        smooth_intensity, smooth_rt = gussian_smooth(calc_intensity, cal_rt, sigma)
        half = float(window_min) / 2.0
        xlim = (feature_rt - half, feature_rt + half)
        plot_eic(
            smooth_rt,
            smooth_intensity,
            feature_id,
            folder_path,
            xlim=xlim,
            width_px=int(image_width_px),
            height_px=int(image_height_px),
            dpi=int(image_dpi),
            normalize_y=False,
            rtmin=float(rtmin_raw),
            rtmax=float(rtmax_raw),
        )


def time_master(func):
    @functools.wraps(func)
    def wrapper_time_master(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper_time_master


def _extract_worker(payload):
    path, df_info, tolerance, unit, method = payload
    return extract_eic(path, df_info, tolerance=tolerance, unit=unit, method=method)


def _draw_worker(payload):
    i, path_list, xic_list, df_info, image_path, sigma, window_min, image_width_px, image_height_px, image_dpi = payload
    draw_eic(i, path_list, xic_list, df_info, image_path, sigma, window_min, image_width_px, image_height_px, image_dpi)


@time_master
def build(paths, info, plot, args):
    """Final EIC execution function.

    - paths: list[Path|str] mzML paths
    - info: DataFrame with Feature_ID/mz/(optional RT)
    - plot: whether to export EIC plots
    - args.processes_number: parallel workers
    - args.method: nearest/window_sum (default nearest)
    - args.unit: ppm/Da (default ppm)
    - args.ppm or args.tolerance: tolerance value
    - args.images_path: output folder for plots
    - args.smooth_sigma: smoothing sigma for draw_eic
    """

    if isinstance(info, pd.DataFrame):
        df_info = info.copy()
    else:
        # minimal compatibility with ndarray/list input: [Feature_ID, mz, RT?]
        arr = np.asarray(info, dtype=object)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("info 必须是 DataFrame 或二维数组(至少两列: Feature_ID, mz)")
        cols = ["Feature_ID", "mz", "RT"][: arr.shape[1]]
        df_info = pd.DataFrame(arr[:, : len(cols)], columns=cols)

    if "Feature_ID" not in df_info.columns or "mz" not in df_info.columns:
        raise ValueError("info 必须包含 'Feature_ID' 与 'mz' 列")

    processes_number = int(getattr(args, "processes_number", 1))
    method = str(getattr(args, "method", "nearest")).strip().lower()
    unit = str(getattr(args, "unit", "ppm")).strip()
    tolerance = float(getattr(args, "tolerance", getattr(args, "ppm", 10.0)))

    path_list = [Path(p) for p in paths]

    unit_norm = "Da" if unit.lower() == "da" else "ppm"
    method_norm = "window_sum" if method == "window_sum" else "nearest"

    tasks = [(str(path), df_info, tolerance, unit_norm, method_norm) for path in path_list]
    if processes_number <= 1:
        xic_list = [_extract_worker(t) for t in tasks]
    else:
        with mp.Pool(processes=processes_number) as pool:
            xic_list = pool.map(_extract_worker, tasks)

    if plot:
        image_path = getattr(args, "images_path", "Results/eic")
        sigma = float(getattr(args, "smooth_sigma", 0))
        # 固定图像参数（深度学习输入一致性）
        window_min = 2.0
        image_width_px = 400
        image_height_px = 300
        image_dpi = 150

        if "RT" not in df_info.columns:
            raise ValueError("plot=True 时 info 必须包含 'RT' 列")

        if processes_number == 1:
            for i in range(len(xic_list)):
                draw_eic(
                    i,
                    path_list,
                    xic_list,
                    df_info,
                    image_path,
                    sigma=sigma,
                    window_min=window_min,
                    image_width_px=image_width_px,
                    image_height_px=image_height_px,
                    image_dpi=image_dpi,
                )
        else:
            draw_tasks = [
                (i, path_list, xic_list, df_info, image_path, sigma, window_min, image_width_px, image_height_px, image_dpi)
                for i in range(len(xic_list))
            ]
            with mp.Pool(processes=processes_number) as pool:
                pool.map(_draw_worker, draw_tasks)

    return xic_list