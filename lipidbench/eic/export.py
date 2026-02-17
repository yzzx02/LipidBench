from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from lipidbench.eic.extract_eic_pyopenms import (
    build_ms1_cache,
    extract_eic_from_cache,
)
from lipidbench.utils.feature_table_io import (
    find_feature_table,
    load_feature_table,
    standardize_rt_columns_for_display,
)


def _safe_name(text: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(text).strip())
    return s or "feature"


@dataclass(frozen=True)
class EICImageStyle:
    width_px: int = 400
    height_px: int = 300
    dpi: int = 100
    line_width: float = 1.0
    normalize_intensity: bool = True
    show_axes: bool = True
    show_title: bool = False
    fixed_rt_window_min: float | None = 2.0


def export_eic_images_from_df(
    *,
    df: pd.DataFrame,
    mzml_path: Path,
    out_dir: Path,
    exp: Any | None = None,
    method: Literal["nearest", "window_sum"] = "window_sum",
    ppm: float = 10.0,
    max_features: int = 200,
    rt_pad_min: float = 0.2,
    image_style: EICImageStyle = EICImageStyle(),
) -> int:
    """Batch export EIC PNG images from a display-ready feature table.

    Expected columns: mz, Feature_ID, optional RTmin/RTmax.
    Returns number of exported images.
    """

    if not mzml_path.exists():
        raise FileNotFoundError(f"mzML not found: {mzml_path}")

    try:
        import pyopenms as oms
    except Exception as e:
        raise RuntimeError(f"pyopenms not available: {e}")

    try:
        from matplotlib.figure import Figure
    except Exception as e:
        raise RuntimeError(f"matplotlib not available: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)

    ms_exp = exp
    if ms_exp is None:
        ms_exp = oms.MSExperiment()
        oms.MzMLFile().load(str(mzml_path), ms_exp)

    cache = build_ms1_cache(ms_exp, ms_level=1)

    exported = 0
    n = min(max(int(max_features), 0), len(df))
    figsize = (float(image_style.width_px) / float(image_style.dpi), float(image_style.height_px) / float(image_style.dpi))
    for i in range(n):
        row = df.iloc[i]
        mz = row.get("mz")
        if mz is None or pd.isna(mz):
            continue
        mz = float(mz)

        rtmin = row.get("RTmin")
        rtmax = row.get("RTmax")
        rtmin = float(rtmin) if pd.notna(rtmin) else None
        rtmax = float(rtmax) if pd.notna(rtmax) else None

        feature_id = str(row.get("Feature_ID", f"F{i+1}"))

        trace = extract_eic_from_cache(
            cache,
            target_mz=mz,
            ppm=float(ppm),
            rt_min_limit=(rtmin - float(rt_pad_min)) if rtmin is not None else None,
            rt_max_limit=(rtmax + float(rt_pad_min)) if rtmax is not None else None,
            method=method,
        )

        y = trace.intensity
        if image_style.normalize_intensity and y.size > 0:
            ymax = float(np.max(y))
            y = (y / ymax) if ymax > 0 else y

        x = trace.rt_min

        fig = Figure(figsize=figsize, dpi=image_style.dpi)
        ax = fig.add_subplot(111)
        ax.plot(x, y, linewidth=float(image_style.line_width), color="black")

        if image_style.fixed_rt_window_min is not None and image_style.fixed_rt_window_min > 0:
            if pd.notna(row.get("RT")):
                center = float(row.get("RT"))
            elif x.size > 0:
                center = float(x[len(x) // 2])
            else:
                center = 0.0
            half = float(image_style.fixed_rt_window_min) / 2.0
            ax.set_xlim(center - half, center + half)

        if image_style.normalize_intensity:
            ax.set_ylim(0.0, 1.05)

        if image_style.show_axes:
            ax.set_xlabel("RT (min)")
            ax.set_ylabel("Intensity")
        else:
            ax.set_axis_off()

        if image_style.show_title:
            ax.set_title(f"{mzml_path.stem} | {feature_id} | m/z={mz:.4f}")
        fig.tight_layout()

        out_path = out_dir / f"{_safe_name(feature_id)}.png"
        fig.savefig(out_path, dpi=int(image_style.dpi))
        exported += 1

    return exported


def export_eic_images_from_results(
    *,
    mzml_path: Path,
    algo: str,
    results_dir: Path,
    out_dir: Path,
    method: Literal["nearest", "window_sum"] = "window_sum",
    ppm: float = 10.0,
    max_features: int = 200,
    rt_pad_min: float = 0.2,
    image_style: EICImageStyle = EICImageStyle(),
) -> int:
    feature_path = find_feature_table(results_dir, algo)
    df_raw = load_feature_table(feature_path, algo)
    df = standardize_rt_columns_for_display(df_raw, algo)
    return export_eic_images_from_df(
        df=df,
        mzml_path=mzml_path,
        out_dir=out_dir,
        method=method,
        ppm=ppm,
        max_features=max_features,
        rt_pad_min=rt_pad_min,
        image_style=image_style,
    )
