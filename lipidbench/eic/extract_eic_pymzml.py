from __future__ import annotations

from pathlib import Path
from typing import Optional

from lipidbench.eic.extract_eic_pyopenms import EICTrace, extract_eic_nearest_ppm


def extract_eic_nearest_ppm_from_mzml(
    mzml_path: str | Path,
    *,
    target_mz: float,
    ppm: float,
    rt_min_limit: Optional[float] = None,
    rt_max_limit: Optional[float] = None,
    ms_level: int = 1,
) -> EICTrace:
    """Backward-compatible wrapper.

    Internally uses pyopenms backend for consistency with the rest of LipidBench.
    """

    try:
        import pyopenms as oms
    except Exception as e:
        raise RuntimeError(f"pyopenms not available: {e}")

    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(Path(mzml_path)), exp)
    return extract_eic_nearest_ppm(
        exp,
        target_mz=target_mz,
        ppm=ppm,
        rt_min_limit=rt_min_limit,
        rt_max_limit=rt_max_limit,
        ms_level=ms_level,
    )
