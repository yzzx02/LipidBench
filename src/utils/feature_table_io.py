from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def is_number_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.to_numeric(s, errors="coerce").notna().any()


def ensure_feature_id(df: pd.DataFrame) -> pd.DataFrame:
    if "Feature_ID" in df.columns:
        return df
    out = df.copy()
    out.insert(0, "Feature_ID", [f"F{i}" for i in range(1, len(out) + 1)])
    return out


def load_feature_table(path: Path, algorithm: str) -> pd.DataFrame:
    """Load a feature table (csv/xlsx) and ensure a Feature_ID column exists."""

    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        if algorithm.strip().lower() == "msdial":
            # Best-effort mapping compatible with src/utils/data_io.py
            column_map = {
                "Precursor m/z": "mz",
                "RT left(min)": "RTmin",
                "RT (min)": "RT",
                "RT right (min)": "RTmax",
                "Height": "Height",
                "Area": "Area",
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        return ensure_feature_id(df)

    df = pd.read_csv(path)
    return ensure_feature_id(df)


def normalize_results_base_dir(results_dir: Path, algorithm: str) -> Path:
    """Treat results_dir as a *base* dir.

    If user selects a subfolder like <base>/pyopenms, normalize back to <base>.
    """

    algo = algorithm.strip().lower()
    results_dir = results_dir.resolve()
    if results_dir.name.lower() == algo:
        return results_dir.parent
    return results_dir


def find_feature_table(results_dir: Path, algorithm: str) -> Path:
    """Find the feature table file under a results directory for a given algorithm."""

    algo = algorithm.strip().lower()
    base = normalize_results_base_dir(results_dir, algo)

    if algo == "xcms":
        p = base / "xcms" / "xcms_features.csv"
        if p.exists():
            return p
        raise FileNotFoundError(f"XCMS feature table not found: {p}")

    if algo == "pyopenms":
        p = base / "pyopenms" / "pyopenms_features.csv"
        if p.exists():
            return p
        raise FileNotFoundError(f"pyOpenMS feature table not found: {p}")

    if algo == "asari":
        preferred = base / "asari" / "preferred_Feature_table.csv"
        full = base / "asari" / "full_Feature_table.csv"
        if preferred.exists():
            return preferred
        if full.exists():
            return full
        raise FileNotFoundError(f"Asari feature table not found: {preferred} or {full}")

    if algo == "msdial":
        msdial_dir = base / "msdial"
        if not msdial_dir.exists():
            raise FileNotFoundError(f"MS-DIAL results dir not found: {msdial_dir}")
        candidates = sorted(msdial_dir.glob("*_processed.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
        xlsx = sorted(msdial_dir.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if xlsx:
            return xlsx[0]
        raise FileNotFoundError(f"MS-DIAL processed table not found under: {msdial_dir}")

    raise ValueError(f"Unknown algorithm: {algorithm}")


def standardize_rt_columns_for_display(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    """Return a copy with RT/RTmin/RTmax columns in minutes for display."""

    out = df.copy()
    if algorithm.strip().lower() == "asari":
        if "rtime" in out.columns and "RT" not in out.columns:
            out["RT"] = out["rtime"]
        for col in ["RT", "RTmin", "RTmax"]:
            if col in out.columns:
                out[col] = (pd.to_numeric(out[col], errors="coerce") / 60.0).round(3)

    for col in ["mz", "RT", "RTmin", "RTmax"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def candidate_peak_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"Feature_ID", "mz", "mzmin", "mzmax", "RT", "RTmin", "RTmax", "rtime"}
    candidates: list[str] = []

    for col in df.columns:
        if col in exclude:
            continue
        if str(col).endswith(".mzML"):
            candidates.append(str(col))
            continue
        if col in ("peak_area", "Area", "intensity", "Height"):
            candidates.append(str(col))
            continue
        if is_number_series(df[col]):
            candidates.append(str(col))

    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def suggest_peak_column(df: pd.DataFrame, algorithm: str) -> Optional[str]:
    algo = algorithm.strip().lower()
    if algo == "asari" and "peak_area" in df.columns:
        return "peak_area"
    if algo == "pyopenms" and "intensity" in df.columns:
        return "intensity"
    if algo == "msdial" and "Area" in df.columns:
        return "Area"
    if algo == "xcms":
        mzml_cols = [c for c in df.columns if str(c).endswith(".mzML")]
        if len(mzml_cols) == 1:
            return str(mzml_cols[0])
    return None
