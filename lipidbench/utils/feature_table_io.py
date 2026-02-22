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
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
        if algorithm.strip().lower() == "msdial":
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
    return ensure_feature_id(pd.read_csv(path))


def normalize_results_base_dir(results_dir: Path, algorithm: str) -> Path:
    algo = algorithm.strip().lower()
    results_dir = results_dir.resolve()
    if results_dir.name.lower() == algo:
        return results_dir.parent
    return results_dir


def find_feature_table(results_dir: Path, algorithm: str) -> Path:
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
    out = df.copy()
    if algorithm.strip().lower() == "asari":
        aliases = {
            "rtime": "RT",
            "rt": "RT",
            "rtime_apex": "RT",
            "rt_apex": "RT",
            "rtime_left_base": "RTmin",
            "rt_left_base": "RTmin",
            "rtime_right_base": "RTmax",
            "rt_right_base": "RTmax",
        }
        for src, dst in aliases.items():
            if src in out.columns and dst not in out.columns:
                out[dst] = out[src]

        for col in ["RT", "RTmin", "RTmax"]:
            if col in out.columns:
                s = pd.to_numeric(out[col], errors="coerce")
                vmax = s.max(skipna=True)
                if pd.notna(vmax) and float(vmax) > 200:
                    s = s / 60.0
                out[col] = s.round(3)

        out = out.drop(
            columns=[
                c
                for c in [
                    "rtime",
                    "rt",
                    "rtime_apex",
                    "rt_apex",
                    "rtime_left_base",
                    "rt_left_base",
                    "rtime_right_base",
                    "rt_right_base",
                ]
                if c in out.columns and c not in {"RT", "RTmin", "RTmax"}
            ],
            errors="ignore",
        )

    for col in ["mz", "RT", "RTmin", "RTmax"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


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


def build_feature_view_table(df_raw: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    """Build GUI display table.

    - Single-sample style: Feature_ID, mz, RT, RTmin, RTmax, PeakArea
    - Multi-sample style (>=2 *.mzML columns): Feature_ID, mz, RT, *.mzML columns
      (drop RTmin/RTmax/PeakArea as requested)
    """

    df = standardize_rt_columns_for_display(df_raw, algorithm)

    mzml_cols = [
        c
        for c in df.columns
        if str(c).lower().endswith(".mzml")
    ]

    base_cols = [c for c in ["Feature_ID", "mz", "RT"] if c in df.columns]

    if len(mzml_cols) >= 2:
        view_cols = base_cols + [c for c in mzml_cols if c not in base_cols]
        return df.loc[:, view_cols].copy()

    peak_col = suggest_peak_column(df_raw, algorithm)
    if peak_col is not None and peak_col in df.columns:
        peak_area = pd.to_numeric(df[peak_col], errors="coerce")
    else:
        peak_area = pd.Series([pd.NA] * len(df), index=df.index)

    return pd.DataFrame(
        {
            "Feature_ID": df.get("Feature_ID"),
            "mz": df.get("mz"),
            "RT": df.get("RT"),
            "RTmin": df.get("RTmin"),
            "RTmax": df.get("RTmax"),
            "PeakArea": peak_area,
        }
    )
