from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from lipidbench.utils.feature_table_io import load_feature_table, standardize_rt_columns_for_display, suggest_peak_column


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _infer_algo_name(path: Path) -> str:
    p = str(path).lower()
    if "pyopenms" in p:
        return "pyopenms"
    if "xcms" in p:
        return "xcms"
    if "asari" in p:
        return "asari"
    if "msdial" in p or "ms-dial" in p:
        return "msdial"
    return path.stem


def load_and_standardize_table(path: Path, algo: str | None = None) -> pd.DataFrame:
    algo_name = (algo or _infer_algo_name(path)).strip().lower()
    raw = load_feature_table(path, algo_name)
    df = standardize_rt_columns_for_display(raw, algo_name)
    if "RT" not in df.columns:
        if "RTmin" in df.columns and "RTmax" in df.columns:
            df["RT"] = (_to_numeric(df["RTmin"]) + _to_numeric(df["RTmax"])) / 2.0
        else:
            raise ValueError(f"{path} 缺少 RT 列")

    if "mz" not in df.columns:
        raise ValueError(f"{path} 缺少 mz 列")

    peak_col = suggest_peak_column(raw, algo_name)
    if peak_col is not None and peak_col in df.columns:
        intensity = _to_numeric(df[peak_col]).fillna(0.0)
    else:
        mzml_cols = [c for c in df.columns if str(c).lower().endswith(".mzml")]
        if mzml_cols:
            intensity = df[mzml_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        else:
            intensity = pd.Series([1.0] * len(df), index=df.index)

    out = pd.DataFrame(
        {
            "Feature_ID": df.get("Feature_ID"),
            "mz": _to_numeric(df["mz"]),
            "RT": _to_numeric(df["RT"]),
            "intensity": _to_numeric(intensity),
        }
    ).dropna(subset=["mz", "RT"])
    return out


def align_feature_tables(
    table_map: dict[str, pd.DataFrame],
    *,
    mz_tol_da: float = 0.01,
    rt_tol_sec: float = 10.0,
) -> pd.DataFrame:
    """Align multi-algorithm feature tables by mz/RT tolerances.

    Returns a consensus table with `mz`, `RT`, and per-algorithm intensity columns.
    """

    if not table_map:
        return pd.DataFrame()

    rt_tol_min = float(rt_tol_sec) / 60.0
    algos = list(table_map.keys())
    rows: list[dict] = []

    for algo in algos:
        df = table_map[algo]
        local = df.copy()
        local = local.sort_values(["mz", "RT"], ascending=[True, True])

        for _, rec in local.iterrows():
            mz = float(rec["mz"])
            rt = float(rec["RT"])
            inten = float(rec.get("intensity", 0.0) or 0.0)

            matched_idx = None
            for i, r in enumerate(rows):
                if abs(float(r["mz"]) - mz) <= float(mz_tol_da) and abs(float(r["RT"]) - rt) <= rt_tol_min:
                    matched_idx = i
                    break

            if matched_idx is None:
                new_row = {"mz": mz, "RT": rt, "_count": 1}
                for a in algos:
                    new_row[f"intensity_{a}"] = 0.0
                new_row[f"intensity_{algo}"] = inten
                rows.append(new_row)
            else:
                r = rows[matched_idx]
                c = int(r["_count"])
                r["mz"] = (float(r["mz"]) * c + mz) / (c + 1)
                r["RT"] = (float(r["RT"]) * c + rt) / (c + 1)
                r["_count"] = c + 1
                r[f"intensity_{algo}"] = max(float(r.get(f"intensity_{algo}", 0.0)), inten)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.drop(columns=["_count"], errors="ignore")
    for a in algos:
        col = f"intensity_{a}"
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out[f"detected_{a}"] = (out[col] > 0).astype(int)

    out = out.sort_values(["mz", "RT"], ascending=[True, True]).reset_index(drop=True)
    out.insert(0, "Aligned_ID", [f"A{i}" for i in range(1, len(out) + 1)])
    return out


def missing_features_for_algo(aligned_df: pd.DataFrame, algo: str, all_algos: Iterable[str]) -> pd.DataFrame:
    target = algo.strip()
    detected_col = f"detected_{target}"
    if detected_col not in aligned_df.columns:
        raise ValueError(f"对齐表缺少列: {detected_col}")

    others = [a for a in all_algos if a != target]
    if others:
        other_detect_sum = aligned_df[[f"detected_{a}" for a in others]].sum(axis=1)
        mask = (aligned_df[detected_col] == 0) & (other_detect_sum > 0)
    else:
        mask = aligned_df[detected_col] == 0

    cols = ["Aligned_ID", "mz", "RT"]
    return aligned_df.loc[mask, cols].reset_index(drop=True)
