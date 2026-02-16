import pandas as pd


def feature_id_sort_key(feature_id_series: pd.Series) -> pd.Series:
    """Return numeric sort keys for mixed Feature_ID formats.

    Supported examples: 1, 001, F1, f2, FT3, ft004.
    """
    s = feature_id_series.astype(str).str.strip()

    # Prefer explicit patterns first
    key = s.str.extract(r"^[Ff](?:[Tt])?(\d+)$")[0]
    key = key.fillna(s.str.extract(r"^(\d+)$")[0])
    # Fallback: extract any trailing digits if present
    key = key.fillna(s.str.extract(r"(\d+)$")[0])

    return pd.to_numeric(key, errors="coerce")


def normalize_feature_id(df: pd.DataFrame, column: str = "Feature_ID") -> pd.DataFrame:
    """Normalize Feature_ID to F1..Fn and place it as the first column.

    Behavior:
    - If `column` exists, rows are stably sorted by numeric Feature_ID when possible,
      then reassigned to F1..Fn.
    - If `column` does not exist, rows keep current order and get F1..Fn.
    """
    out = df.copy()

    if column in out.columns:
        key = feature_id_sort_key(out[column])
        if key.notna().any():
            out = (
                out.assign(_feature_id_sort_key=key)
                .sort_values("_feature_id_sort_key", kind="mergesort")
                .drop(columns=["_feature_id_sort_key"])
                .reset_index(drop=True)
            )
        else:
            out = out.reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    out[column] = [f"F{i}" for i in range(1, len(out) + 1)]

    cols = [column] + [c for c in out.columns if c != column]
    return out[cols]
