import pandas as pd


def feature_id_sort_key(feature_id_series: pd.Series) -> pd.Series:
    s = feature_id_series.astype(str).str.strip()
    key = s.str.extract(r"^[Ff](?:[Tt])?(\d+)$")[0]
    key = key.fillna(s.str.extract(r"^(\d+)$")[0])
    key = key.fillna(s.str.extract(r"(\d+)$")[0])
    return pd.to_numeric(key, errors="coerce")


def normalize_feature_id(df: pd.DataFrame, column: str = "Feature_ID") -> pd.DataFrame:
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
