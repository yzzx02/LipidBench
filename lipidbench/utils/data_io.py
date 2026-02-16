import pandas as pd
from pathlib import Path

from lipidbench.utils.feature_id import normalize_feature_id


def load_asari_results(project_dir, output_dir=None, cleanup_project=True):
    project_dir = Path(project_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else project_dir

    full_tsv = project_dir / "export" / "full_Feature_table.tsv"
    pref_tsv = project_dir / "preferred_Feature_table.tsv"
    required = [full_tsv, pref_tsv]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Asari tables: {missing}")

    def _normalize_and_sort(df: pd.DataFrame) -> pd.DataFrame:
        if "id_number" in df.columns:
            df = df.rename(columns={"id_number": "Feature_ID"})
        if "[peak]id_number" in df.columns:
            df = df.rename(columns={"[peak]id_number": "Feature_ID"})
        if "rtime_left_base" in df.columns:
            df = df.rename(columns={"rtime_left_base": "RTmin"})
        if "rtime_right_base" in df.columns:
            df = df.rename(columns={"rtime_right_base": "RTmax"})

        if "Feature_ID" not in df.columns:
            raise ValueError("Asari table missing Feature_ID (id_number)")

        return normalize_feature_id(df, column="Feature_ID")

    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = _normalize_and_sort(pd.read_csv(full_tsv, sep="\t"))
    pref_df = _normalize_and_sort(pd.read_csv(pref_tsv, sep="\t"))

    full_csv = output_dir / "full_Feature_table.csv"
    pref_csv = output_dir / "preferred_Feature_table.csv"
    full_df.to_csv(full_csv, index=False)
    pref_df.to_csv(pref_csv, index=False)

    full_tsv.unlink(missing_ok=True)
    pref_tsv.unlink(missing_ok=True)

    if cleanup_project and project_dir != output_dir:
        import shutil

        try:
            under_output = project_dir.is_relative_to(output_dir)
        except AttributeError:
            under_output = str(project_dir).lower().startswith(str(output_dir).lower())

        if under_output:
            shutil.rmtree(project_dir, ignore_errors=True)

    print(f"Processed Asari CSVs written to: {output_dir}")
    return output_dir


def load_xcms_results(file_path):
    data = pd.read_csv(file_path, index_col=0)
    new_data = {}
    sample_cols = [col for col in data.columns if col.endswith(".mzML")]
    has_sn = "sn" in data.columns

    for col in sample_cols:
        data[col] = data[col].fillna(0)
    for index, row in data.iterrows():
        entry = {
            "mz": round(row["mzmed"], 4),
            "RTmin": round(row["rtmin"] / 60, 3),
            "RT": round(row["rtmed"] / 60, 3),
            "RTmax": round(row["rtmax"] / 60, 3),
            "npeaks": row["npeaks"],
        }
        if has_sn:
            entry["sn"] = row["sn"]
        for col in sample_cols:
            entry[col] = row[col]
        new_data[index] = entry

    df = pd.DataFrame.from_dict(new_data, orient="index")
    df = normalize_feature_id(df, column="Feature_ID")
    df.to_csv(file_path, index=False, float_format="%.4f")
    return df


def load_msdial_results(file_path, outputfile):
    data = pd.read_excel(file_path, index_col=0)
    sample_cols = [col for col in data.columns if col.endswith(".mzML")]
    data[sample_cols] = data[sample_cols].fillna(0)
    column_map = {
        "Precursor m/z": "mz",
        "RT left(min)": "RTmin",
        "RT (min)": "RT",
        "RT right (min)": "RTmax",
        "Height": "Height",
        "Area": "Area",
        "Estimated noise": "Estimated noise",
        "S/N": "S/N",
        "Sharpness": "Sharpness",
        "Gaussian similarity": "Gaussian similarity",
        "Ideal slope": "Ideal slope",
        "Symmetry": "Symmetry",
    }
    keep_cols = list(column_map.keys()) + sample_cols
    df = data[keep_cols].copy()
    df.rename(columns=column_map, inplace=True)
    df = normalize_feature_id(df, column="Feature_ID")
    df.to_csv(f"{outputfile}", index=False)
    print(f"Processed MS-DIAL results saved to {outputfile}")
    return df


def load_pyopenms_results(
    file_path,
    input_dir=None,
    mz_tol=10.0,
    min_fwhm=2.5,
    max_fwhm=60.0,
    noise=1000.0,
    sn=5.0,
    force_recompute_bounds=False,
):
    df = pd.read_csv(file_path)

    rt_in_seconds = None
    if "RT" in df.columns:
        rt_numeric = pd.to_numeric(df["RT"], errors="coerce")
        rt_max = rt_numeric.max(skipna=True)
        rt_in_seconds = bool(rt_max and rt_max > 200)
    rt_to_seconds_factor = 1.0 if rt_in_seconds else 60.0
    rt_to_minutes_factor = 1.0 / 60.0 if rt_in_seconds else 1.0

    for col in ["RTstart", "RTend"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.mask(s.abs() > 1e300)
            if rt_in_seconds is False:
                s = s * rt_to_seconds_factor
            df[col] = s

    for col in ["MZstart", "MZend"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.mask(s.abs() > 1e300)
            df[col] = s

    has_rt_bounds = "RTstart" in df.columns and "RTend" in df.columns
    has_mz_bounds = "MZstart" in df.columns and "MZend" in df.columns

    needs_rt_bounds = force_recompute_bounds or ("RTmin" not in df.columns or "RTmax" not in df.columns)
    needs_mz_bounds = force_recompute_bounds or ("mzmin" not in df.columns or "mzmax" not in df.columns)

    if needs_rt_bounds and not has_rt_bounds:
        raise ValueError("pyOpenMS CSV missing RTstart/RTend")
    if needs_mz_bounds and not has_mz_bounds:
        raise ValueError("pyOpenMS CSV missing MZstart/MZend")

    if has_rt_bounds and needs_rt_bounds:
        valid_rt = df["RTstart"].notna() & df["RTend"].notna() & (df["RTstart"] <= df["RTend"])
        df.loc[valid_rt, "RTmin"] = df.loc[valid_rt, "RTstart"]
        df.loc[valid_rt, "RTmax"] = df.loc[valid_rt, "RTend"]

    if has_mz_bounds and needs_mz_bounds:
        valid_mz = df["MZstart"].notna() & df["MZend"].notna() & (df["MZstart"] <= df["MZend"])
        df.loc[valid_mz, "mzmin"] = df.loc[valid_mz, "MZstart"]
        df.loc[valid_mz, "mzmax"] = df.loc[valid_mz, "MZend"]

    if "RT" in df.columns:
        rt_numeric = pd.to_numeric(df["RT"], errors="coerce")
        if rt_in_seconds is False:
            rt_numeric = rt_numeric * rt_to_seconds_factor
        df["RT"] = rt_numeric

    if "RT" not in df.columns and "RTstart" in df.columns and "RTend" in df.columns:
        df["RT"] = (pd.to_numeric(df["RTstart"], errors="coerce") + pd.to_numeric(df["RTend"], errors="coerce")) / 2.0
    if "mz" not in df.columns and "MZstart" in df.columns and "MZend" in df.columns:
        df["mz"] = (pd.to_numeric(df["MZstart"], errors="coerce") + pd.to_numeric(df["MZend"], errors="coerce")) / 2.0

    cols_to_drop = ["RTstart", "RTend", "MZstart", "MZend", "peptide_sequence", "peptide_score", "ID_filename", "ID_native_id", "charge", "quality", "sequence"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    df = normalize_feature_id(df, column="Feature_ID")

    for col in ["mz", "mzmin", "mzmax"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    for col in ["RT", "RTmin", "RTmax"]:
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce") * rt_to_minutes_factor).round(3)

    base_cols = [c for c in ["Feature_ID", "mz", "mzmin", "mzmax", "RT", "RTmin", "RTmax"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    df.to_csv(file_path, index=False, float_format="%.4f")
    print(f"Processed pyOpenMS results overwritten: {file_path}")
    return df
