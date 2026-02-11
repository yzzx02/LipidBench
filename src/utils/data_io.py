import pandas as pd


def _asari_sort_key_feature_id(feature_id_series: pd.Series) -> pd.Series:
    s = feature_id_series.astype(str)
    extracted = s.str.extract(r"^[Ff](\d+)$")[0]
    key = pd.to_numeric(extracted, errors="coerce")
    return key


def load_asari_results(project_dir, output_dir=None, cleanup_project=True):
    """Post-process Asari project outputs.

    Spec:
    - Keep only two CSVs in the Asari output folder: full_Feature_table + preferred_Feature_table.
    - Rename columns:
      - id_number / [peak]id_number -> Feature_ID
      - rtime_left_base / rtime_right_base -> RTmin / RTmax
    - Sort by Feature_ID in ascending F<number> order.
    - Delete the original TSVs after writing CSV.
    - Optionally delete the whole project directory if it is under output_dir.
    """

    from pathlib import Path

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

        key = _asari_sort_key_feature_id(df["Feature_ID"])
        if key.notna().any():
            df = (
                df.assign(_sort_key=key)
                .sort_values("_sort_key", kind="mergesort")
                .drop(columns=["_sort_key"])
            )
        else:
            df = df.sort_values("Feature_ID", kind="mergesort")
        return df

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
    data=pd.read_csv(file_path,index_col=0)
    new_data={}
    sample_cols=[col for col in data.columns if col.endswith(".mzML") ]
    
    # Check if 'sn' column exists (only for single sample)
    has_sn = 'sn' in data.columns

    for col in sample_cols:
        data[col]=data[col].fillna(0)
    for index,row in data.iterrows():
        entry = {
            'mz':round(row['mzmed'],4),
            'RTmin':round(row['rtmin']/60,3),
            'RT':round(row['rtmed']/60,3),
            'RTmax':round(row['rtmax']/60,3),
            'npeaks':row['npeaks'],
            # 'snthresh':row['snthresh'], # Removed as snthresh is user param, not result
        }
        
        # Add actual SN if available
        if has_sn:
            entry['sn'] = row['sn']

        for col in sample_cols:
            entry[col]=row[col]
            
        new_data[index] = entry

    df = pd.DataFrame.from_dict(new_data,orient='index')
    df.to_csv(file_path,index_label='Feature_ID',float_format='%.4f')
    return df

def load_msdial_results(file_path,outputfile):
    data=pd.read_excel(file_path,index_col=0)
    sample_cols=[col for col in data.columns if col.endswith(".mzML") ]
    data[sample_cols] = data[sample_cols].fillna(0)
    column_map = {
            'Precursor m/z': 'mz',
            'RT left(min)': 'RTmin',
            'RT (min)': 'RT',
            'RT right (min)': 'RTmax',
            'Height': 'Height',
            'Area': 'Area',
            'Estimated noise': 'Estimated noise',
            'S/N': 'S/N',
            'Sharpness': 'Sharpness',
            'Gaussian similarity': 'Gaussian similarity',
            'Ideal slope': 'Ideal slope',
            'Symmetry': 'Symmetry'
        }
    keep_cols = list(column_map.keys()) + sample_cols
    df = data[keep_cols].copy()
    df.rename(columns=column_map, inplace=True)
    df.to_csv(f'{outputfile}', index=True, index_label='Feature_ID')
    print(f"Processed MS-DIAL results saved to {outputfile}")
    #添加一列数字标记feature id
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
    """Post-process pyOpenMS CSV in-place.

    Assumes FeatureFindingMetabo ran with `report_convex_hulls=true`, so the
    exported get_df() CSV contains: RTstart/RTend/MZstart/MZend.

    - Ensures RTmin/RTmax exist (derived from RTstart/RTend).
    - Ensures mzmin/mzmax exist (derived from MZstart/MZend).
    - Drops ID/quality columns and invalid RTstart/RTend/MZstart/MZend.
    - Converts RT units to minutes and overwrites the original CSV.
    """

    df = pd.read_csv(file_path)

    # Heuristic: determine whether RT is in seconds (raw pyopenms) or minutes (already processed)
    rt_in_seconds = None
    if "RT" in df.columns:
        rt_numeric = pd.to_numeric(df["RT"], errors="coerce")
        rt_max = rt_numeric.max(skipna=True)
        # LC-MS RT in minutes is typically < ~200; seconds can be >200 easily.
        rt_in_seconds = bool(rt_max and rt_max > 200)
    rt_to_seconds_factor = 1.0 if rt_in_seconds else 60.0  # if minutes -> seconds
    rt_to_minutes_factor = 1.0 / 60.0 if rt_in_seconds else 1.0

    # Prefer true bounds from get_df() (convex hull bounding box).
    # pyopenms may write missing bounds as +/- max float; treat those as NaN.
    for col in ["RTstart", "RTend"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            s = s.mask(s.abs() > 1e300)
            # Convert to seconds for internal computations if file is already in minutes
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
        raise ValueError(
            "pyOpenMS CSV is missing RTstart/RTend (required to compute RTmin/RTmax). "
            "Please enable report_convex_hulls=true in FeatureFindingMetabo and re-run pyOpenMS."
        )
    if needs_mz_bounds and not has_mz_bounds:
        raise ValueError(
            "pyOpenMS CSV is missing MZstart/MZend (required to compute mzmin/mzmax). "
            "Please enable report_convex_hulls=true in FeatureFindingMetabo and re-run pyOpenMS."
        )

    if has_rt_bounds and needs_rt_bounds:
        valid_rt = df["RTstart"].notna() & df["RTend"].notna() & (df["RTstart"] <= df["RTend"])
        df.loc[valid_rt, "RTmin"] = df.loc[valid_rt, "RTstart"]
        df.loc[valid_rt, "RTmax"] = df.loc[valid_rt, "RTend"]

    if has_mz_bounds and needs_mz_bounds:
        valid_mz = df["MZstart"].notna() & df["MZend"].notna() & (df["MZstart"] <= df["MZend"])
        df.loc[valid_mz, "mzmin"] = df.loc[valid_mz, "MZstart"]
        df.loc[valid_mz, "mzmax"] = df.loc[valid_mz, "MZend"]

    # Convert RT column to seconds for internal computations
    if "RT" in df.columns:
        rt_numeric = pd.to_numeric(df["RT"], errors="coerce")
        if rt_in_seconds is False:
            rt_numeric = rt_numeric * rt_to_seconds_factor
        df["RT"] = rt_numeric

    # If RT is missing for some reason, compute as midpoint of bounds.
    if "RT" not in df.columns and "RTstart" in df.columns and "RTend" in df.columns:
        df["RT"] = (pd.to_numeric(df["RTstart"], errors="coerce") + pd.to_numeric(df["RTend"], errors="coerce")) / 2.0
    if "mz" not in df.columns and "MZstart" in df.columns and "MZend" in df.columns:
        df["mz"] = (pd.to_numeric(df["MZstart"], errors="coerce") + pd.to_numeric(df["MZend"], errors="coerce")) / 2.0

    # Drop unhelpful/invalid columns
    cols_to_drop = [
        # invalid bounds (often +/- max float)
        "RTstart",
        "RTend",
        "MZstart",
        "MZend",
        # ID-related fields
        "peptide_sequence",
        "peptide_score",
        "ID_filename",
        "ID_native_id",
        # usually not needed
        "charge",
        "quality",
        # legacy names
        "sequence",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Create Feature_ID
    if "Feature_ID" not in df.columns:
        df.insert(0, "Feature_ID", range(1, len(df) + 1))

    # Unit conversions / rounding
    for col in ["mz", "mzmin", "mzmax"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    # Convert seconds->minutes if needed; otherwise keep existing minutes
    for col in ["RT", "RTmin", "RTmax"]:
        if col in df.columns:
            df[col] = (pd.to_numeric(df[col], errors="coerce") * rt_to_minutes_factor).round(3)

    # Keep a minimal, model-friendly set of columns (others, e.g., consensus intensity columns, are kept)
    base_cols = [c for c in ["Feature_ID", "mz", "mzmin", "mzmax", "RT", "RTmin", "RTmax"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + other_cols]

    # Overwrite original CSV in-place
    df.to_csv(file_path, index=False, float_format="%.4f")
    print(f"Processed pyOpenMS results overwritten: {file_path}")
    return df