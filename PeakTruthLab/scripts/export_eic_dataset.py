from pathlib import Path
from types import SimpleNamespace
import pandas as pd

from lipidbench.eic.extract_eic_pyopenms import build


def main() -> None:
    feature_csv = Path(r"D:\LipidBench\Results\pyopenms\xcms\xcms_features.csv")
    mzml_path = Path(r"D:\LipidBench\data\DIA_mzML\HILIC-Pos-SWATH-25Da-20140701_08_GB004467_Swath25Da.mzML")
    out_root = Path(r"D:\LipidBench\PeakTruthLab\datasets\eic_images")

    if not feature_csv.exists():
        raise FileNotFoundError(feature_csv)
    if not mzml_path.exists():
        raise FileNotFoundError(mzml_path)

    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(feature_csv)
    keep_cols = [c for c in ["Feature_ID", "mz", "RTmin", "RT", "RTmax"] if c in df.columns]
    df_info = df[keep_cols].dropna(subset=["Feature_ID", "mz", "RT"]).copy()

    args = SimpleNamespace(
        processes_number=1,
        method="nearest",
        unit="ppm",
        tolerance=10.0,
        images_path=str(out_root),
        smooth_sigma=0.0,
    )

    build(paths=[mzml_path], info=df_info, plot=True, args=args)

    img_dir = out_root / mzml_path.stem
    jpgs = sorted(img_dir.glob("*.jpeg"))
    jsons = sorted(img_dir.glob("*.json"))
    print("feature_rows:", len(df_info))
    print("output_dir:", img_dir)
    print("jpeg_count:", len(jpgs))
    print("json_count:", len(jsons))


if __name__ == "__main__":
    main()
