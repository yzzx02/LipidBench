import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

from lipidbench.utils.config_io import load_config
from lipidbench.utils.feature_table_io import find_feature_table, load_feature_table, standardize_rt_columns_for_display


def _get_runner(algo: str):
    """Return the pipeline function for an algorithm without importing all runners upfront."""

    algo = algo.strip().lower()
    mapping = {
        "xcms": ("lipidbench.runners.run_xcms", "run_xcms_pipeline"),
        "ms-dial": ("lipidbench.runners.run_msdial", "run_msdial_pipeline"),
        "pyopenms": ("lipidbench.runners.run_pyopenms", "run_pyopenms_pipeline"),
        "asari": ("lipidbench.runners.run_asari", "run_asari_pipeline"),
    }
    if algo not in mapping:
        raise ValueError(f"Unknown algorithm: {algo}")

    mod_name, fn_name = mapping[algo]
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def parse_args():
    parser = argparse.ArgumentParser(description="LipidBench algorithm runner")
    parser.add_argument("--algo", default="asari,xcms,ms-dial,pyopenms", help="Algorithm(s) to run, comma-separated")
    parser.add_argument("--export-eic", action="store_true", help="Run EIC image export after each algorithm")
    parser.add_argument("--eic-mzml", type=Path, help="mzML file used for EIC export")
    parser.add_argument("--eic-ppm", type=float, help="m/z tolerance ppm for EIC export")
    parser.add_argument("--eic-unit", choices=["ppm", "Da"], help="EIC tolerance unit")
    parser.add_argument("--eic-method", choices=["nearest", "window_sum"], help="EIC extraction method")
    parser.add_argument("--eic-max-features", type=int, help="Maximum features to export")
    parser.add_argument("--eic-processes", type=int, help="Parallel worker count for EIC build")
    parser.add_argument("--eic-smooth-sigma", type=float, help="Gaussian smoothing sigma for EIC plotting")
    return parser.parse_args()


def _yaml_get_eic_export_cfg(config: dict) -> dict:
    cfg = config.get("eic_export", {})
    if not isinstance(cfg, dict):
        return {}
    return cfg


def main():
    config = load_config()
    args = parse_args()
    eic_cfg = _yaml_get_eic_export_cfg(config)
    paths_cfg = config.get("paths", {}) if isinstance(config, dict) else {}

    do_export = bool(args.export_eic or bool(eic_cfg.get("enabled", False)))

    eic_mzml = args.eic_mzml or (Path(eic_cfg["mzml"]).resolve() if eic_cfg.get("mzml") else None)
    output_root = Path(paths_cfg.get("output_dir", "./Results")).resolve()
    eic_ppm = float(args.eic_ppm if args.eic_ppm is not None else eic_cfg.get("ppm", 10.0))
    eic_unit = str(args.eic_unit if args.eic_unit is not None else eic_cfg.get("unit", "ppm"))
    eic_method = str(args.eic_method if args.eic_method is not None else eic_cfg.get("method", "nearest"))
    eic_max_features = int(args.eic_max_features if args.eic_max_features is not None else eic_cfg.get("max_features", 200))
    eic_processes = int(args.eic_processes if args.eic_processes is not None else config.get("common_params", {}).get("n_workers", 1))
    eic_smooth_sigma = float(args.eic_smooth_sigma if args.eic_smooth_sigma is not None else eic_cfg.get("smooth_sigma", 0.0))
    # 固定图像参数（深度学习输入一致性）
    eic_window_min = 2.0
    eic_image_width_px = 400
    eic_image_height_px = 300
    eic_image_dpi = 100

    algo_list = [a.strip().lower() for a in args.algo.split(",") if a.strip()]
    for algo in algo_list:
        runner = _get_runner(algo)
        runner(config)

        if do_export:
            if eic_mzml is None:
                print(
                    "[EIC export] skipped: missing eic_mzml "
                    "(set CLI args or config.yaml:eic_export)"
                )
                continue
            try:
                from lipidbench.eic.extract_eic_pyopenms import build as build_eic

                feature_path = find_feature_table(output_root, algo)
                df_raw = load_feature_table(feature_path, algo)
                df = standardize_rt_columns_for_display(df_raw, algo)
                if "RT" not in df.columns:
                    print(f"[EIC export] {algo}: skipped (feature table has no RT column)")
                    continue
                keep_cols = [c for c in ["Feature_ID", "mz", "RT", "RTmin", "RTmax"] if c in df.columns]
                df_info = df[keep_cols].dropna(subset=["mz", "RT"]).head(eic_max_features)
                if df_info.empty:
                    print(f"[EIC export] {algo}: skipped (no valid Feature_ID/mz/RT rows)")
                    continue

                out_dir = (output_root / "eic_export").resolve()
                eic_args = SimpleNamespace(
                    processes_number=eic_processes,
                    method=eic_method,
                    unit=eic_unit,
                    tolerance=eic_ppm,
                    images_path=str(out_dir),
                    smooth_sigma=eic_smooth_sigma,
                    window_min=eic_window_min,
                    image_width_px=eic_image_width_px,
                    image_height_px=eic_image_height_px,
                    image_dpi=eic_image_dpi,
                )
                _ = build_eic([eic_mzml], df_info, True, eic_args)
                print(f"[EIC export] {algo}: {len(df_info)} images -> {out_dir}/<mzML_name>")
            except Exception as e:
                print(f"[EIC export] {algo}: failed: {e}")


if __name__ == "__main__":
    main()
