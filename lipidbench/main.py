import argparse
import importlib
from pathlib import Path

from lipidbench.utils.config_io import load_config


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
    parser.add_argument("--eic-results-dir", type=Path, help="Results base directory for feature tables")
    parser.add_argument("--eic-out-dir", type=Path, help="Output directory for exported EIC images")
    parser.add_argument("--eic-ppm", type=float, help="m/z tolerance ppm for EIC export")
    parser.add_argument("--eic-method", choices=["nearest", "window_sum"], help="EIC extraction method")
    parser.add_argument("--eic-max-features", type=int, help="Maximum features to export")
    parser.add_argument("--eic-rt-pad-min", type=float, help="RT padding (minutes) around RTmin/RTmax")
    parser.add_argument("--eic-width-px", type=int, help="Export image width in pixels")
    parser.add_argument("--eic-height-px", type=int, help="Export image height in pixels")
    parser.add_argument("--eic-dpi", type=int, help="Export image DPI")
    parser.add_argument("--eic-line-width", type=float, help="Line width for EIC plot")
    parser.add_argument("--eic-normalize-intensity", choices=["true", "false"], help="Normalize intensity to [0,1]")
    parser.add_argument("--eic-show-axes", choices=["true", "false"], help="Show axes in EIC images")
    parser.add_argument("--eic-show-title", choices=["true", "false"], help="Show title in EIC images")
    parser.add_argument("--eic-fixed-rt-window-min", type=float, help="Fixed RT window size in minutes; <=0 disables")
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

    do_export = bool(args.export_eic or bool(eic_cfg.get("enabled", False)))

    eic_mzml = args.eic_mzml or (Path(eic_cfg["mzml"]).resolve() if eic_cfg.get("mzml") else None)
    eic_results_dir = args.eic_results_dir or (
        Path(eic_cfg["results_dir"]).resolve() if eic_cfg.get("results_dir") else None
    )
    eic_out_dir = args.eic_out_dir or (Path(eic_cfg["out_dir"]).resolve() if eic_cfg.get("out_dir") else None)
    eic_ppm = float(args.eic_ppm if args.eic_ppm is not None else eic_cfg.get("ppm", 10.0))
    eic_method = str(args.eic_method if args.eic_method is not None else eic_cfg.get("method", "window_sum"))
    eic_max_features = int(args.eic_max_features if args.eic_max_features is not None else eic_cfg.get("max_features", 200))
    eic_rt_pad_min = float(args.eic_rt_pad_min if args.eic_rt_pad_min is not None else eic_cfg.get("rt_pad_min", 0.2))
    eic_width_px = int(args.eic_width_px if args.eic_width_px is not None else eic_cfg.get("width_px", 400))
    eic_height_px = int(args.eic_height_px if args.eic_height_px is not None else eic_cfg.get("height_px", 300))
    eic_dpi = int(args.eic_dpi if args.eic_dpi is not None else eic_cfg.get("dpi", 100))
    eic_line_width = float(args.eic_line_width if args.eic_line_width is not None else eic_cfg.get("line_width", 1.0))
    eic_normalize_intensity = (
        str(args.eic_normalize_intensity).lower() == "true"
        if args.eic_normalize_intensity is not None
        else bool(eic_cfg.get("normalize_intensity", True))
    )
    eic_show_axes = (
        str(args.eic_show_axes).lower() == "true"
        if args.eic_show_axes is not None
        else bool(eic_cfg.get("show_axes", True))
    )
    eic_show_title = (
        str(args.eic_show_title).lower() == "true"
        if args.eic_show_title is not None
        else bool(eic_cfg.get("show_title", False))
    )
    eic_fixed_rt_window_min_raw = (
        float(args.eic_fixed_rt_window_min)
        if args.eic_fixed_rt_window_min is not None
        else float(eic_cfg.get("fixed_rt_window_min", 2.0))
    )
    eic_fixed_rt_window_min = eic_fixed_rt_window_min_raw if eic_fixed_rt_window_min_raw > 0 else None

    algo_list = [a.strip().lower() for a in args.algo.split(",") if a.strip()]
    for algo in algo_list:
        runner = _get_runner(algo)
        runner(config)

        if do_export:
            if eic_mzml is None or eic_results_dir is None or eic_out_dir is None:
                print(
                    "[EIC export] skipped: missing eic_mzml/eic_results_dir/eic_out_dir "
                    "(set CLI args or config.yaml:eic_export)"
                )
                continue
            try:
                from lipidbench.eic.export import EICImageStyle, export_eic_images_from_results

                out_dir = (eic_out_dir / algo).resolve()
                style = EICImageStyle(
                    width_px=eic_width_px,
                    height_px=eic_height_px,
                    dpi=eic_dpi,
                    line_width=eic_line_width,
                    normalize_intensity=eic_normalize_intensity,
                    show_axes=eic_show_axes,
                    show_title=eic_show_title,
                    fixed_rt_window_min=eic_fixed_rt_window_min,
                )
                count = export_eic_images_from_results(
                    mzml_path=eic_mzml,
                    algo=algo,
                    results_dir=eic_results_dir,
                    out_dir=out_dir,
                    method=eic_method,
                    ppm=eic_ppm,
                    max_features=eic_max_features,
                    rt_pad_min=eic_rt_pad_min,
                    image_style=style,
                )
                print(f"[EIC export] {algo}: {count} images -> {out_dir}")
            except Exception as e:
                print(f"[EIC export] {algo}: failed: {e}")


if __name__ == "__main__":
    main()
