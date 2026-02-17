import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EIC plots without GUI")
    p.add_argument("--mzml", required=True, help="Path to a single .mzML file")
    p.add_argument("--algo", required=True, choices=["asari", "pyopenms", "xcms", "msdial"], help="Which algorithm's feature table to use")
    p.add_argument("--results-dir", required=True, help="Results directory containing xcms/pyopenms/asari/msdial subfolders")
    p.add_argument("--ppm", type=float, default=10.0, help="m/z tolerance in ppm for nearest-point EIC")
    p.add_argument("--out-dir", required=True, help="Output directory for EIC images")
    p.add_argument("--max-features", type=int, default=200, help="Limit number of features exported")
    p.add_argument("--rt-pad-min", type=float, default=0.2, help="RT padding (minutes) around RTmin/RTmax")
    p.add_argument("--method", choices=["nearest", "window_sum"], default="window_sum", help="EIC extraction method")
    p.add_argument("--width-px", type=int, default=400)
    p.add_argument("--height-px", type=int, default=300)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--line-width", type=float, default=1.0)
    p.add_argument("--normalize-intensity", choices=["true", "false"], default="true")
    p.add_argument("--show-axes", choices=["true", "false"], default="true")
    p.add_argument("--show-title", choices=["true", "false"], default="false")
    p.add_argument("--fixed-rt-window-min", type=float, default=2.0, help="Fixed RT window size in minutes; <=0 to disable")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mzml_path = Path(args.mzml).resolve()
    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not mzml_path.exists():
        raise FileNotFoundError(mzml_path)
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)

    from lipidbench.eic.export import EICImageStyle, export_eic_images_from_results

    style = EICImageStyle(
        width_px=int(args.width_px),
        height_px=int(args.height_px),
        dpi=int(args.dpi),
        line_width=float(args.line_width),
        normalize_intensity=str(args.normalize_intensity).lower() == "true",
        show_axes=str(args.show_axes).lower() == "true",
        show_title=str(args.show_title).lower() == "true",
        fixed_rt_window_min=(float(args.fixed_rt_window_min) if float(args.fixed_rt_window_min) > 0 else None),
    )

    count = export_eic_images_from_results(
        mzml_path=mzml_path,
        algo=args.algo,
        results_dir=results_dir,
        out_dir=out_dir,
        method=args.method,
        ppm=float(args.ppm),
        max_features=int(args.max_features),
        rt_pad_min=float(args.rt_pad_min),
        image_style=style,
    )

    print(f"Exported {count} EIC images to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
