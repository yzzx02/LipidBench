import os
import subprocess

from utils.config_io import get_base_dir, _resolve_path


def extract_xcms_params(config):
    """Pull XCMS parameters from config with safe defaults."""
    xcms_params = config.get("parameters", {}).get("xcms", {})
    peak_picking = xcms_params.get("peak_picking", {})
    common_params = config.get("common_params", {})

    polarity = xcms_params.get("polarity", "positive")
    mz_tol = peak_picking.get("ppm", common_params.get("mz_tolerance_ppm", 10))
    peakwidth = peak_picking.get("peakwidth", [5, 50])
    if len(peakwidth) != 2:
        peakwidth = [5, 50]
    minwidth, maxwidth = peakwidth
    noise = peak_picking.get("noise", 1000)
    sn = peak_picking.get("snthresh", 3)
    prefilter = peak_picking.get("prefilter_val", 3)
    mzdiff = peak_picking.get("mzdiff", 0.001)

    return {
        "polarity": polarity,
        "mz_tol": mz_tol,
        "minwidth": minwidth,
        "maxwidth": maxwidth,
        "noise": noise,
        "mzdiff": mzdiff,
        "sn": sn,
        "prefilter": prefilter,
    }


def run_xcms(input_dir, output_file, polarity, mz_tol, minwidth, maxwidth, noise=1000, sn=3, prefilter=3, mzdiff=0.001):
    """Invoke the XCMS R script via subprocess."""
    r_script_path = os.path.join(os.path.dirname(__file__), "xcms.R")
    cmd = [
        "Rscript",
        str(r_script_path),
        "--dir",
        str(input_dir),
        "--output",
        str(output_file),
        "--polarity",
        str(polarity),
        "--mz_tol",
        str(mz_tol),
        "--minwidth",
        str(minwidth),
        "--maxwidth",
        str(maxwidth),
        "--noise",
        str(noise),
        "--sn",
        str(sn),
        "--prefilter",
        str(prefilter),
        "--mzdiff",
        str(mzdiff),
    ]

    print(f"Executing XCMS: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_xcms_pipeline(config):
    """Set up paths and parameters, then run XCMS."""
    base_dir = get_base_dir()
    input_dir = _resolve_path(base_dir, config["paths"]["input_dir"])
    output_dir = _resolve_path(base_dir, config["paths"]["xcms_output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "xcms_features.csv"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    params = extract_xcms_params(config)
    run_xcms(input_dir=input_dir, output_file=output_file, **params)


