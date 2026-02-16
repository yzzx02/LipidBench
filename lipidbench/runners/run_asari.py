from pathlib import Path
import subprocess

from lipidbench.utils.data_io import load_asari_results


def extract_asari_params(config):
    asari_params = config.get("parameters", {}).get("asari", {})
    ppm = asari_params.get("ppm", 10)
    autoheight = asari_params.get("autoheight", False)
    mode = asari_params.get("mode", "pos")
    min_intensity_threshold = asari_params.get("min_intensity_threshold", 1000)
    min_peak_height = asari_params.get("min_peak_height", 5000)
    return {
        "ppm": ppm,
        "autoheight": autoheight,
        "mode": mode,
        "min_intensity_threshold": min_intensity_threshold,
        "min_peak_height": min_peak_height,
    }


def run_asari_pipeline(config):
    input_dir = Path(config.get("paths", {}).get("input_dir", "")).resolve()
    output_dir = Path(config.get("paths", {}).get("asari_output", "")).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    params = extract_asari_params(config)
    output_prefix = output_dir / "asari"
    cmd = [
        "asari", "process",
        "-i", str(input_dir),
        "-o", str(output_prefix),
        "--ppm", str(int(params["ppm"])),
        "--autoheight", str(params["autoheight"]),
        "--mode", str(params["mode"]),
        "--min_intensity_threshold", str(params["min_intensity_threshold"]),
        "--min_peak_height", str(params["min_peak_height"]),
    ]
    print(f"Executing Asari: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    candidates = list(output_dir.glob(f"{output_prefix.name}_asari_project_*"))
    if not candidates:
        candidates = list(output_dir.parent.glob(f"{output_dir.name}_asari_project_*"))
    if candidates:
        project_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        load_asari_results(project_dir, output_dir=output_dir, cleanup_project=True)
