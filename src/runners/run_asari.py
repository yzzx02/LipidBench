from pathlib import Path
import subprocess
from utils.data_io import load_asari_results
def extract_asari_params(config):
    asari_params = config.get("parameters", {}).get("asari", {})
    ppm = asari_params.get("ppm", 10)
    autoheight = asari_params.get('autoheight',False)
    mode = asari_params.get("mode", "pos")
    min_intensity_threshold = asari_params.get("min_intensity_threshold", 1000)
    return {
        'ppm': ppm,
        'autoheight': autoheight,
        'mode': mode,
        'min_intensity_threshold': min_intensity_threshold,
    }
def run_asari_pipeline(config):
    input_dir = config.get('paths',{}).get('input_dir','')
    output_dir = config.get('paths',{}).get('asari_output','')
    input_dir = Path(input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir = Path(output_dir).resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True,exist_ok=True)
    file_count = len(list(input_dir.glob("*.mzML")))
    params = extract_asari_params(config)
    # if file_count == 1:
    #     # 控制台输出  asari analyze --input <input_file> --output <output_dir> --ppm <ppm> --autoheight <autoheight> --mode <mode> --min_intensity_threshold <min_intensity_threshold>
    #     input_file = list(input_dir.glob("*.mzML"))[0]
    #     cmd = [
    #         "asari",
    #         "analyze",
    #         "--input",
    #         str(input_file),
    #         "--ppm",
    #         str(int(params['ppm'])),
    #         "--autoheight",
    #         str(params['autoheight']).lower(),
    #         "--mode",
    #         str(params['mode']),
    #         "--min_intensity_threshold",
    #         str(params['min_intensity_threshold']),
    #     ]
    #     print(f"Executing Asari: {' '.join(cmd)}")
    #     subprocess.run(cmd, check=True)
    # else:
        #控制台输出 asari process --input <input_dir> --output <output_dir> --ppm <ppm> --autoheight <autoheight> --mode <mode> --min_intensity_threshold <min_intensity_threshold>
    # Asari will create a project directory by appending `_asari_project_<id>` to the
    # provided output path. Using a prefix under output_dir keeps results inside it.
    output_prefix = output_dir / "asari"

    cmd = [
        "asari",
        "process",
        "-i",
        str(input_dir),
        "-o",
        str(output_prefix),
        "--ppm",
        str(int(params['ppm'])),
        "--autoheight",
        str(params['autoheight']),
        "--mode",
        str(params['mode']),
        "--min_intensity_threshold",
        str(params['min_intensity_threshold']),
    ]
    print(f"Executing Asari: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Locate the newest project directory and post-process outputs.
    candidates = list(output_dir.glob(f"{output_prefix.name}_asari_project_*"))
    if not candidates:
        # Backward-compatibility: older runs may have used output_dir directly.
        candidates = list(output_dir.parent.glob(f"{output_dir.name}_asari_project_*"))
    if candidates:
        project_dir = max(candidates, key=lambda p: p.stat().st_mtime)
        load_asari_results(project_dir, output_dir=output_dir, cleanup_project=True)