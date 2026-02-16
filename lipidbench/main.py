import argparse
import importlib

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
    return parser.parse_args()


def main():
    config = load_config()
    args = parse_args()
    algo_list = [a.strip().lower() for a in args.algo.split(",") if a.strip()]
    for algo in algo_list:
        runner = _get_runner(algo)
        runner(config)


if __name__ == "__main__":
    main()
