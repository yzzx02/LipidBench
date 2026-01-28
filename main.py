import sys
from pathlib import Path
import argparse

# Ensure src is on path when running from project root
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config_io import load_config
from runners.run_xcms import run_xcms_pipeline
from runners.run_msdial import run_msdial_pipeline

def run_pyopenms_pipeline(_config):
    raise NotImplementedError("pyOpenMS pipeline not implemented yet")


def run_asari_pipeline(_config):
    raise NotImplementedError("Asari pipeline not implemented yet")


ALGORITHM_REGISTRY = {
    "xcms": run_xcms_pipeline,
    "ms-dial": run_msdial_pipeline,
    "pyopenms": run_pyopenms_pipeline,
    "asari": run_asari_pipeline,
}


def parse_args():
    parser = argparse.ArgumentParser(description="LipidBench algorithm runner")
    parser.add_argument(
        "--algo",
        default="ms-dial",
        help="Algorithm(s) to run, comma-separated (e.g., xcms,ms-dial)",
    )
    return parser.parse_args()


def main():
    config = load_config()
    args = parse_args()

    algo_list = [a.strip().lower() for a in args.algo.split(",") if a.strip()]
    for algo in algo_list:
        runner = ALGORITHM_REGISTRY.get(algo)
        if runner is None:
            raise ValueError(f"Unknown algorithm: {algo}")
        runner(config)


if __name__ == "__main__":
    main()
