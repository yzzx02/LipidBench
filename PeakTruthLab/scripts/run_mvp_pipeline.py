from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_step(step_no: int, total: int, title: str, cmd: list[str]) -> None:
    print(f"\n[STEP {step_no}/{total}] {title}")
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({title}), return code={result.returncode}")


def main() -> int:
    root = Path(__file__).resolve().parent

    split_script = root / "train_val_split.py"
    train_script = root / "train_model.py"
    eval_script = root / "eval_mvp.py"

    data_root = Path(r"D:\LipidBench\PeakTruthLab\datasets\small_trainset")
    split_root = Path(r"D:\LipidBench\PeakTruthLab\datasets\split")
    val_root = split_root / "val"
    weights_path = Path(r"D:\LipidBench\PeakTruthLab\models\fasterrcnn_mvp_core3.pth")
    outputs_root = Path(r"D:\LipidBench\PeakTruthLab\outputs_core3")
    label_map = Path(r"D:\LipidBench\PeakTruthLab\configs\label_map.json")

    py = sys.executable

    # STEP 1: split
    _run_step(
        1,
        3,
        "Splitting datasets (8:2)",
        [
            py,
            str(split_script),
            "--source-root",
            str(data_root),
            "--out-root",
            str(split_root),
            "--val-ratio",
            "0.2",
            "--seed",
            "42",
        ],
    )

    # STEP 2: train
    _run_step(
        2,
        3,
        "Training Faster R-CNN MVP",
        [
            py,
            str(train_script),
            "--data-root",
            str(split_root / "train"),
            "--label-map",
            str(label_map),
            "--epochs",
            "20",
            "--batch-size",
            "4",
            "--lr",
            "1e-4",
            "--optimizer",
            "sgd",
            "--save-path",
            str(weights_path),
        ],
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Training finished but weight file not found: {weights_path}. Stop before eval."
        )

    # STEP 3: eval visualization
    _run_step(
        3,
        3,
        "Running inference visualization on val set",
        [
            py,
            str(eval_script),
            "--image-root",
            str(val_root),
            "--label-map",
            str(label_map),
            "--weights",
            str(weights_path),
            "--out-root",
            str(outputs_root),
            "--score-threshold",
            "0.2",
        ],
    )

    print("\nPipeline completed successfully.")
    print(f"Weights: {weights_path}")
    print(f"Visual outputs: {outputs_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
