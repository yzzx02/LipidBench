# PeakTruthLab on NVIDIA RTX

This profile targets an NVIDIA RTX 4070 SUPER on Windows 11 while keeping the
framework versions aligned with the validated AMD environment.

## Version contract

| Component | AMD workstation | NVIDIA workstation |
| --- | --- | --- |
| Python | 3.12 | 3.12 |
| PyTorch | 2.9.1 + ROCm 7.2.1 | 2.9.1 + CUDA 12.8 |
| torchvision | 0.24.1 + ROCm 7.2.1 | 0.24.1 + CUDA 12.8 |
| Input | 480 x 480 | 480 x 480 |
| Mode | `image_only` | `image_only` |
| AMP | FP16 | FP16 |

The model architecture, parameter names, manifests, random seed, loss
definitions, and checkpoints are shared. GPU kernels are backend-specific, so
ROCm and CUDA results should be statistically comparable but are not expected
to be bit-for-bit identical.

## Prepare the environment

Install the latest NVIDIA Studio or Game Ready driver that supports the
PyTorch CUDA 12.8 runtime. A separate system CUDA Toolkit is not required by
the PyTorch wheel.

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_nvidia_cu128.ps1
```

The script creates `.venv-nvidia-cu128`, installs the pinned dependencies, and
verifies that PyTorch detects the NVIDIA GPU.

## Obtain the dataset

Download the `PeakTruthLab-dataset-v1.zip` asset from:

<https://github.com/yzzx02/LipidBench/releases/tag/peaktruthlab-dataset-v1>

Extract it to a local directory. The extracted root contains:

```text
PeakTruthLab-dataset-v1/
  eic_images_flat/
  manifests/
    train.jsonl
    val.jsonl
    test.jsonl
  dataset_info.json
```

The packaged manifests use relative image paths and therefore do not depend on
the original `D:\LipidBench` location.

## Run the staged 2000/500 training

The test manifest is packaged for later final evaluation but is not read by
this command.

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\run_peak_image_only_nvidia.ps1 `
  -DatasetRoot "D:\datasets\PeakTruthLab-dataset-v1"
```

This runs:

- 2,000 training images and 500 validation images;
- physical batch size 8;
- 5 epochs;
- 480 x 480 input;
- FP16 AMP;
- RPN proposal limits 128 train / 64 validation;
- image-only detection and Seed validation;
- no 13-attribute fusion;
- no test-set evaluation.

Each epoch records the five training losses and validation metrics. Outputs
include `best_detection.pt`, `best_seed.pt`, `last.pt`, `history.json`, and
prediction overlays.
