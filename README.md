# LipidBench / PeakTruthLab

LipidBench is a reproducible LC-MS workflow focused on three tasks:

1. feature and EIC extraction from mzML files;
2. chromatographic peak detection and true/false Seed classification;
3. peak-boundary quantification and result export.

The current paper dataset and model are the RTX 4070 final release (2026-08-18). The earlier ~15k-image experiment archive is intentionally retired.

## Current final assets

- Final merged dataset: `PeakTruthLab/datasets/PeakTruthLab_final_merged_20260814`
- Reproducible protocol: `PeakTruthLab/docs/RTX4070_MERGE_AND_SPLIT_PROMPT.md`
- Compact paper-ready delivery: `PeakTruthLab/final_delivery/rtx4070_final_20260818`
- GitHub Release assets: final dataset, complete non-checkpoint results, and the two Main-model checkpoints

Only the Main split's `best_detection.pt` and `best_seed.pt` are published. LODO/cross-domain checkpoints and `last.pt` files are deliberately excluded; their metrics, histories, thresholds, and predictions remain available.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA CUDA 12.8 training:

```powershell
pip install -r requirements-peak-nvidia-cu128.txt
```

## Core commands

Run feature extraction algorithms configured in `config.yaml`:

```powershell
python main.py --algo pyopenms
python main.py --algo xcms,asari
```

Export EIC images while running a feature detector:

```powershell
python main.py --algo pyopenms --export-eic --eic-mzml D:\data\sample.mzML
```

Verify the final dataset:

```powershell
python PeakTruthLab/scripts/data_prep/verify_rtx4070_final_merged_dataset.py --help
```

Build the three release archives after the formal experiment has completed:

```powershell
python PeakTruthLab/scripts/reporting/package_final_rtx4070_release.py
```

Training and locked evaluation entry points are documented in `PeakTruthLab/README.md`. All attribute imputation and standardization must be fit on Train only; Val selects checkpoints and thresholds; Test/heldout data are evaluated only after everything is locked.

## Repository layout

- `lipidbench/runners/`: XCMS, pyOpenMS, MS-DIAL, and Asari adapters
- `lipidbench/eic/`: EIC extraction and export
- `lipidbench/models/`: peak detection, Seed classification, and fusion models
- `lipidbench/utils/`: peak attributes, boundary refinement, I/O, and alignment
- `PeakTruthLab/scripts/data_prep/`: dataset construction and QC
- `PeakTruthLab/scripts/convnext/`: Seed classification and ablation workflows
- `PeakTruthLab/scripts/detection/`: joint detection + Seed training/evaluation
- `PeakTruthLab/final_delivery/`: compact paper-ready results and field documentation
- `tests/`: unit and interface tests

Large mzML files, images, model weights, and full experiment outputs are distributed as GitHub Release assets rather than committed to Git history.
