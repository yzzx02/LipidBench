# ChromaPeak

**ChromaPeak: joint chromatographic peak detection and candidate validation for LC-MS.**

ChromaPeak combines EIC images with candidate-specific chromatographic attributes. One shared ConvNeXt-Tiny + FPN feeds two tasks: full-window peak detection and validation of an upstream algorithm's original candidate. The project focuses on three tasks:

1. feature and EIC extraction from mzML files;
2. chromatographic peak detection and true/false candidate classification;
3. peak-boundary quantification and result export.

The current paper dataset and model are the RTX 4070 final release (2026-08-18). The model uses exactly 16 attributes; training entry points are limited to the maintained RTX workflows.

## Current final assets

- Final merged dataset: `PeakTruthLab/datasets/PeakTruthLab_final_merged_20260814`
- Reproducible protocol: `PeakTruthLab/docs/RTX4070_MERGE_AND_SPLIT_PROMPT.md`
- Compact paper-ready delivery: `PeakTruthLab/final_delivery/rtx4070_final_20260818`
- GitHub Release assets: final dataset, complete non-checkpoint results, and the two Main-model checkpoints

Only the Main split's `best_detection.pt` and `best_seed.pt` are published. The latter is the candidate-classification checkpoint under its historical filename. LODO/cross-domain checkpoints and `last.pt` files are deliberately excluded; their metrics, histories, thresholds, and predictions remain available.

## Model and terminology

- Backbone: ConvNeXt-Tiny C2-C5 + FPN P2-P5, computed once per batch.
- Detection: image-only Faster R-CNN, predicting zero, one, or multiple `True_Peak` boxes.
- Candidate validation: pool the original candidate's RoI from the shared FPN and combine it with its own attributes.
- Current released model: 480 x 480 EIC images, 16 attributes, and naive concatenation. Gated fusion remains an ablation option.
- Attribute imputation and standardization are fitted on Train only.

The public terminology is **Candidate (候选峰)**. Historical `seed_*` API keys and checkpoint filenames retain their original spelling for compatibility. `lipidbench/` remains the implementation package; `PeakTruthLab/` remains the annotated dataset and experiment workspace. Data paths, manifests, saved weights, and `python main.py` commands remain usable. Obsolete training entry points have been removed; see `PeakTruthLab/scripts/README_MODELS.md` for the current entry points.

The Main split preserves source proportions and keeps duplicate groups together. Cross-domain claims should use the LODO experiments; the Main split is not a complete mzML-disjoint evaluation. See [the architecture review](PeakTruthLab/docs/PROJECT_REVIEW_20260930.md) for implementation findings and scope.

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
python -m chromapeak --algo pyopenms
python -m chromapeak --algo xcms,asari
```

Export EIC images while running a feature detector:

```powershell
python -m chromapeak --algo pyopenms --export-eic --eic-mzml D:\data\sample.mzML
```

Verify the final dataset:

```powershell
python PeakTruthLab/scripts/data_prep/verify_rtx4070_final_merged_dataset.py --help
```

Build the three release archives after the formal experiment has completed:

```powershell
python PeakTruthLab/scripts/reporting/package_final_rtx4070_release.py
```

Training and locked evaluation entry points are documented in `PeakTruthLab/README.md`. All attribute imputation and standardization must be fit on Train only; Val selects checkpoints and thresholds; Test/heldout data are evaluated only after everything is locked. Maintained RTX workflows infer the checkout root from their script location. `CHROMAPEAK_PROJECT_ROOT` can override it; the legacy `LIPIDBENCH_PROJECT_ROOT` remains supported.

## Repository layout

- `chromapeak/`: project CLI entry point (`python -m chromapeak`)
- `lipidbench/runners/`: XCMS, pyOpenMS, MS-DIAL, and Asari adapters
- `lipidbench/eic/`: EIC extraction and export
- `lipidbench/models/`: peak detection, candidate classification, and fusion models
- `lipidbench/utils/`: peak attributes, boundary refinement, I/O, and alignment
- `PeakTruthLab/scripts/data_prep/`: dataset construction and QC
- `PeakTruthLab/scripts/convnext/`: candidate classification and ablation workflows
- `PeakTruthLab/scripts/detection/`: joint detection + candidate training/evaluation
- `PeakTruthLab/final_delivery/`: compact paper-ready results and field documentation
- `tests/`: unit and interface tests

Large mzML files, images, model weights, and full experiment outputs are distributed as GitHub Release assets rather than committed to Git history.
