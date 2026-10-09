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
- Machine-local paper figures and reproduction materials: `<local-root>/artifacts/reproduction/`
- GitHub Release assets: final dataset, complete non-checkpoint results, and the two Main-model checkpoints

Only the Main split's `best_detection.pt` and `best_seed.pt` are published. The latter is the candidate-classification checkpoint under its historical filename. LODO/cross-domain checkpoints and `last.pt` files are deliberately excluded; their metrics, histories, thresholds, and predictions remain available.

## Model and terminology

- Backbone: ConvNeXt-Tiny C2-C5 + FPN P2-P5, computed once per batch.
- Detection: image-only Faster R-CNN, predicting zero, one, or multiple `True_Peak` boxes.
- Candidate validation: pool the original candidate's RoI from the shared FPN and combine it with its own attributes.
- Current released model: 480 x 480 EIC images, 16 attributes, and naive concatenation. Gated fusion remains an ablation option.
- Attribute imputation and standardization are fitted on Train only.

The public terminology is **Candidate (候选峰)**. Historical `seed_*` API keys and checkpoint filenames retain their original spelling for compatibility. `lipidbench/` remains the implementation package; `PeakTruthLab/` retains the maintained benchmark source and locked dataset metadata. Machine-local input/output paths come from configuration or explicit CLI arguments. Obsolete training entry points have been removed; see `PeakTruthLab/scripts/README_MODELS.md` for the current entry points.

The Main split preserves source proportions and keeps duplicate groups together. Cross-domain claims should use the LODO experiments; the Main split is not a complete mzML-disjoint evaluation. See [the architecture review](PeakTruthLab/docs/PROJECT_REVIEW_20260930.md) for implementation findings and scope.

## Install

```powershell
. .\scripts\use_local_workspace.ps1
python -m venv "$env:CHROMAPEAK_LOCAL_ROOT\environments\main"
& "$env:CHROMAPEAK_LOCAL_ROOT\environments\main\Scripts\Activate.ps1"
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
- `tests/`: unit and interface tests

The released Main-model checkpoints and canonical training dataset remain available
as GitHub Release assets. Paper figures, local evaluation tables, reproduction
packages, temporary work and caches are kept outside the source checkout and are
not committed or uploaded as new paper-figure Releases.

## Local storage

The default local root is a sibling of the checkout: `D:\LipidBench-local` for
`D:\LipidBench`. `CHROMAPEAK_LOCAL_ROOT` overrides it; `CHROMAPEAK_CACHE_ROOT`
can override just the cache location. `config.yaml` expands these two placeholders
before passing paths to the feature-extraction adapters.

- `artifacts/reproduction/outputs/`: saved experiments, figures and replot data
- `artifacts/reproduction/PeakTruthLab/final_delivery/`: local paper deliverables
- `artifacts/reproduction/data/`: machine-local input files
- `artifacts/reproduction/results/`: feature-extraction output
- `cache/`: Python bytecode, pip cache, temporary files, pytest cache and training locks
- `archives/`: preserved historical source snapshots

Dot-source `scripts/use_local_workspace.ps1` before commands to put Python bytecode,
pip and temporary files in the external cache. Existing Python/GPU environments
remain usable. Maintained source and locked canonical dataset metadata stay in Git;
the ignore rules exclude all generated figure formats and local result directories.

## RT validation and workstation transfer

MS2-guided missing-feature recovery is available through
`python -m chromapeak rescue-ms2 --help`. It accepts an identification table,
extracts MS1 EICs at precursor m/z and MS2 RT, detects and refines new candidates,
recomputes their attributes and validates them with the candidate checkpoint.
It exports deduplicated rescued peaks and counts; differential analysis is outside
its scope. See the [input schema and usage](PeakTruthLab/docs/MS2_GUIDED_RESCUE.md).

The local RT validation delivery contains inspectable summaries for 16 mzML files,
1,000 joint-baseline features and seven actual RT shifts. Its paper draft, figures
and paired analysis tables are local artifacts. The previously published RT
runtime/input archives omit paper figures and remain available for historical
workstation replay. The
[portable replay guide](PeakTruthLab/scripts/rt_validation/README.md) includes
RX 9070 XT preflight checks and comparison with the saved RTX4070 results.
This experiment retains its historical model and does not replace the current
16-attribute naive-concat default. Its measured result is similar stability for
both methods, with XCMS slightly better on the selected cohort.
