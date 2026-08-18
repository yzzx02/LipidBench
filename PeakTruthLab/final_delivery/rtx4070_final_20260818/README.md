# RTX 4070 final delivery (2026-08-18)

This directory is the compact, Git-viewable index for the second and final PeakTruthLab experiment. Large files are distributed by the GitHub Release `rtx4070-final-20260818`.

## What is published

1. `LipidBench_Main_model_weights_20260818.zip`: Main `best_detection.pt`, Main `best_seed.pt`, preprocessing, configuration, selection record, and locked Main Test metrics.
2. `LipidBench_second_experiment_results_no_fold_weights_20260818.zip`: smoke test, four-mode ablation, Seed LODO, and joint detection+Seed Main/11-fold LODO results. It contains all CSV/JSON histories, threshold sweeps, per-sample predictions, PR/ROC data, audits, and logs, but no checkpoints.
3. `PeakTruthLab_final_dataset_20260814.zip`: 19,817 PNG/LabelMe pairs, raw 16-attribute Seed and peak-instance tables, locked splits, manifests, QC, provenance, and hashes.

The invalid batch-8/15-epoch joint run and the earlier ~15k-image experiment archive are not part of the final release.

## Core files in Git

- `experiment_results.csv`: Main Test, each of 11 held-out domains, and macro mean ± sample SD
- `experiment_results.json`: machine-readable equivalent
- `EXPERIMENT_AND_RESULTS.md`: protocol, selection rules, outcomes, and timing
- `DATA_QUALITY.md`: final dataset counts and leakage/QC checks
- `FIELDS.md`: master-table and result-field definitions
- `DOWNLOAD_ON_ANOTHER_PC.md`: download and verification instructions

The original 16-attribute tables and data audits are committed under `PeakTruthLab/datasets/PeakTruthLab_final_merged_20260814/tables` and `audits` for direct browser inspection.
