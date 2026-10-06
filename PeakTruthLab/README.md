# ChromaPeak annotated data and experiments (PeakTruthLab)

PeakTruthLab is the annotated EIC benchmark and experiment workspace within ChromaPeak. The maintained workflow covers feature extraction, 16-attribute calculation, joint peak detection and candidate classification, and peak-boundary quantification. Candidate (候选峰) is the public term; historical `seed_*` field names remain compatible with saved data and checkpoints.

## Final dataset

`datasets/PeakTruthLab_final_merged_20260814` contains 19,817 EIC images with same-name LabelMe JSON files, 19,817 candidate rows, and 18,915 annotated peak instances. The locked Main split is 15,853 Train / 1,982 Val / 1,982 Test.

The 16 raw attributes are:

`SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE`

Raw missing states are preserved. Every experiment fits imputation and standardization only on the corresponding Train partition.

## Final training protocol

- Input: 480 x 480
- Batch size: 16
- Maximum epochs: 30
- Precision: FP16
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Random seed: 20260814
- Augmentation: none
- Checkpoint and detection/candidate thresholds: selected on Val
- Test/heldout: evaluated once after locking

The paper model is Naive concat with image features plus all 16 attributes. The published general-purpose weights are only `best_detection.pt` and `best_seed.pt` from the Main split. The 11 LODO folds are retained as result tables but their checkpoints are not published.

## Maintained entry points

- Build merged dataset: `scripts/data_prep/build_rtx4070_final_merged_dataset.py`
- Verify dataset: `scripts/data_prep/verify_rtx4070_final_merged_dataset.py`
- Build detection manifests: `scripts/data_prep/build_rtx4070_detection_manifests.py`
- Build LODO splits: `scripts/data_prep/build_rtx4070_lodo_splits.py`
- Candidate ablation/LODO: `scripts/convnext/run_rtx4070_fusion_experiment.py`, `scripts/convnext/run_rtx4070_lodo_pipeline.py`
- Joint detection + candidate validation: `scripts/detection/run_rtx4070_multitask_concat_pipeline.py`
- Locked target evaluation: `scripts/detection/evaluate_rtx4070_multitask_locked_target.py`
- Package final release: `scripts/reporting/package_final_rtx4070_release.py`

The exact ordered protocol is in `docs/RTX4070_MERGE_AND_SPLIT_PROMPT.md`. Compact results, data-quality statistics, and cross-PC instructions are in `final_delivery/rtx4070_final_20260818`.
