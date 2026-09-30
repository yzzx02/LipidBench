# ChromaPeak maintained model workflows

All current model workflows use 16 candidate attributes in this order:

`SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE`

## Joint peak detection and candidate validation

- `detection/run_rtx4070_multitask_concat_pipeline.py`: the released Main + LODO workflow, using naive concatenation.
- `detection/run_rtx4070_multitask_fusion_experiment.py`: shared-backbone trainer used by that workflow and fusion/task comparisons.
- `detection/evaluate_rtx4070_multitask_locked_target.py`: locked Main Test / heldout evaluation.
- `detection/evaluate_rtx4070_multitask_final.py`: inference and evaluation components.
- `detection/queue_detection_after_seed_lodo.py`: schedule joint training after the candidate-classification LODO workflow completes.
- `detection/smoke_test_peak_multitask.py`: synthetic 16-attribute forward/backward verification, without project data.

The shared batch helpers live in `lipidbench/data/training_utils.py`. Current training does not import another training entry point.

## Candidate-classification comparisons

- `convnext/run_rtx4070_fusion_experiment.py`: current 16-attribute classification trainer and fixed fusion comparisons.
- `convnext/run_rtx4070_lodo_pipeline.py`: the 11-fold candidate-classification LODO workflow.
- `convnext/evaluate_rtx4070_locked_main_test.py`: locked Main classification evaluation.
- `convnext/evaluate_rtx4070_lodo_heldout.py`: locked classification domain-holdout evaluation.

Reusable model, scaling and image-transform components live in `convnext/candidate_components.py`, which has no training CLI. These comparisons use ConvNeXt-Tiny and the four current modes: image-only, attribute-only, naive concatenation and gated fusion.

## Final dataset and release

- `data_prep/build_rtx4070_final_merged_dataset.py`
- `data_prep/verify_rtx4070_final_merged_dataset.py`
- `data_prep/build_rtx4070_detection_manifests.py`
- `data_prep/build_rtx4070_lodo_splits.py`
- `reporting/audit_rtx4070_ablation_sanity.py`
- `reporting/package_final_rtx4070_release.py`

The final dataset and saved checkpoints keep their current paths and fields. Imputation and standardization are fitted on the applicable Train partition only; Val selects models and thresholds before Test/heldout access.
