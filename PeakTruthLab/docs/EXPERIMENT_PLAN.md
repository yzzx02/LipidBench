# ChromaPeak current experiment protocol

The maintained experiments use the final dataset at `PeakTruthLab/datasets/PeakTruthLab_final_merged_20260814` and exactly 16 candidate attributes:

`SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE`

## Tasks and data

Each manifest record describes one EIC window, zero or more true-peak boxes, one original candidate box, its independent binary label, its 16 attributes, and source/domain metadata. Missing attributes retain their missing state until train-fitted preprocessing.

Candidate attributes belong only to the original candidate. The detection branch uses image features; the candidate-validation branch uses its RoI features and attributes. A false candidate may share its window with true peaks elsewhere.

## Current model and comparisons

The released model uses ConvNeXt-Tiny C2-C5 + FPN P2-P5, Faster R-CNN detection and naive concatenation for candidate validation. The four current fusion comparisons are image-only, attribute-only, naive concatenation and gated fusion, all on the same 16-attribute dataset. The trainer also supports task-loss comparisons between joint training, detection and candidate validation.

Whole-image classification comparisons use the maintained ConvNeXt-Tiny classification workflow and are distinct from the joint FPN/RoI model. Training and evaluation entry points are listed in `../scripts/README_MODELS.md`.

## Fixed training settings

- 480 x 480 input; batch size 16; at most 30 epochs.
- FP16 AMP; AdamW; learning rate 1e-4; weight decay 1e-4.
- Random seed 20260814; no augmentation.
- Attribute imputation and scaling fitted on the corresponding Train partition only.
- Val selects checkpoints and thresholds before locked Test/heldout access.

Main uses the locked 15,853 / 1,982 / 1,982 split, preserving source proportions and duplicate groups. It is not a complete mzML-disjoint split. The 11-fold LODO protocol evaluates domain holdout separately.

## Evaluation

Report candidate AUROC, average precision, F1, balanced accuracy and calibration; detection AP50, AP75, mAP@0.50:0.95, precision, recall, matched IoU, boundary errors and peak-count errors. Summarize domain and difficult-subset results using the locked evaluation outputs.

The network returns image-coordinate boxes and candidate probabilities. EIC extraction, RT-boundary refinement and signal integration have separate tools; automated per-detection RT mapping, attribute recomputation and downstream quantitative validation remain a separate integration stage.

The ordered final data/experiment protocol is in `RTX4070_MERGE_AND_SPLIT_PROMPT.md`; model details are in `PEAK_MULTITASK_ARCHITECTURE.md`.
