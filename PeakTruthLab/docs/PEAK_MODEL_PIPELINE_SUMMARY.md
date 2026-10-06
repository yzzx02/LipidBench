# ChromaPeak model pipeline

The maintained model performs joint chromatographic peak detection and candidate validation for LC-MS. Candidate (候选峰) is the public term; historical `seed_*` keys remain compatible with existing files and weights.

## Current data and released configuration

- Dataset: `PeakTruthLab/datasets/PeakTruthLab_final_merged_20260814`
- 19,817 EIC windows and candidate rows; 18,915 annotated peak instances.
- Main split: 15,853 Train / 1,982 Val / 1,982 Test.
- Image input: 480 x 480.
- Attributes: `SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE`.
- Raw missing values remain identifiable; fill and scaling statistics are fitted only on Train.
- Released joint model: ConvNeXt-Tiny + FPN + Faster R-CNN, with naive concatenation in the candidate head.
- Image-only, attribute-only and gated-fusion variants remain available for comparisons.

## Upstream and model flow

1. Read mzML and obtain candidate features and initial bounds from an upstream algorithm.
2. Extract the EIC and produce an image with matching candidate coordinates.
3. Compute attributes only inside that original candidate's bounds.
4. Build manifests with detection boxes and an independently assigned candidate label. A false candidate can share its window with a true peak elsewhere.
5. Compute ConvNeXt C2-C5 and FPN P2-P5 once for both model tasks.
6. Detect zero, one or multiple true peaks from full-window image features.
7. Pool the original candidate RoI from the shared FPN, encode its attributes, and output its true-peak probability.

Candidate attributes are not broadcast to RPN proposals or unrelated detections.

## Training and evaluation

Maintained joint training runs through `scripts/detection/run_rtx4070_multitask_concat_pipeline.py`. The released protocol uses batch size 16, up to 30 epochs, FP16, AdamW, learning rate 1e-4, weight decay 1e-4, and no augmentation. Val selects checkpoints and thresholds before the locked Test/heldout evaluation.

The detector and candidate classifier have separate selected checkpoints: `best_detection.pt` and the historical filename `best_seed.pt`. Whole-image classification comparisons under `scripts/convnext/` use a different model path.

Main preserves source proportions and keeps duplicate groups together; it is not a complete mzML-disjoint split. LODO evaluates domain holdout separately.

## Implemented tools and integration boundary

Manifest loading, Dataset/collation, attribute preprocessing, real training, locked evaluation, EIC extraction, RT-boundary refinement, and signal integration utilities are implemented. The model returns image-coordinate boxes and candidate probabilities. A unified automated second stage that maps every predicted box to RT and recomputes its attributes and area is still a separate integration task.

For model details see `PEAK_MULTITASK_ARCHITECTURE.md`; for the current review and fixes see `PROJECT_REVIEW_20260930.md`. Earlier 10,000-image binary-only descriptions are superseded by this flow.
