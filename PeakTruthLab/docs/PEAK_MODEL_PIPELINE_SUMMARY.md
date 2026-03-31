# PeakTruthLab Peak Detection / Peak Quality Classification Pipeline Summary

## 1. Goal
- Build a binary peak-quality model that decides whether a candidate feature is a true usable chromatographic peak.
- Input is multimodal:
  - EIC image
  - 13 peak-shape / peak-quality attributes
- Current final task is binary classification with label column `is_true_peak`.

## 2. Final Active Dataset
- Final image root: `PeakTruthLab/datasets/eic_images_flat/`
- Final feature table: `PeakTruthLab/datasets/feature_table_final_10000.csv`
- Each sample is keyed by `Feature_ID`
- Each valid sample must have:
  - one CSV row
  - one PNG
  - one LabelMe JSON

## 3. Upstream Data Construction
1. Start from mzML files and algorithm-generated candidate features.
2. Build a feature pool with core columns:
   - `source_file`
   - `source_path`
   - `Feature_ID`
   - `mz`
   - `RTmin`
   - `RT`
   - `RTmax`
3. For each feature, extract EIC trace around target `mz`.
4. Render a fixed-window EIC image.
5. Save a same-name LabelMe JSON for later manual annotation and RT-boundary correction.
6. Compute 13 literature-derived peak attributes:
   - `SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG`
7. Merge image-linked metadata and attributes into the final CSV.

## 4. Annotation and Data Cleaning Flow
1. Human annotates each EIC image in LabelMe.
2. Positive sample:
   - keep a rectangle around the true peak
   - label is typically `True_Peak`
3. Negative sample:
   - JSON `shapes` can be empty
4. If RT boundary is wrong but a valid rectangle exists:
   - map rectangle pixel X range back to RT range
   - recompute `RTmin/RTmax`
   - optionally update `RT`
   - recompute the 13 peak attributes
   - sync corrected RT boundary back into JSON
5. Any sample with unresolved invalid attributes or broken image/CSV pairing is removed from the final training set.

## 5. Current Model Architecture
### Input branch A: image branch
- Backbone: `ConvNeXt-Tiny`
- Optional enhanced version: `ConvNeXt-Tiny + LWGA`
- Image preprocessing:
  - train: random resized crop, small translation, brightness/contrast jitter, optional Gaussian blur
  - eval: resize only
  - normalize with ImageNet mean/std

### Input branch B: attribute branch
- 13 standardized attributes
- Missing values are median-filled from train split
- Attribute encoder:
  - Linear(attr_dim -> 64)
  - GELU
  - Dropout
  - Linear(64 -> 64)
  - GELU

### Fusion
1. Extract image embedding from ConvNeXt.
2. Extract 64-d attribute embedding from MLP.
3. Concatenate `[image_feat, attr_feat]`.
4. Pass through a gating MLP to generate an image-feature gate.
5. Apply gate to image embedding.
6. Concatenate `[gated_image_feat, attr_feat]`.
7. Feed into classifier head.

### Output
- Binary mode:
  - one logit
  - loss: `BCEWithLogitsLoss`
- Metrics:
  - AUC
  - PR-AUC
  - F1
  - Accuracy

## 6. Training Flow
1. Prepare `train.csv` and `val.csv`.
2. Ensure each row contains:
   - image relative path
   - 13 attributes
   - `is_true_peak`
3. Build attribute scaler only from train split:
   - median fill
   - z-score standardization
4. Build train/val datasets and dataloaders.
5. Train fusion model using AdamW.
6. Evaluate every epoch on validation set.
7. Save:
   - `best_model.pth`
   - `history.json`
   - `attr_scaler.json`

## 7. Inference Flow
1. Given a candidate feature with `mz`, `RT`, `RTmin`, `RTmax`
2. Extract/render EIC image
3. Compute 13 attributes
4. Apply saved attribute scaler
5. Run image branch + attribute branch + fusion head
6. Output probability of `true peak`
7. Threshold probability to accept/reject candidate peak

## 8. One-Line Flowchart Version
`mzML -> candidate features -> EIC extraction -> PNG + LabelMe JSON -> RT boundary correction / attribute recomputation -> final CSV + images -> train/val split -> ConvNeXt image branch + MLP attribute branch -> gated fusion -> binary true-peak score`

## 9. Important Current Status
- Active final dataset is now fully synchronized:
  - `csv_count == png_count == json_count`
- Samples with all-NaN peak attributes have been removed from the active final dataset.
- `is_true_peak` labels are still not filled in the final CSV, so supervised model training is not yet ready until annotation merge is completed.

## 10. Files Most Relevant To The Pipeline
- `PeakTruthLab/docs/CONVNEXT_FUSION_EXECUTION_PLAN.md`
- `PeakTruthLab/scripts/convnext/train_convnext_fusion.py`
- `PeakTruthLab/scripts/annotation/annotate_is_true_peak.py`
- `PeakTruthLab/scripts/annotation/recompute_rt_bounds_from_labelme.py`
- `PeakTruthLab/datasets/feature_table_final_10000.csv`
- `PeakTruthLab/datasets/eic_images_flat/`
