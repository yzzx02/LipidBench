# PeakTruthLab v2: manuscript-ready Methods, Data and Results

## Dataset and annotation quality

The finalized dataset contains 15,317 EIC images and 18,575 manually annotated true-peak boxes (1.213 boxes/image). Images with no peak, one peak and multiple peaks account for 25.59%, 44.25% and 30.16%, respectively.

Data were grouped by source file before splitting into Train (10,525 images; 12,402 boxes), Validation (2,507; 2,989) and Test (2,285; 3,184). No source file occurs in more than one split. The Test split was not used for training, checkpoint selection or threshold selection.

A model-assisted review queue of 150 Validation images was inspected manually in LabelMe. 66 annotations were changed; 52 boxes were added and 23 were removed (net change +29). Seed labels changed for 29 images (21 negative-to-positive and 8 positive-to-negative). Original files were preserved and the corrected labels were released as a separate v2 dataset.

For manually changed samples, Seed labels were recalculated from the overlap between the unchanged candidate Seed box and corrected true-peak boxes. A new match required two-dimensional IoU >=0.05. For an originally positive Seed, the prior linked target was retained when its overlap with a corrected box was >=0.30; otherwise the new IoU rule was applied. The 13 numeric attributes (SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT and JAG) describe the unchanged Seed candidate and were therefore preserved, not recomputed.

## Model and training

A shared ConvNeXt-Tiny feature pyramid supplied a Faster R-CNN true-peak detector and a candidate-specific Seed classifier. The detector used four FPN levels, anchors of 16/32/64/128 pixels, and tall-anchor aspect ratios of 1, 2, 4, 8 and 16. The Seed branch pooled a 7 x 7 RoI feature for the supplied candidate box. The image embedding was 256-D; the standardized 13-attribute vector was encoded to 64-D by a two-layer MLP with GELU activations and dropout 0.2.

Four joint-training conditions were compared: (i) image-only Seed classification, (ii) attribute-only Seed classification, (iii) direct concatenation of image and attribute embeddings, and (iv) gated fusion, where a learned sigmoid gate conditioned on both modalities modulated the image embedding before concatenation. The detection branch remained image-based in every condition; the ablation changes only the Seed head and its auxiliary gradients through the shared backbone.

All models used 480 x 480 inputs, batch size 8, automatic mixed precision (FP16), AdamW (learning rate 1e-4; weight decay 1e-4), a class-weighted binary cross-entropy Seed loss, and 10 epochs without data augmentation or a learning-rate scheduler. ImageNet-pretrained backbone weights and random seed 20260725 were used. Validation was performed after every epoch. The best detection checkpoint maximized Validation F1 at IoU=0.50; the best Seed checkpoint maximized Validation balanced accuracy.

## Final evaluation and statistics

After checkpoint selection, the detection confidence threshold was selected on Validation data by maximizing F1 over thresholds 0.05-0.95. The Seed probability threshold was selected on Validation data by maximizing balanced accuracy over thresholds 0.000-1.000. Test data were then evaluated once. Detection endpoints include precision, recall, F1, mean matched IoU, AP50, AP75, mAP50:95, boundary error and peak-count error. Seed endpoints include balanced accuracy, AUROC, average precision, F1, sensitivity, specificity, Brier score and 10-bin expected calibration error. Ninety-five percent confidence intervals were estimated with 1,000 nonparametric image-level bootstrap replicates.

## Core Test results

| Condition | Detection F1 (95% CI) | Mean IoU (95% CI) | AP50 | mAP50:95 | Seed BA (95% CI) | Seed AUROC (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| Image only | 0.8309 (0.8195 to 0.8419) | 0.8623 (0.8580 to 0.8662) | 0.8891 | 0.6216 | 0.9630 (0.9534 to 0.9720) | 0.9948 (0.9925 to 0.9967) |
| Attributes only (Seed head) | 0.8496 (0.8380 to 0.8611) | 0.8713 (0.8672 to 0.8750) | 0.8689 | 0.6258 | 0.9501 (0.9394 to 0.9608) | 0.9894 (0.9861 to 0.9925) |
| Naive concatenation | 0.8406 (0.8291 to 0.8511) | 0.8475 (0.8439 to 0.8514) | 0.9032 | 0.6065 | 0.9707 (0.9628 to 0.9779) | 0.9961 (0.9945 to 0.9975) |
| Gated fusion | 0.8432 (0.8318 to 0.8532) | 0.8528 (0.8487 to 0.8567) | 0.9045 | 0.6102 | 0.9603 (0.9503 to 0.9697) | 0.9942 (0.9914 to 0.9964) |

The highest Test detection F1 was obtained by Attributes only (Seed head) (0.8496). The highest Test Seed balanced accuracy was obtained by Naive concatenation (0.9707).

## Reproducibility and interpretation notes

- All four experiments are single-run, fixed-seed comparisons; bootstrap intervals quantify Test-sample uncertainty, not between-training-run variability.
- Model-assisted review was restricted to a manually inspected Validation subset; Test labels and Train labels were unchanged.
- `Attributes only` refers to the Seed classifier. The object detector always consumes EIC images.
- AP is reported from the detector's retained predictions (internal minimum score 0.05); operating-point precision/recall/F1 use the Validation-selected threshold.
