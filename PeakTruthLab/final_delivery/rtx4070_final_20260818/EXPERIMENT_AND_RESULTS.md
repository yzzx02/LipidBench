# Experiment protocol and final results

## Locked protocol

The final experiment used 480 x 480 input, batch size 16, FP16, AdamW, learning rate 1e-4, weight decay 1e-4, at most 30 epochs, random seed 20260814, and no data augmentation. All attribute imputation and standardization parameters were fit using Train only. Val selected the detection and Seed checkpoints and both decision thresholds. Main Test and each held-out LODO domain were accessed only after selection was locked.

The model is Naive concat: image features are concatenated with 16 raw peak attributes after Train-fitted preprocessing. The outputs are joint peak detection/boundary regression and Seed true/false classification.

## Main Test

| Metric | Value |
|---|---:|
| Detection Precision | 0.9140 |
| Detection Recall | 0.8726 |
| Detection F1 | 0.8928 |
| Matched mean IoU | 0.8825 |
| AP50 | 0.8823 |
| AP75 | 0.7448 |
| mAP50:95 | 0.6553 |
| Left-boundary MAE | 2.836 px |
| Right-boundary MAE | 4.694 px |
| Peak-count MAE | 0.1544 |
| Seed balanced accuracy | 0.9605 |
| Seed AUROC | 0.9885 |
| Seed average precision | 0.9855 |
| Seed F1 | 0.9660 |

The Val-selected best detection checkpoint was epoch 27; the best Seed checkpoint was epoch 16. Main training required 10,938 s (3.04 h) and peaked at 6,425,288,704 bytes (5.98 GiB) allocated GPU memory.

## LODO generalization

Across all 11 independently held-out domains, the macro mean ± sample SD was:

| Metric | Mean ± SD |
|---|---:|
| Detection Precision | 0.8448 ± 0.1042 |
| Detection Recall | 0.8374 ± 0.0802 |
| Detection F1 | 0.8352 ± 0.0692 |
| Matched mean IoU | 0.8629 ± 0.0153 |
| AP50 | 0.8431 ± 0.0692 |
| AP75 | 0.6644 ± 0.0766 |
| mAP50:95 | 0.5908 ± 0.0632 |
| Left-boundary MAE | 3.342 ± 0.964 px |
| Right-boundary MAE | 5.468 ± 1.550 px |
| Peak-count MAE | 0.2423 ± 0.1921 |
| Seed balanced accuracy | 0.9429 ± 0.0366 |
| Seed AUROC | 0.9831 ± 0.0168 |
| Seed average precision | 0.9780 ± 0.0264 |
| Seed F1 | 0.9490 ± 0.0392 |

Per-domain raw values are in `experiment_results.csv`. LODO checkpoint files are intentionally excluded from publication; the locked configuration, threshold choices, histories, predictions, and metrics are retained.

## Interpretation boundary

The Main Test estimates performance on a sample-level held-out subset of the merged benchmark. The 11-domain LODO analysis tests transfer to the independent studies covered by this benchmark. It should not be interpreted as universal validation for every future LC-MS platform or acquisition protocol.
