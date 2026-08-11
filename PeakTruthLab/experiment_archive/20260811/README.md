# LipidBench experiment result archive (2026-08-11)

This directory consolidates the numerical outputs and provenance from the LipidBench / PeakTruthLab experiments completed in this working session. It is intentionally optimized for paper tables, plots, auditing and downstream re-analysis.

## Contents

- `00_protocol_and_splits/`: canonical source-domain metadata and the exact archive-building script.
- `01_pilot_and_image_only/`: the 2,000/500 pilot and the full 10-epoch image-only baseline (config, preprocessing, epoch history and summary).
- `02_mixed_domain_ablation/`: reviewed-data quality report, single-seed ablation, epoch-15 probe, four-model three-seed stability experiment, threshold locks, compact per-image Test outputs and paper-ready tables.
- `03_cross_domain/`: frozen Train/Val/Seen/External A/B/C split tables, per-seed training histories, validation threshold selection, per-domain raw numerical evaluation and final three-seed summary.
- `MANIFEST.csv`: SHA-256 and original source path for every archived file.
- `CHECKPOINT_INDEX.csv`: local location and size of every excluded checkpoint.
- `EXCLUDED_ARTIFACTS_SUMMARY.csv`: excluded binary/image categories and sizes.

The full domain manifest remains at `PeakTruthLab/datasets/domain_generalization_manifest.jsonl`; it is tracked once at its canonical location rather than duplicated here.

## Locked experimental protocol

- Input: 480 x 480; batch size 8; FP16.
- Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4.
- No data augmentation; 15 epochs for the final stability and cross-domain protocols.
- Attribute preprocessing was fitted from Train only.
- Checkpoints and Detection/Seed thresholds were selected using Val only.
- Test and External A/B/C were evaluated only after selections were locked.
- Cross-domain external sets were frozen as A = ST003127, B = ST003941 and C = ST003514, and were never pooled into one reported Test metric.

## Core three-seed stability results (mean +/- sample SD)

| Model | Detection F1 | Mean IoU | Seed balanced accuracy | Seed AUROC |
|---|---:|---:|---:|---:|
| Image only | 0.8608 +/- 0.0045 | 0.8696 +/- 0.0060 | 0.9651 +/- 0.0019 | 0.9938 +/- 0.0009 |
| Attributes only | 0.8521 +/- 0.0103 | 0.8777 +/- 0.0016 | 0.9520 +/- 0.0033 | 0.9901 +/- 0.0006 |
| Naive concat | 0.8613 +/- 0.0060 | 0.8765 +/- 0.0029 | 0.9663 +/- 0.0058 | 0.9954 +/- 0.0014 |
| Gated fusion | 0.8502 +/- 0.0083 | 0.8727 +/- 0.0065 | 0.9652 +/- 0.0048 | 0.9947 +/- 0.0013 |

## Core cross-domain Naive-concat results (mean +/- sample SD, n = 3)

| Domain | Detection F1 | Mean IoU | Seed balanced accuracy | Seed AUROC |
|---|---:|---:|---:|---:|
| Seen | 0.8460 +/- 0.0317 | 0.8637 +/- 0.0138 | 0.9261 +/- 0.0096 | 0.9796 +/- 0.0057 |
| External A | 0.8081 +/- 0.0192 | 0.8760 +/- 0.0095 | 0.9085 +/- 0.0235 | 0.9903 +/- 0.0026 |
| External B | 0.7900 +/- 0.0073 | 0.8331 +/- 0.0068 | 0.8663 +/- 0.0033 | 0.9431 +/- 0.0090 |
| External C | 0.8787 +/- 0.0053 | 0.8719 +/- 0.0068 | 0.9541 +/- 0.0086 | 0.9889 +/- 0.0007 |
| External macro | 0.8256 +/- 0.0065 | 0.8603 +/- 0.0062 | 0.9096 +/- 0.0115 | 0.9741 +/- 0.0038 |

External B was consistently the hardest domain, particularly for mean IoU and Seed classification. See `03_cross_domain/summary/external_B_failure_analysis.csv` and `external_B_attribute_shift.csv` for the quantitative shift analysis.

## Deliberate exclusions

No training was restarted for this archive. Large `.pt` files, dataset-release copies, review images, prediction overlays and generated PNG/PDF figures were not uploaded. These artifacts remain on the local D: drive and are indexed or summarized here. The preserved CSV/JSON/XLSX files are sufficient to redraw figures without depending on the excluded artwork.
