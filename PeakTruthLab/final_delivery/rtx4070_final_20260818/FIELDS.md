# Field guide

## Final data tables

`tables/seed_master_16attrs.csv` contains one row per image/Seed candidate. Important fields include identifiers, old/new batch provenance, source mzML SHA-256, domain, m/z, Seed RT bounds, binary `seed_label`, human-box counts, portable image/JSON paths, the 16 raw attributes, content hashes, duplicate groups, conflict/exclusion flags, and split.

`tables/peak_instances_master_16attrs.csv` contains one row per annotated peak or `OUT_FIG` shape. It records peak/Seed/image IDs, domain and source, box coordinates, RT boundaries/apex, label provenance, paths, the 16 raw attributes, and split.

`tables/domain_definitions.csv` maps domain IDs to studies/source files. `tables/field_dictionary.csv` gives the compact attribute definitions.

## Result fields

- `detection_precision`, `detection_recall`, `detection_f1`: detection metrics after the Val-locked confidence threshold and IoU matching rule.
- `detection_mean_iou`: mean IoU of matched prediction/ground-truth boxes.
- `detection_ap50`, `detection_ap75`, `detection_map_50_95`: area under detection precision-recall curves at IoU 0.50, 0.75, and averaged from 0.50 to 0.95.
- `left_boundary_mae_px`, `right_boundary_mae_px`: mean absolute boundary error for matched boxes in pixels.
- `peak_count_mae`: mean absolute difference between predicted and annotated peak counts per image.
- `seed_balanced_accuracy`: average of Seed true-positive and true-negative rates after the Val-locked threshold.
- `seed_auroc`, `seed_average_precision`, `seed_f1`: threshold-independent AUROC/AP and thresholded F1 for Seed classification.
- `best_detection_epoch`, `best_seed_epoch`: Val-selected checkpoint epochs.
- `*_sample_sd`: sample standard deviation across the 11 LODO domains; populated in the macro row.

The Release result archive preserves the underlying threshold sweeps, PR/ROC curve points, image/Seed predictions, subgroup tables, epoch histories, configurations, and selection-before-target-access records.
