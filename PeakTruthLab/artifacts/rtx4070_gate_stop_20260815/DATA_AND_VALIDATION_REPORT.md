# RTX 4070 merged-data and validation-gate report

## Outcome

The four frozen main-ablation runs completed successfully. The predeclared gate stopped the experiment before Test and LODO because Image-gated-Attribute, not Naive concat, achieved the highest validation ROC-AUC. No Test predictions or Test-derived tuning were performed.

## Protected and merged data

- Protected old-final images: 15,317
- New v2 images: 4,500
- Merged images / Seed candidates: 19,817
- Seed positives / negatives: 11,109 / 8,708
- Peak instances: 18,915 (18,815 True_Peak; 100 OUT_FIG)
- Exact ID or image-SHA duplicate groups: 0 / 0
- Near-duplicate groups: 43; audited label-conflict groups: 4; Test-excluded conflict rows: 8
- Sixteen raw attributes: SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG, SYM, MOD, EDGE
- Missing old-final values were preserved; median imputation and population z-score parameters were fitted on Train only for every run.

## Frozen split and leakage checks

- Train: 15,853 (8,887 positive / 6,966 negative)
- Val: 1,982 (1,108 positive / 874 negative)
- Locked Test: 1,982 (1,114 positive / 868 negative)
- Train/Val/Test intersections are zero for image ID, Seed ID, path, annotation path, image SHA-256 and duplicate/split group.
- Locked Test CSV SHA-256: `b92aefb41be6dfdbbb025059873e7b5cbd3ce381c534d153b109fa7bd04de9c8`
- Locked Test JSONL SHA-256: `618abf3661ffb5a2bfe5a58b64ba444604353c0e27f678e52dcbd3a7f49b4461`
- Independent verification status: `ok`

## Frozen training configuration

Input 480x480; batch size 16; 30 epochs; seed 20260814; FP16; AdamW; learning rate 1e-4; weight decay 1e-4; dropout 0.2; no augmentation; pretrained ConvNeXt-Tiny where images are used. Checkpoint selection used maximum Val ROC-AUC. The classification threshold was selected on Val by a 0.001 grid maximizing balanced accuracy, then F1, then closeness to 0.5.

## Main-ablation validation results

| Rank | Model | Best epoch | ROC-AUC | PR-AUC | F1 | Precision | Recall | Balanced accuracy | Threshold | Minutes | Peak GiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Image-gated-Attribute | 7 | 0.995450 | 0.996471 | 0.969969 | 0.963491 | 0.976534 | 0.964812 | 0.155 | 68.68 | 4.71 |
| 2 | Naive concat | 7 | 0.995133 | 0.996248 | 0.970936 | 0.977148 | 0.964801 | 0.968099 | 0.502 | 68.53 | 4.71 |
| 3 | Image-only | 7 | 0.993898 | 0.995282 | 0.963964 | 0.962230 | 0.965704 | 0.958825 | 0.299 | 68.39 | 4.71 |
| 4 | Attribute-only | 30 | 0.989765 | 0.992215 | 0.952162 | 0.970919 | 0.934116 | 0.949323 | 0.602 | 2.79 | 0.06 |

## Gate decision

Image-gated-Attribute exceeded Naive concat by 0.000317020 Val ROC-AUC. Because Naive concat was not rank 1, the frozen protocol requires an immediate stop. No extra seeds were launched, the locked main Test was not evaluated, and LODO was not started.

## Reproducibility and archive policy

The core archive contains source tables, split manifests, QC/audit tables, smoke-test outputs, all four run configurations, environments, Train-only scalers, epoch histories, Val predictions, threshold sweeps and selection summaries. Large `.pth` files remain in the local result directories and are represented by path, size and SHA-256 in `checkpoint_sha256.csv`.
