# Script Layout (Model-Oriented)

## convnext
- `scripts/convnext/train_convnext_fusion.py`
  - ConvNeXt-Tiny + 属性融合训练（含 `--vision-backbone lwga_convnext` 开关）

## detection
- `scripts/detection/train_fasterrcnn.py`
- `scripts/detection/eval_fasterrcnn.py`
- `scripts/detection/analyze_fasterrcnn_predictions.py`
- `scripts/detection/run_fasterrcnn_pipeline.py`
- `scripts/detection/train_val_split.py`

## data_prep
- `scripts/data_prep/build_pyopenms_feature_pool.py`
- `scripts/data_prep/generate_eic_images_from_pool.py`
- `scripts/data_prep/rebuild_feature_table_with_peak_attrs.py`
- `scripts/data_prep/rebuild_dataset_from_xcms_subset.py`
- `scripts/data_prep/refresh_top550_area_and_finalize.py`
- `scripts/data_prep/export_eic_dataset.py`

## annotation
- `scripts/annotation/annotate_is_true_peak.py`
- `scripts/annotation/inspect_labels.py`
- `scripts/annotation/remap_labelme_json_labels.py`
- `scripts/annotation/recompute_rt_bounds_and_attrs.py`
- `scripts/annotation/recompute_rt_bounds_from_labelme.py`

## root policy
- `scripts/` 根目录仅保留架构子目录与本说明文档。
- 统一按任务域归类，不再保留重复入口脚本。
- 任何会改写 `PeakTruthLab/datasets/feature_table_final_10000.csv` 的脚本，必须先自动备份到 `PeakTruthLab/datasets/backups/`（时间戳文件名，不覆盖历史）。
