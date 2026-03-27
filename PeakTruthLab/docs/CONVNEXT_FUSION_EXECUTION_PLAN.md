# ConvNeXt + 峰属性融合：实验执行与数据集构建计划

## 0. 项目文件管理原则
- **临时代码统一管理**：所有数据清洗、过渡验证的临时脚本统一放置在 `PeakTruthLab/scripts/temp/` 文件夹中。主线执行使用固定的、干净的核心脚本。
- **去除冗余依赖**：已清理早期临时文件夹与失效数据源（从 `data/` 以及 `PeakTruthLab/` 目录中清理）。

### 0.1 强制执行规则（每次改代码前必读）
1. **先读后改**：每次开始任何代码改动前，必须先阅读本执行计划与仓库根目录 `README.md` 的“必读规则”章节，确认当前阶段目标与输出路径。
2. **唯一数据集目录**：EIC 最终数据集唯一有效目录固定为 `PeakTruthLab/datasets/eic_images_flat/`。
3. **唯一标签主表**：最终特征表唯一有效文件固定为 `PeakTruthLab/datasets/feature_table_final_10000.csv`。
4. **CSV-图像强一致**：对 `feature_table_final_10000.csv` 的增删改（尤其删除特征）必须同步到 `eic_images_flat`，保证一条特征对应一对文件（`*.png` + `*.json`），不得残留孤立图片或孤立记录。
5. **禁止中间目录沉积数据集**：`results/` 下仅允许过程分析产物，不允许作为最终训练数据集目录；若产生重复 EIC 图像目录，任务结束前必须清理。
6. **输出收敛原则**：同一批次数据只保留一份“最终可训练版本”，禁止多处并行保存导致口径不一致。

## 1. 模型架构与系统设计
当前确定的模型架构为**ConvNeXt（视觉分支） + 峰属性（属性分支）的多模态融合深度学习网络**。
1. **视觉分支**：基于 ConvNeXt 网络解析 EIC 二维特征。
2. **属性分支**：通过两层 MLP（64 hidden）提取 13 项通过文献验证的核心峰物理属性特征。
3. **特征融合**：Concat + Gating 机制将两路特征融合。
4. **输出层**：统一评分器（二分类：预测是否为纯净真峰）。

## 2. 数据集构建方案（最新版）
当前的基础数据池已生成，具体路径如下：
- **EIC 图像数据集**：`D:\LipidBench\PeakTruthLab\datasets\eic_images_flat`。该文件夹存放所有来自不同 mzML 文件的特征峰波形截图（一对一对应）。
- **峰特征综合表**：`D:\LipidBench\PeakTruthLab\datasets\feature_table_final_10000.csv`。该全量表仅包含核心基准信息以及13个经过筛选保留的核心统计与文献峰形属性：（`SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG`），所有历史冗余的 `peak_*` 数据均已移除。

### 2.0 数据一致性验收门禁（必过）
- 对目标 mzML 子集执行验收时，必须同时满足：
  - `csv_feature_count == png_count == json_count`
  - `extra_png == 0`
  - `missing_png == 0`
- 任何一项不满足，视为数据集未完成，不允许进入训练阶段。

### 2.1 标注与拆分流程（下一步）
- 通过 **labelme** 对 EIC 图像进行快速标注（后续详细进行）。
- 拿到标注信息后，依据 Label 结果，通过脚本自动将 `eic_images_flat` 中的样本**物理分离或通过CSV拆分划归到“真峰 (True_Peak)”与“假峰 (False_Peak)”对应的文件/类别中**。
- 将最终标签同步回 `feature_table_final_10000.csv` 里的 `is_true_peak` 列，随后切分 Train / Test 数据进行深度学习模型训练。

## 3. 训练配置预案 (RTX 5060)
- **Batch Size**: 16
- **Epochs**: 30
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)

### 3.1 训练脚本调整预期
后续训练时，需要更新模型读取的列名：
```bash
python PeakTruthLab/scripts/convnext/train_convnext_fusion.py \
  --train-csv PeakTruthLab/datasets/train.csv \
  --val-csv PeakTruthLab/datasets/val.csv \
  --image-root PeakTruthLab/datasets/eic_images_flat \
  --attr-columns SNR,CV,GS,TPAS,H2B,ZZ,DZZ,PCC,SKEW,DENT,DM,ENT,JAG \
  --batch-size 16 \
  --epochs 30 \
  --save-dir PeakTruthLab/models/convnext_fusion_exp1
```
