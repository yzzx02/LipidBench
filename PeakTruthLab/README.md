# PeakShape 模块（论文向）

目标：基于固定窗口 EIC 图像与峰属性，做色谱峰真假判别（binary classification）。

## 最终决策（已锁定）
- **主模型**：`ConvNeXt-Tiny + 峰属性融合（gating）`
- **主任务**：`is_true_peak` 二分类（0/1）
- **当前不做**：目标检测/分割（作为后续扩展）
- **输出**：`P(true_peak)` + 分类结果 + 合理性评分（High/Medium/Low）

## LabelMe 是否必须？
- **二分类主线：不必须。**
- 训练最小必需数据：`image` + `is_true_peak` + 峰属性列。
- 仅当做检测（bbox）或分割（mask）时，才必须依赖 LabelMe 框选。
- 若你目前只能“看图后人工判断”，可先用 LabelMe 做人工标注，再自动转成二分类 CSV（见下方脚本）。

## 当前约束（已固定）
- EIC 窗口：2.0 min（RT±1.0）
- 图像尺寸：400x300
- DPI：100
- 强度：原始强度（不做 y 归一化）

## 数据输入建议
1. `EIC JPEG`（固定参数）
2. `peak_attributes.csv`（slope/sharpness/height/SN/width/mass accuracy...）
3. 标签：`is_true_peak`（0/1）

### CSV 最小字段
- `image`
- `is_true_peak`

### CSV 推荐字段
- `sample_id`, `batch_id`, `instrument`, `algorithm_source`
- `slope`, `sharpness`, `height`, `sn`, `width`, `mass_accuracy`

## 模型建议（比 QuanFormer 更可控）
- **Baseline-A**: LightCNN (ResNet18/EfficientNet-B0) + MLP 融合峰属性
- **Baseline-B**: ConvNeXt-Tiny + 属性门控融合
- **Baseline-C**: ViT-S/16（数据量足够时）
- **Advanced**: DETR/Transformer ROI detector（对应 QuanFormer 路线）

> 当前版本主线固定为 Baseline-B（二分类）。

## 推荐论文主线
- 主模型：轻量 CNN+属性融合（二分类）
- 对照1：仅图像模型
- 对照2：仅峰属性模型（XGBoost/LightGBM）
- 对照3：QuanFormer 推理结果（如可复现）

## 推理输出规范（单张图合理性判断）
- `pred_label`（0/1）
- `p_true_peak`（模型概率）
- `rule_score`（规则一致性）
- `attr_score`（属性质量分）
- `final_score`（综合分）
- `rationale`（文字解释）

推荐综合分：

$$
S = 0.5\,P_{model} + 0.3\,R_{rule} + 0.2\,A_{attr}
$$

分级建议：
- `S >= 0.75`：High confidence
- `0.50 <= S < 0.75`：Medium confidence
- `S < 0.50`：Low confidence

## 评估指标
- AUC / PR-AUC
- F1 / Recall@Precision>=0.95
- 跨仪器与跨算法泛化（asari/pyopenms/xcms）

## 目录约定（与 LipidBench 解耦）
- `datasets/`：训练数据（EIC 图 + LabelMe JSON）
- `scripts/`：数据构建脚本
- `models/`：模型定义
- `train/`：训练与评估脚本
- `configs/`：实验配置

## 当前已生成数据
- 原始 mzML：`data/DIA_mzML/HILIC-Pos-SWATH-25Da-20140701_08_GB004467_Swath25Da.mzML`
- 特征表：`Results/pyopenms/xcms/xcms_features.csv`
- 输出目录：`PeakTruthLab/datasets/eic_images/HILIC-Pos-SWATH-25Da-20140701_08_GB004467_Swath25Da`
- 数量：`299` 张 JPEG + `299` 个同名 LabelMe JSON

## 新增训练框架（ConvNeXt-Tiny + 峰属性融合）
- 训练脚本：`scripts/train_convnext_fusion.py`
- 执行计划：`docs/CONVNEXT_FUSION_EXECUTION_PLAN.md`
- 支持模式：二分类主线 + 多类别扩展（无需回退到 Faster R-CNN）。
- 适用场景：约 1w 张数据的真假峰二分类 + 传统算法候选峰质量比较。

## 执行顺序（简版）
1. 整理 `train/val/test.csv`（先做二分类）
2. 跑三组基线：image-only / attr-only / fusion
3. 固定阈值并在测试集导出 ROC、PR、混淆矩阵
4. 在 asari/pyopenms/xcms 候选峰上统一打分比较
5. 导出单图解释报告（概率 + 评分 + 原因）

## 快速数据拆分（无需手动移动图片）
- 新增脚本：`scripts/make_binary_split_csv.py`
- 作用：自动生成 `train.csv / val.csv / test.csv`，不需要手动把图片移动到 true/false 文件夹。
- 支持两种模式：
	1. `csv` 模式：从已有标签 CSV 直接分层拆分（推荐）；
	2. `folder` 模式：若你已按 true/false 文件夹整理，也可自动生成 CSV。

## LabelMe 标注快速转二分类 CSV
- 新增脚本：`scripts/labelme_to_binary_csv.py`
- 用途：把 LabelMe JSON 自动转换为 `image,is_true_peak`，并可选导出 true/false 文件夹供人工复核。
- 适用逻辑：
	- `shapes` 为空 => `is_true_peak=0`
	- 含 `True_Peak`（及别名，如共流出/锯齿等）=> `is_true_peak=1`

## 决策口径（统一）
- 训练与推理的最终判定以 **Fusion 模型输出** 为准（图像分支 + 属性分支端到端学习）。
- 为可解释性可同时保留 image-only / attr-only 分数作为审计信息。
- 属性分支默认使用训练集统计量做 z-score 标准化（避免量纲差异导致训练失衡）。

