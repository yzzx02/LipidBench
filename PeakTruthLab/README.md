# PeakShape 模块（论文向）

目标：基于固定窗口 EIC 图像与峰属性，做色谱峰真假判别（binary classification）。

## 当前约束（已固定）
- EIC 窗口：2.0 min（RT±1.0）
- 图像尺寸：400x300
- DPI：100
- 强度：原始强度（不做 y 归一化）

## 数据输入建议
1. `EIC JPEG`（固定参数）
2. `peak_attributes.csv`（slope/sharpness/height/SN/width/mass accuracy...）
3. 标签：`is_true_peak`（0/1）

## 模型建议（比 QuanFormer 更可控）
- **Baseline-A**: LightCNN (ResNet18/EfficientNet-B0) + MLP 融合峰属性
- **Baseline-B**: ConvNeXt-Tiny + 属性门控融合
- **Baseline-C**: ViT-S/16（数据量足够时）
- **Advanced**: DETR/Transformer ROI detector（对应 QuanFormer 路线）

## 推荐论文主线
- 主模型：轻量 CNN+属性融合（二分类）
- 对照1：仅图像模型
- 对照2：仅峰属性模型（XGBoost/LightGBM）
- 对照3：QuanFormer 推理结果（如可复现）

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

