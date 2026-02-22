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

