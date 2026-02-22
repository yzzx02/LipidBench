# 最终执行方案（投稿版）

## 阶段1：数据固化（1-2周）
- 用 GUI 导出两类峰属性数据：
  - 单算法全特征峰属性
  - 多算法对齐后未检出峰属性
- 固定 EIC 图生成参数（已完成）
- 人工审核或规则生成标签（true/false peak）

产物：
- `dataset/train|val|test/images/*.jpeg`
- `dataset/*.csv`（feature_id, label, attrs...）

## 阶段2：基线模型（1周）
- 属性模型：XGBoost / LightGBM
- 图像模型：ResNet18
- 融合模型：ResNet18 + MLP(attrs)

目标：建立可复现实验基线

## 阶段3：Transformer对照（1周）
- 复现实验：QuanFormer（可行时）
- 轻量 Transformer：ViT-S/16 或 ConvNeXt-Tiny 对照

## 阶段4：泛化验证（1周）
- 跨算法训练/测试：train(pyopenms) -> test(asari/xcms)
- 跨批次/跨样本验证

## 阶段5：论文撰写（1周）
- 方法：固定输入协议 + 双模态融合
- 结果：主指标 + 消融实验 + 泛化
- 复现：提供配置、版本、随机种子

## 可优化空间（相对 QuanFormer）
1. 从检测任务降维到二分类任务，减少标注成本
2. 引入峰属性融合，提高可解释性
3. 固定输入参数，减少域偏移
4. 先做轻模型，再扩展 Transformer
