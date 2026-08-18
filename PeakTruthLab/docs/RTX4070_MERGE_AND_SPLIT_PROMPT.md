# RTX 4070 Super：最终数据合并、主实验与完整留域实验执行方案

> 给 RTX 4070 Super 电脑上 Codex 的指令：**读取本文件，并按顺序完整执行。直接检查文件、编写或调整脚本、发送并运行命令、保存结果；不要只给出命令让我手动运行。遇到会覆盖本机人工标注、划分泄漏、数据不一致或 Concat 未通过门槛时停止并报告。**

## 0. 固定原则

- 这台 RTX 4070 Super 电脑上，GitHub 上传后又人工修改约 150 张的旧数据，才是旧批次最终版；GitHub 中更早的旧数据不是最终版。
- 先保护本机旧最终数据。不得执行 `git reset --hard`，不得覆盖 dirty worktree 或已有人工标注；下载与解压应放到单独的干净目录。
- 新旧数据合并后写入新目录，不在原目录上修改。
- 所有实验统一使用 16 个 peak attributes：

  ```text
  SNR,CV,GS,TPAS,H2B,ZZ,DZZ,PCC,SKEW,DENT,DM,ENT,JAG,SYM,MOD,EDGE
  ```

- 不再安排任何 13 属性实验。
- 主实验先完成四种固定模式的消融；只有验证集确认 `Concat` 最佳后，才继续完整留域实验。
- 主实验的 test 一经生成立即锁定，绝不参与模型选择、阈值选择、早停、插补、标准化或超参数调整。

## 1. 下载并核验新增 4500 张数据

发布信息：

- Repository：`yzzx02/LipidBench`
- Branch：`codex/manual-negative-4500-v2`
- Release tag：`peaktruthlab-manual-negative-4500-v2`
- Asset：`PeakTruthLab_manual_negative_4500_v2_20260814.zip`
- 下载地址：<https://github.com/yzzx02/LipidBench/releases/download/peaktruthlab-manual-negative-4500-v2/PeakTruthLab_manual_negative_4500_v2_20260814.zip>
- SHA256：`F89632310064E5F92A484AB7FD75A05ABF3DC55631F7077B2C9A328D4F0CB409`

下载后先核验 SHA256，再解压。新增包应满足：

- 4500 张 PNG；
- 4500 个修正后的 LabelMe JSON；
- 4500 条 Seed 记录；
- 335 个人工峰实例，其中 328 个 `True_Peak`、7 个 `OUT_FIG`；
- Seed 与人工峰表均包含上述 16 属性，且无 NaN/inf；
- `F0108` 已恢复到原始 Seed 边界。

任何关键数量或校验值不符，都应停止合并并报告。

## 2. 定位并保护本机旧最终数据

1. 在本机定位“人工修改约 150 张之后”的旧最终数据，不能用 GitHub 中较早的版本替代。
2. 对候选目录进行只读盘点：绝对路径、PNG/JSON/Seed 数量、人工框数量、最近修改时间、代表性文件哈希和已有自检报告。
3. 若存在多个无法可靠区分的候选最终目录，先列出证据并停止，不要猜。
4. 旧数据预计约有 15317 个可用 Seed 图像，但以本机最终标注和自检结果为准。
5. 不修改、不移动、不覆盖旧最终目录；合并产物写到新的版本目录。

## 3. 统一到 16 属性并合并

若旧最终表仍只有 13 属性：

- 原 13 属性的算法和值保持不变；
- 仅补算 `SYM`、`MOD`、`EDGE`；
- EIC extraction、Seed window 和 apex/seed index `A` 的定义与现有流程一致；
- 人工修改后的 x 边界是权威边界，不得再次运行自动边界精修或找谷底算法去覆盖人工结果；
- 原始表中的缺失状态应保留，模型用插补器只能在每一训练折的 train 上拟合。

建议新目录：

```text
PeakTruthLab/datasets/PeakTruthLab_final_merged_20260814
```

合并要求：

- 保留全部人工框、同图多框、`OUT_FIG` 和无框的 Seed 假峰图；
- 统一且唯一的 `image_id`、Seed ID、峰实例 ID，同时保留原 ID、source mzML、dataset/domain、old/new batch 等溯源字段；
- 图片和 JSON 路径改为可移植的相对路径，不能写死原电脑盘符；
- 精确去重：ID、原文件 SHA256、图片 SHA256；
- 近重复检查：同一来源内 m/z 约 5 ppm、RT 约 2 秒、图片感知哈希高度相似；
- 精确或近重复样本写入统一 `duplicate_group_id`，以后不得跨 split；
- 新旧标注冲突必须写入冲突审计，禁止静默覆盖。人工确认前优先保留 4070 本机旧最终版，并把冲突 group 排除出 test；
- 若旧数据确为 15317 且没有重复，合并后预期为 19817；实际数量必须由审计报告解释。

输出至少包括：

- 合并后的 Seed 主表、人工峰实例表、PNG/JSON 数据目录；
- 重复组、冲突和排除项审计；
- 16 属性的 NaN/inf、min、分位数、median、mean、max；
- 按 dataset/domain、source mzML、真假 Seed、是否有框、框标签、old/new batch、困难类型的统计；
- 完整 manifest、配置、文件 SHA256 和可复现自检报告。

## 4. 主实验划分

固定随机种子 `20260814`，以图片/候选为最小单位做样本级分层随机划分：

```text
80% train / 10% val / 10% test
```

这不是完整 mzML 互斥划分。执行规则：

- 同一图片的所有框必须处于同一 split；
- exact/near-duplicate group 必须整体进入同一 split；
- 主要保持人工 Seed 真/假比例、来源文件比例和 old/new batch 比例；在可行范围内兼顾是否有框与峰标签分布；
- `NOISE/JAG/PLAT/SHOULDER/DOUBLE/EDGE` 等字段仅是挖掘来源或形态元数据，不是真假标签，也不得自动当作假峰；只做分布审计，不强制用它们构造联合分层；
- 小来源无法机械满足 80/10/10 时，采用确定性分配并记录偏差；不得复制样本；
- test manifest 生成后保存 SHA256 并锁定；
- 输出 train/val/test 的 CSV 或 JSONL，以及 split 审计报告；
- 确认图片 ID、路径、内容哈希和 duplicate group 在三个 split 间的交集均为 0。

## 5. 数据预处理与固定训练配置

所有模式和所有留域折统一遵循：

| 项目 | 固定设置 |
|---|---|
| 输入尺寸 | 480×480 |
| Batch size | 16 |
| Optimizer | AdamW |
| Learning rate | `1e-4`，不按 batch size 线性放大 |
| Weight decay | `1e-4` |
| 最大 epoch | 30 |
| Random seed | `20260814` |
| 二分类最佳 checkpoint | 最高 `val_auc`，与现有 `train_convnext_fusion.py` 一致 |

此外，预训练权重、数据增强、dropout、loss、检测 head、Seed head、阈值策略等，沿用上一次已冻结的最终实验设置，不在本轮随意重调。若需变更，必须先记录原因，并对所有比较模式一视同仁。

16 属性处理必须防止泄漏：

- 原始表保留原始缺失值；
- 插补器、标准化器及任何统计变换只在对应 train 上拟合；
- 将同一组已拟合参数应用到 val/test；
- 保存拟合参数、缺失掩码统计和特征顺序；
- 在模型入口断言属性维数严格为 16，并检查列顺序。

先做一次 batch 16 的 1 epoch smoke test，并完整跑一次验证，检查显存、CUDA 错误、NaN、梯度、保存和恢复。确认稳定后从头开始正式 30 epoch 训练，不能把 smoke 权重接入正式比较。

## 6. 主实验：固定四组消融

在完全相同的 main train/val/test、数据处理和训练预算上，从头分别训练：

1. `Attribute-only`：仅使用全部 16 属性；
2. `Image-only`：仅使用图像；
3. `Concat`：图像特征与全部 16 属性直接拼接；
4. `Image-gated-Attribute`：沿用已经确定的“图像 gate 属性”实现，不重新设计结构。

四个模式都要保存训练日志、逐 epoch 验证指标、最佳 checkpoint、最后 checkpoint、配置、随机种子和逐样本预测。模型选择只能看 validation：

- 主指标：`val_auc`；
- 次指标：`val_pr_auc`、F1、Recall、Precision；
- 不允许依据 main test 选择模式、epoch 或阈值。

### Concat 继续门槛

完整留域实验开始前，必须先确认 `Concat` 是验证集最佳方案：

1. 若 `Concat` 的 `val_auc` 明确最高，则锁定 Concat-16；
2. 若 Concat 与第二名的 `val_auc` 差值小于 `0.005`，只对 Concat 和第二名再补两个随机种子，形成各 3 个种子，比较 `val_auc` 均值与标准差；仅在 Concat 经多种子确认最佳后继续；
3. 若 Concat 不是最佳，立即停止并报告，不要开始 LODO，也不要为使其获胜而查看或反复调整 test；
4. Concat 通过后锁定模型结构、输入、超参数、训练轮数和选择规则。

模型依据 validation 全部预先选定并锁定后，四种模式各自只在锁定的 main test 上评估一次，形成最终主实验消融表。报告 ROC-AUC、PR-AUC、F1、Precision、Recall、specificity、混淆矩阵及适用的检测/Seed 指标，并保存逐样本预测，便于置信区间和配对比较。

## 7. 完整逐数据集留域实验（LODO）

Concat-16 通过上一节门槛后，执行 Leave-One-Dataset/Domain-Out。这里的“域”按独立公开数据集或独立采集项目定义，不是每个 mzML 都单独作为一个域：

- 同一公开项目、同一采集协议中的技术重复或明显相关 mzML 合并为一个 domain；
- 每个科学上独立的数据集/domain 都必须轮流被完整留出一次，不能只挑表现好或差异明显的几个；
- 每折仅训练已锁定的 `Concat + 16 attributes`，不在每折重复四种消融；
- 每一折从头训练，不能载入上一折权重；
- 留出的完整 domain 仅作该折 test；其余 domains 组成 train/val；
- 其余 domains 内可按 Seed 真/假分层划分 train/val；留出 test domain 保留天然真假比例，不重采样成训练比例；
- 留出 test domain 不得参与 checkpoint、阈值、插补、标准化、学习率、早停或任何选择；
- 每折使用同一套已锁定配置和同一个固定种子 `20260814`；第一轮不自动扩展成多种子 LODO；
- 如果某个留出数据集只有一个类别，也必须保留该折并报告能计算的指标，将 ROC-AUC/PR-AUC 等不可定义项明确写为 `undefined`，不得因结果差而删折。

每折保存：domain 定义及所属 mzML、train/val/test 数量与真假比例、配置、最佳/最后 checkpoint、逐 epoch 日志、逐样本预测、混淆矩阵和全部指标。

最终同时报告：

- 每个留出 domain 的独立结果；
- 所有 domain 的 macro mean、median、standard deviation；
- 最佳与最差 domain；
- 若有必要，再报告按样本数加权的总体值，但不能用它替代 macro 统计。

## 8. 验收与最终交付

数据阶段先提交一份报告，确认：

1. 本机旧最终数据路径及“约 150 张修改后版本”的证据；
2. 新包 SHA256 与数量校验；
3. 合并前后图片、Seed、人工框、真假样本数量；
4. 重复、近重复、冲突及排除项；
5. 16 属性 QC；
6. 80/10/10 划分统计与零泄漏检查；
7. test manifest 的路径和 SHA256。

训练阶段按顺序交付：

1. batch 16 smoke test 结果；
2. 四组主实验 validation 结果；
3. Concat 继续门槛判定，必要时两模型 3-seed 比较；
4. 四组锁定模型的 main test 最终消融表；
5. Concat-16 的每数据集完整 LODO 结果与汇总；
6. 全部脚本、配置、环境版本、manifest、日志、checkpoint、预测表、图和 SHA256。

论文结论应限制为：四模式主实验说明为何选择 Concat；完整逐数据集 LODO 检验模型对研究所覆盖独立数据域的未见域泛化能力。不要把结果表述成对所有未来 LC-MS 数据都普遍有效。

## 9. 执行顺序（不得跳步）

```text
保护并确认 4070 本机旧最终数据
→ 下载并校验新增 4500 张 v2
→ 新目录合并、去重、冲突审计、补齐 16 属性
→ 生成并锁定 main 80/10/10 split
→ batch 16 / 1 epoch smoke test
→ 从头训练四种固定消融模式
→ 仅依据 validation 判断 Concat 是否最佳
→ 必要时做 Concat 与第二名的 3-seed 复核
→ Concat 未胜：停止并报告
→ Concat 胜出：锁定配置并一次性评估 main test
→ 用锁定的 Concat-16 对每个独立 dataset/domain 完整 LODO
→ 汇总、审计并归档所有可复现材料
```
