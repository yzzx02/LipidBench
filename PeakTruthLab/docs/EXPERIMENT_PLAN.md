# Experiment Plan

## 研究目标

实验需要分别回答三个问题：原始 Seed 属性能否改善候选真假验证；共享的 Seed 验证任务能否改善全窗口峰检测；模型在 source、study、instrument 隔离后能否保持泛化，并最终改善边界、拆峰与定量结果。

所有主要实验应预先固定数据版本、划分、随机种子、阈值选择规则和主评价指标。训练集用于拟合，验证集用于模型选择和阈值确定，测试集只用于一次性最终报告。同一 `source_file` 不得跨越不同集合。

## Dataset 单元

每个样本对应一个完整 EIC 窗口，至少包含：

- `image`：EIC 图像；
- `boxes`：窗口内全部 `True_Peak` 矩形框，允许 0、1 或多个；
- `seed_box`：传统算法生成的原始候选框，恰好一个；
- `seed_label`：原始 Seed 的独立真假标签；
- `attributes`：只由原始 Seed 计算的 13 项基础属性，可扩展到预先定义的 15 项版本；
- `source_file`、`study_id`、`instrument_id` 等分组与域信息；
- 困难子集标记和后续定量真值（如可用）。

检测框标签与 `seed_label` 相互独立。例如，原始 Seed 可以是假峰，但同一窗口的其他位置仍存在真实峰。

### 当前 manifest/Dataset 接口

`lipidbench.data.PeakManifestRecord` 定义 JSONL manifest 的单条记录；`PeakMultiTaskDataset` 在 `__getitem__` 时才通过可注入 loader 打开图像，初始化和审计阶段不会读取图像。`collate_peak_multitask_batch` 保留每张图不同数量的检测框，并整理为现有模型所需的列表/张量结构。

13 项基础属性的固定顺序为：

```text
SNR, CV, GS, TPAS, H2B, ZZ, DZZ, PCC, SKEW, DENT, DM, ENT, JAG
```

manifest 必填字段为：

```json
{
  "sample_id": "unique-window-id",
  "image_path": "relative/path/to/eic.png",
  "boxes": [[120, 20, 180, 455]],
  "seed_box": [118, 18, 184, 458],
  "seed_label": 1,
  "attributes": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
  "source_file": "source-A",
  "study_id": "study-A",
  "instrument_id": "instrument-A",
  "subsets": ["low_snr"],
  "metadata": {}
}
```

`boxes` 可以为空列表。缺失属性用 JSON `null` 表示；Dataset 会返回 NaN 以及对应的 `attribute_mask`，不会静默填零。正式训练前必须仅使用训练集拟合插补和标准化参数，并把处理策略写入实验配置。

当前工具入口为：

```text
scripts/detection/audit_peak_manifest.py
scripts/detection/split_peak_manifest.py
```

审计入口只生成统计和 Anchor 建议，不修改正式 YAML。切分入口默认按 `source_file` 分组，拒绝覆盖已有切分文件，并在输出前再次检查 source 泄漏。

## Seed 融合消融

在相同骨干初始化、数据划分、训练轮数、优化器和检测损失设置下比较：

1. `image_only`：只使用 Seed RoI 视觉特征；
2. `attr_only`：只使用 Seed 属性；
3. `naive_concat`：视觉嵌入与属性嵌入直接拼接；
4. `gated_fusion`：属性与视觉联合产生门控后再分类。

主要比较 Seed 验证性能，同时记录检测性能，检查融合方式是否通过共享骨干影响检测任务。除融合头必要差异外，其他训练设置保持一致。至少报告多个随机种子的均值、标准差和置信区间。

## 单任务与多任务消融

比较：

- `single-task detection`：只优化 Faster R-CNN 检测损失；
- `multitask detection`：使用相同检测结构，同时优化检测损失和 `loss_seed_cls`；
- 可选的 `single-task seed validation`：只训练 Seed 验证任务，用于分析共享检测监督的反向增益。

主比较必须使用同一检测骨干、FPN、Anchor、图像预处理和数据划分。报告检测主指标以及不同 `seed_loss_weight` 下的敏感性分析，避免把参数量或训练预算变化误认为多任务收益。

## 13 维与 15 维属性比较

- 13 维版本作为基础属性集合；
- 15 维版本只增加预先定义的两项扩展属性；
- 使用同一批样本，禁止因扩展属性缺失而形成不同测试人群；
- 同时比较属性缺失率、相关性、标准化方式和性能增量；
- 增加属性打乱或置换对照，排查数据泄漏和伪相关；
- 报告增量的置信区间，而不只比较单次最高分。

模型接口允许任意正整数属性维度，但当前论文重点只比较预先注册的 13 维与 15 维方案。

## 分组与跨域验证

### 基础分组切分

以 `source_file` 为不可拆分组执行 train/val/test 划分。任何由同一原始文件导出的窗口、增强样本或重复候选都必须进入同一集合。切分后运行自动泄漏检查。

### 跨 study 验证

采用 leave-one-study-out 或预先指定外部 study 测试：训练和验证不包含测试 study。若 study 数量允许，轮流留出并汇总宏平均与各 study 结果。

### 跨 instrument 验证

将未见过的仪器型号、采集平台或实验室作为外部测试域。报告域内验证与域外验证差距，并同时给出各 instrument 的样本量、Seed 阳性率和峰几何分布。

所有跨域实验仍必须满足 `source_file` 隔离；不得因为同一 source 的衍生图像出现在不同域标签下而绕过检查。

## 困难子集评价

在不参与测试阈值调优的前提下，预先定义并分别评价：

- 低 SNR：依据训练集统计或分析标准预先确定阈值；
- 肩峰：存在肩部结构、主峰与肩峰边界容易混淆；
- 双峰/相邻峰：一个窗口内需要恢复两个相邻真实峰；
- 拖尾：左右形状明显不对称或尾部延伸；
- RT 偏移：传统 Seed 中心或边界相对人工真值发生位移。

同一样本可以属于多个子集。需要报告每个子集定义、样本量、阳性率和置信区间，避免只选择表现较好的案例。

## 评价指标

### Seed 验证

- AUROC 与 AUPRC（类别不平衡时以 AUPRC 为主要指标）；
- 灵敏度、特异度、precision、recall、F1 和 balanced accuracy；
- 固定高 precision 下的 recall，或固定高 sensitivity 下的 specificity；
- Brier score、校准曲线和 ECE；
- 混淆矩阵及按 source/study/instrument 的分层结果；
- bootstrap 置信区间和配对模型差异。

分类阈值只能在验证集确定，不能使用测试集重新优化。

### 全窗口检测与边界

- COCO 风格 AP@[0.50:0.95]、AP50、AP75；
- 每图检测 recall、precision、漏检数和假阳性数；
- 匹配框的 mean/median IoU；
- 左边界和右边界的绝对误差，分别以像素和映射后的 RT 表示；
- 峰中心偏移、预测宽度误差和边界覆盖率；
- 空框、单框、多框样本的分层指标。

匹配规则和 NMS/分数阈值必须预先固定。当前 Faster R-CNN 使用常规 proposal/RoI 机制，不引入 DETR 或匈牙利算法。

### 拆峰恢复

- 双峰窗口中两个真实峰均被恢复的比例；
- 被传统算法合并的峰中，新增正确实例的召回率；
- 过拆分率和每窗口额外假阳性数；
- 峰个数绝对误差；
- 相邻峰间最小距离或重叠程度分层后的恢复率。

### 定量评价

在未来完成独立面积积分流程后评价：

- 峰面积绝对误差、相对误差、median absolute percentage error；
- 与人工或参考方法面积的 Pearson/Spearman 相关性和回归斜率；
- Bland-Altman bias 及一致性区间；
- 重复样本 CV、批内/批间稳定性；
- 边界校正前后和拆峰前后的配对定量误差变化。

面积计算本身不是本文核心创新；重点是检测与校正是否带来可重复的定量改善。本阶段不实现 mzML 读取或面积积分。

## Anchor 审计

训练前统计检测框宽、高、面积、`height/width` 和尺度分位数，并输出建议 Anchor 尺度与宽高比。建议仅作为实验诊断，不自动修改 `configs/peak_multitask.yaml`。任何正式 Anchor 变更都需要在训练集统计基础上单独记录和消融。

## 当前停止边界

本轮只实现文档、manifest/Dataset 接口、source 分组切分、数据审计和合成测试。暂不实现正式训练、mzML 读取、面积积分、预测峰属性重算、Instance Fusion Head、QuanFormer/DETR、匈牙利算法或新的模型骨干。
