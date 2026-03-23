# ConvNeXt-Tiny + 峰属性融合：完整执行计划（1w 张、RTX 5060）

> 版本说明（2026-03-23）：本版本已按“二分类优先、不过早做目标检测”的策略锁定最终路线。

## 0A. 最终拍板（本项目当前最终方案）

### 最终模型（主线）
- **模型名称**：`ConvNeXt-Tiny + 峰属性融合（gating）`
- **任务定义**：**真假峰二分类**（`is_true_peak`：0/1）
- **输入**：固定参数 EIC 图像（400x300） + `peak_attributes.csv`
- **输出**：`P(true_peak)` + 二分类标签 + 合理性评分（High/Medium/Low）

### 是否需要 LabelMe 框选峰？
- **主线不需要**（当前论文主任务：真假峰分类）。
- 只要每张图对应一个标签（0/1）和属性，即可训练。
- **需要 LabelMe 框选**的场景仅限：
  1) 你要做目标检测（bbox）；
  2) 你要做分割（mask）；
  3) 你要输出精确峰边界作为学习目标。

### 你的问题的直接结论
- **不框选**时，当前任务就是标准图像分类（可融合属性）。
- 多峰图像可采用“中心优先规则”做样本筛选/过滤，但不改变主任务定义。

## 0B. 方案复核与优化结论（吸收外部建议后）

### 架构定位（保留并强化）
- 本方案属于 **多模态融合（Multi-modal Fusion）/双分支网络（Dual-branch）**。
- 图像分支（ConvNeXt）学习峰形态，属性分支（MLP）学习物理/化学绝对量纲信息。
- 最终采用单 logit 端到端训练，**最终判定以融合模型输出为准**（深度学习主导决策）。

### 必做工程约束（已纳入执行）
1. **属性标准化必须开启**（训练集 z-score，验证/测试复用训练统计量）。
2. 图像输入保持固定参数与物理语义（禁旋转、保守增强）。
3. 小样本阶段优先二分类，避免过早上检测/分割导致过拟合。

### 标注与数据组织（与你当前习惯兼容）
- 可以先人工看图并用 LabelMe 标注，再自动转二分类 CSV（无需手动搬图）。
- 主训练仍以 `train.csv/val.csv/test.csv` 为统一输入格式。

### 决策解释策略
- 对外报告：融合模型概率 `P(true_peak)` 为主结论。
- 对内审计：同步导出 image-only / attr-only 分数和属性摘要，用于解释假阳性来源。

## 0. 目标（论文导向）
1. 使用相对更新的视觉模型（ConvNeXt-Tiny）完成真假峰二分类（主线），并保留多类别扩展能力。
2. 将模型作为统一评分器，比较传统算法（XCMS/OpenMS/asari）候选峰质量差异。
3. 只保留必要深度学习图表：loss 曲线、ROC/PR、混淆矩阵、算法比较图。

---

## 1. 数据与标注要求

### 1.1 数据规模与拆分
- 目标数据量：约 10,000 张 EIC 图像。
- 推荐拆分：
  - 训练集 70%
  - 验证集 15%
  - 测试集 15%
- 避免同一样本来源（同一批次/同一化合物窗口）同时出现在 train/test。

### 1.2 图像参数一致性
- 保持当前项目固定参数：窗口长度、图像尺寸、DPI 和强度绘图方式一致。
- 增强后输出仍保持 `400x300`，不要改变最终输入尺寸。

### 1.3 属性特征建议
- 至少包含：`slope`, `sharpness`, `height`, `sn`, `width`, `mass_accuracy`。
- 在 CSV 中提供：
  - `image`（相对路径）
  - `is_true_peak`（0/1）
  - 上述属性列

### 1.4 标签格式（主线）
- **必须字段**：`image`, `is_true_peak`
- **推荐字段**：`sample_id`, `batch_id`, `instrument`, `algorithm_source`
- **可选字段**：`quality_note`（人工备注，不参与训练）

### 1.5 LabelMe 使用策略（必须统一）
- 当前主线：**LabelMe JSON 非必需**。
- 如果历史上已有 LabelMe 文件，可保留用于：
  - 质控核对（人工查看峰位置是否合理）；
  - 后续检测/分割扩展。
- 训练脚本主线只依赖 `train.csv / val.csv / test.csv`。

---

## 2. 数据增强策略（面向峰形语义）

### 推荐默认增强
- 轻度随机裁剪（再 resize 回 300x400）
- 轻度平移（RandomAffine, 无旋转）
- 亮度/对比度轻微扰动

### 可选增强
- 高斯滤波（低概率，如 0.15）作为鲁棒性增强。

### 不建议
- 任意角度旋转（会破坏 RT/强度轴物理语义）。

---

## 3. 模型结构（已在脚本实现）

- 图像分支：`ConvNeXt-Tiny`（ImageNet 预训练可开关）
- 属性分支：两层 MLP（64 hidden）
- 融合：concat + gating（属性辅助调制图像特征）
- 输出：默认单 logit（二分类），可切换为多类别 softmax head

脚本：`PeakTruthLab/scripts/train_convnext_fusion.py`


### 3.1 任务模式
- 默认：`--task-type binary`（真假峰，推荐先完成论文主线）
- 扩展：`--task-type multiclass`（如 `True_Peak/True_Peak_Tailing/True_Peak_Jagged/...`）
- 多类别无需依赖 Faster R-CNN，可直接沿用 ConvNeXt 融合框架。

---

## 4. 训练配置建议（RTX 5060）

### 建议起步参数
- `batch_size=16`（显存不足时调到 8）
- `epochs=30`
- `optimizer=AdamW`
- `lr=1e-4`
- `weight_decay=1e-4`
- `dropout=0.2`

### 4.1 必须做的基线（论文最小可发表集合）
1. **Image-only**：仅 ConvNeXt-Tiny（无属性）
2. **Attr-only**：XGBoost 或 MLP（仅峰属性）
3. **Fusion（主模型）**：ConvNeXt-Tiny + 属性融合（gating）

结论必须基于这 3 组对比，而不是只报主模型。

### 训练命令模板
```bash
python PeakTruthLab/scripts/train_convnext_fusion.py \
  --train-csv PeakTruthLab/datasets/train.csv \
  --val-csv PeakTruthLab/datasets/val.csv \
  --image-root PeakTruthLab/datasets/eic_images \
  --attr-columns slope,sharpness,height,sn,width,mass_accuracy \
  --batch-size 16 \
  --epochs 30 \
  --enable-gaussian-blur \
  --save-dir PeakTruthLab/models/convnext_fusion_exp1
```

多类别扩展示例：
```bash
python PeakTruthLab/scripts/train_convnext_fusion.py \
  --task-type multiclass \
  --class-col peak_class \
  --train-csv PeakTruthLab/datasets/train.csv \
  --val-csv PeakTruthLab/datasets/val.csv \
  --image-root PeakTruthLab/datasets/eic_images \
  --attr-columns slope,sharpness,height,sn,width,mass_accuracy \
  --save-dir PeakTruthLab/models/convnext_fusion_multiclass
```

---

## 5. 必要输出（论文最低集）

脚本会导出：
- `best_model.pth`
- `history.json`（二分类含 val_auc/val_pr_auc/val_f1；多类别含 val_f1_macro/val_acc）

论文图表建议：
1. 训练/验证 loss 曲线
2. ROC 曲线 + PR 曲线
3. 测试集混淆矩阵
4. 不同传统算法候选峰的“高置信真峰比例”柱状图

---

## 6. 传统算法比较方案

### 6.1 输入
- 分别运行 XCMS/OpenMS/asari，得到候选峰列表。
- 用统一 EIC 绘图参数生成候选峰图片与属性。

### 6.2 评分
- 用同一个融合模型对三种算法候选峰打分（真峰概率）。
- 比较：
  - 平均概率
  - 超过阈值（如 0.5/0.8）的比例
  - Top-N 高置信峰的人工复核一致率

### 6.3 统计检验
- 两两比较可用 Mann–Whitney U 或 bootstrap CI。

---

## 6A. 单张图“检测判断与合理性解释”输出规范（你当前需求）

即使不做目标检测，也要给“可解释判断”。每张图推理后输出：

1. `pred_label`：0/1
2. `p_true_peak`：模型概率
3. `rule_score`：规则一致性（例如峰顶位置、左右对称性阈值）
4. `attr_score`：属性质量分（由 SN/width/sharpness 等归一化计算）
5. `final_score`：综合分
6. `rationale`：文字原因（例如“SN 高、峰宽合理、模型高置信”）

推荐综合分（可写入方法学章节）：

$$
S = 0.5\,P_{model} + 0.3\,R_{rule} + 0.2\,A_{attr}
$$

并定义：
- `S >= 0.75`：High confidence
- `0.50 <= S < 0.75`：Medium confidence
- `S < 0.50`：Low confidence

---

## 7. 风险与对策
- 类别不平衡：使用分层抽样/正类加权。
- 标签噪声：对低置信样本做人审回标。
- 过拟合：优先早停 + 保守增强，不追求过深模型堆叠。

---

## 8. 建议里程碑（6 周）
1. 周1：数据整合与 QC（10k 建库完成）
2. 周2：跑通 image-only / attr-only / fusion
3. 周3：超参微调与固定最佳模型
4. 周4：三种传统算法候选峰统一评分
5. 周5：统计图表与显著性分析
6. 周6：写作与补实验

---

## 9. 顶刊级必要条件（从 QuanFormer/EVA 借鉴的硬性清单）

为达到接近 QuanFormer / EVA 级别的说服力，建议把下列内容作为“必须交付项”：

### 9.1 高质量金标准数据集
- 目标：不仅是 10k，而是持续扩展至 20k+（若资源允许）并覆盖多平台。
- 最低建议：
  - 多供应商/多平台数据（Agilent / Bruker / Sciex / Thermo 至少覆盖 2 家以上）。
  - 多批次、多色谱条件（避免模型只记住单一实验条件）。
  - 人工标注 SOP（至少双人复核子集，记录一致率）。
- 交付：`DATA_CARD.md`（记录来源、仪器、批次、标注规则、训练/测试隔离规则）。

### 9.2 算法创新点必须明确
- 当前主线创新定位：
  - 新视觉 backbone（ConvNeXt）+ 峰属性融合 + gating。
  - 同一模型统一评测不同传统软件候选峰。
- 必做消融：
  - image-only
  - attr-only
  - image+attr（fusion）
  - （可选）binary vs multiclass 对照
- 结论应回答：你的模型为何比“简单 CNN”或“纯规则阈值”更适合峰识别。

### 9.3 基准对比必须“正面对打”
- 对比对象建议至少包含：`XCMS`、`MS-DIAL`、`MZmine`（若可得）和当前常见 DL 工具（若可复现）。
- 对比维度：
  - 峰真伪识别：AUC / PR-AUC / F1 / Recall@高Precision
  - 定量稳定性：CV、MAE、Fold-change 误差
  - 覆盖率：高置信真峰数量与比例

### 9.4 可用性与传播
- 建议最少提供：
  - 一键训练命令 + 一键推理命令
  - 推理输出标准格式（CSV+可视化图）
- 如有精力：提供简易 GUI 或 notebook workflow，提升期刊审稿人可复现感受。

### 9.5 真实生物学应用闭环
- 至少一个真实队列（临床或公开代谢组数据集）作为落地验证。
- 输出包括：差异分析（火山图）、分组可分性（PCA/PLS-DA）、通路富集（KEGG）。
- 要强调：模型改进不仅提升“指标”，还改变了下游生物结论质量。

---

## 10. 论文必备图表模板（建议按此顺序写文）

### Figure 1 — 总体框架图（必须）
- 从 mzML/mzXML 输入 -> EIC/ROI 构建 -> 模型推理 -> 输出峰真伪/类别/定量。
- 标注训练与推理两条流程，显示你方法相对传统流程的插入位置。

### Figure 2 — 模型性能与可解释性（必须）
- 主结果：ROC、PR、混淆矩阵、阈值-性能曲线。
- 可解释性：Grad-CAM/注意力热图，展示模型聚焦峰顶、边界、拖尾区域。

### Figure 3 — 与主流软件基准对比（必须）
- 箱线图/散点图：比较 MAE、CV、检测覆盖率。
- 强调统计检验（bootstrap CI 或非参数检验）。

### Figure 4 — 典型难例展示（必须）
- 逐例展示：拖尾峰、锯齿峰、共流出峰、RT 漂移、强噪声背景。
- 每例显示：原始 EIC、传统算法输出、模型输出（概率/类别）及人工结论。

### Figure 5 — 真实应用结果（顶刊关键）
- 使用你流程处理真实队列后的 PCA、火山图、通路富集图。
- 建议附加：与传统流程相比新增/更稳定的候选代谢物数量。

---

## 11. 文档修订结论（针对当前需求）
- 当前方案“二分类优先 + 多类别可扩展”与投稿策略一致：
  - 二分类先形成稳定主结果，降低项目风险；
  - 多类别作为增强章节/后续工作，避免首稿过重。
- 需要补强的核心不是继续堆模型复杂度，而是：
  1) 跨平台高质量标注；
  2) 对主流软件的系统 benchmark；
  3) 真实生物学场景闭环。

---

## 12. 面向脂质组学的算法升级路线（在现有框架上可迭代）

结合你总结的文献经验，建议把后续方法创新拆为“可发表且可执行”的三层路线：

### 12.1 Backbone 升级方向（按实现难度递进）
1. **当前主线（已具备）**：ConvNeXt-Tiny + 属性融合（binary 主线，multiclass 可扩展）。
2. **时序优先路线**：1D-CNN / TCN 直接处理 EIC 序列，减少 2D 图像化信息损失。
3. **全局上下文路线**：CNN + Transformer（或 BiLSTM）增强 RT 漂移与共流出建模能力。
4. **高难度路线**：ROI 检测或分割（DETR/U-Net 思路），解决重叠峰与拖尾边界。

> 建议投稿策略：首稿保留 binary 主线 + 消融；第二阶段再引入时序/Transformer 扩展。

### 12.2 任务定义建议（先稳后强）
- **第一阶段（推荐先完成）**：真假峰二分类（最稳、最容易形成可复现证据链）。
- **第二阶段（可选增强）**：多类别峰形（拖尾/锯齿/共流出）分类。
- **第三阶段（进阶）**：检测或分割，输出峰边界与积分区间。

### 12.3 损失函数与不平衡处理
- 二分类：BCE（可加 class weight） -> 若假峰占多数，优先尝试 Focal Loss。
- 多类别：CrossEntropy（类别加权）或 Focal CE。
- 检测任务（后续）：可参考 L1 + IoU/CIoU 联合损失。

### 12.4 自动化 benchmark（必须工程化）
- 固定同一测试集输入：你的模型、XCMS、MS-DIAL、asari（可加 MZmine）。
- 自动输出：AUC/PR-AUC/F1、假阳性率、MAE/CV、RT 漂移鲁棒性曲线。
- 保证可复现：统一随机种子、统一阈值策略、统一统计检验脚本。

---

## 13. 你现在“先做数据集”的最小闭环清单（建议先执行）

在继续堆模型前，先把数据资产做成“可投稿版本”：

1. **数据来源台账**
   - 记录仪器厂商、采集模式、色谱条件、批次、样本类型。
2. **标注规则 SOP**
   - 明确真峰/假峰边界定义；多类别标签判据（拖尾、锯齿、共流出）。
3. **双人复核与一致率**
   - 至少抽取子集计算标注一致率（Cohen’s kappa 或简单一致率）。
4. **训练/验证/测试隔离策略**
   - 按批次或项目隔离，避免信息泄漏。
5. **数据集卡（Data Card）**
   - 数据规模、类别分布、仪器覆盖、排除标准、已知偏差。
6. **首版基线跑通**
   - 二分类先拿到稳定结果（history + ROC/PR + 混淆矩阵）。

> 当 1–6 完成后，再引入更复杂模型（Transformer/检测/分割）会更稳、更容易中稿。

---

## 14. 逐步执行清单（可直接照做）

### 阶段 A：任务与数据定义（1–2 天）
1. 冻结任务：`binary classification`（真假峰）
2. 冻结主模型：`ConvNeXt-Tiny + Attr Fusion`
3. 冻结输入规范：400x300、固定窗口、原始强度
4. 整理 `train/val/test.csv`（必须包含 `image,is_true_peak`）

### 阶段 B：训练与验证（2–4 天）
1. 先跑 Image-only
2. 再跑 Attr-only
3. 最后跑 Fusion 主模型
4. 固定最佳阈值（基于验证集 F1 或 Recall@Precision>=0.95）

### 阶段 C：测试与比较（2–4 天）
1. 在测试集导出 ROC、PR、混淆矩阵
2. 在 asari/pyopenms/xcms 三来源候选峰上统一打分
3. 报告高置信真峰比例与统计显著性

### 阶段 D：可解释输出（1–2 天）
1. 对每张图导出 `p_true_peak + final_score + rationale`
2. 生成 20–30 个代表性案例图（真峰/假峰/难例）
3. 完成“模型判断合理性”小节图文

### 阶段 E：论文成稿（持续）
1. 方法学：规则层 + 融合模型 + 综合评分
2. 结果：三基线对比 + 跨算法泛化
3. 讨论：优势、局限、后续检测扩展


