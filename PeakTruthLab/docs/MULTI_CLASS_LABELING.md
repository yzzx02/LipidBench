# PeakTruthLab 多类别真峰标注（small_trainset）

你现在的目录 `PeakTruthLab/datasets/small_trainset` 是：
- 每张图一个 `xxx.jpeg`
- 同名一个 `xxx.json`（LabelMe 格式）

当前训练脚本使用 Faster R-CNN 做“框 + 类别”学习。

## 1) 可以怎么标（推荐方案）

### 真假峰（是否存在真峰框）
- **真峰**：保留一个矩形框（rectangle），框住真峰所在的 RT 区间。
- **假峰**：把 `shapes` 清空（即不画框），保存后 JSON 里 `"shapes": []`。

这会让模型学到“有框=真峰，没框=假峰”。

### 真峰细分类型（互斥单选）
把矩形框的 `label` 改成下面之一（见 `PeakTruthLab/configs/labelme_labels.txt`）：
- `True_Peak`（普通真峰/不确定类型）
- `True_Peak_HQ`（高质量峰）
- `True_Peak_Coelution`（共流出峰）
- `True_Peak_Jagged`（锯齿峰）
- `True_Peak_Tailing`（拖尾峰）

约定（建议）：
- 每张图最多 2 个框（用于 2min 窗口里出现两个可分的真峰时分开框住）。
- 仍按“单框单类别”的原则：每个框只选一个最主要类别。

也支持中文别名（见 `PeakTruthLab/configs/label_map.json` 的 `aliases`）。

> 如果你希望“一个真峰同时有多个属性”（多标签），当前 MVP 会更复杂（需要多任务/多标签 head）。
> 现阶段可以用一个折中：优先标注最显著的形态（例如拖尾/共流出优先于 HQ）。

## 1.1) 单框单类别：简明判别口径（建议）

为了让不同人标注更一致，建议按“最主要形态特征”选一个类别：

- `True_Peak_HQ`：单峰、尖、对称、峰顶平滑，基线干净。
- `True_Peak_Coelution`：出现明显肩峰/双峰趋势（结构性重叠），而不是随机噪声造成的抖动。
- `True_Peak_Jagged`：峰顶粗糙、锯齿/抖动明显（多个小尖/不规则平台），但没有清晰可分的双峰结构。
- `True_Peak_Tailing`：一侧（多为右侧）拖出明显长尾，回落很慢。
- `True_Peak`：真峰但不确定属于哪类/不想细分时用。

关于“双峰/重叠峰”的推荐做法：
- 如果两个峰明显重叠、只表现为肩峰/双顶趋势：更建议用 1 个框整体框住，标 `True_Peak_Coelution`。
- 如果在 2min 窗口内出现两个峰且足够可分：可以画 2 个框分别框住两个峰，各自标 `True_Peak` 或 `True_Peak_HQ`（按分离质量决定）。

如果你发现某种特殊形态很常见，建议先用现有类别里最贴近的标；
确认样本量足够且标注一致性高时，再新增一个类别。

## 2) 如何用 LabelMe 快速开始

1. 安装（如果你没装过）：
   - `pip install labelme`
2. 启动：
  - 推荐（带 labels 列表）：
    - `labelme --labels PeakTruthLab\configs\labelme_labels.txt`
3. 打开目录：
   - `Open Dir` 选择 `PeakTruthLab/datasets/small_trainset`
4. 导入标签列表：
  - 如果你用上面的启动方式（带 `--labels`），这一步可以跳过
  - 注意：不要用“Open(打开标注文件)”去打开 `labelme_labels.txt`，它不是 JSON
5. 标注规则：
   - 真峰：画一个 rectangle，label 选对应类型
   - 假峰：删除 rectangle，让 shapes 为空

## 3) 训练（多类别）

训练脚本已支持读取标签映射文件：

```bash
python PeakTruthLab/scripts/train_model.py \
  --data-root D:\\LipidBench\\PeakTruthLab\\datasets\\small_trainset \
  --label-map D:\\LipidBench\\PeakTruthLab\\configs\\label_map.json \
  --epochs 20 --batch-size 4
```

可视化预测：

```bash
python PeakTruthLab/scripts/eval_mvp.py \
  --image-root D:\\LipidBench\\PeakTruthLab\\datasets\\small_trainset \
  --label-map D:\\LipidBench\\PeakTruthLab\\configs\\label_map.json \
  --weights D:\\LipidBench\\PeakTruthLab\\models\\fasterrcnn_mvp.pth \
  --out-root D:\\LipidBench\\PeakTruthLab\\outputs \
  --score-threshold 0.5
```

标注检查（看各类数量、是否有未知 label）：

```bash
python PeakTruthLab/scripts/inspect_labels.py \
  --data-root D:\\LipidBench\\PeakTruthLab\\datasets\\small_trainset \
  --label-map D:\\LipidBench\\PeakTruthLab\\configs\\label_map.json
```
