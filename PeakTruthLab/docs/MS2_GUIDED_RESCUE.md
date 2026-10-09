# MS2 引导补峰接口

软件只负责补峰、候选验证和结果导出，不进行差异分析。

流程：鉴定表的前体 m/z + MS2 RT → 原始 mzML 的 MS1 EIC → 图像检测 →
边界修正 → 新候选的 16 项属性 → 候选真假分类 → 补回峰表和数量统计。

## 输入约定

每行表示一个样本中的一个鉴定目标。同一个目标在不同样本中可有多行。

| 字段 | 必需 | 含义 |
| --- | --- | --- |
| `sample_id` | 是 | 样本编号，同一编号只对应一个原始文件 |
| `mzml_path` | 是 | 原始 mzML 路径；相对路径按表格目录或 `--mzml-root` 解析 |
| `precursor_mz` | 是 | 前体离子的 m/z，不能填碎片 m/z 或中性质量 |
| `ms2_rt` | 是 | MS2 扫描 RT；默认分钟，可明确选择秒 |
| `has_ms1` | 是 | 原始特征表中是否已经有对应 MS1 特征 |
| `has_ms2` | 否 | 是否有 MS2；默认 true，false 的条目跳过 |
| `target_id` | 否 | 鉴定/目标编号；缺省自动生成行编号 |

`has_ms1=false` 表示传统软件没有提取出该特征。原始 mzML 中仍须有可恢复的
MS1 信号。如果文件只有 MS2 扫描或目标没有原始 MS1 信号，不会生成补回峰。
存在性字段接受 true/false、1/0、yes/no、有/无。其余鉴定信息保留在 `targets.csv`。

CSV、TSV、XLSX 均可读取；`requirements.txt` 包含 XLSX 所需的 `openpyxl`。
如果鉴定表的列名不同，可通过 JSON 指定映射，例如：

```json
{"precursor_mz": "Precursor m/z", "ms2_rt": "MS2 RT", "has_ms1": "已有MS1"}
```

MS2 RT 用于窗口定位和候选关联；峰顶取自原始 MS1 EIC，而不是直接使用 MS2 RT。
该接口使用用户提供的 MS2 注释，不执行谱库检索或重新鉴定。

## 命令行

先生成输入表模板，不需要模型或原始文件：

```powershell
python -m chromapeak rescue-ms2 --write-template targets.csv
```

运行补峰（权重和配套文件的路径按实际下载位置填写）：

```powershell
python -m chromapeak rescue-ms2 `
  --table targets.csv --output results/ms2_rescue_run1 `
  --detection-checkpoint weights/best_detection.pt `
  --classification-checkpoint weights/best_seed.pt `
  --preprocessor weights/attribute_preprocessing.json `
  --selection weights/selection_before_test.json `
  --rt-unit min
```

`--selection` 使用正式 Val 选择记录中的检测和分类阈值；也可显式给出
`--detection-threshold` 和 `--classification-threshold`。接口不会猜测阈值。
两个权重可以不同：第一轮使用检测权重，第二轮使用候选分类权重。
属性填补和标准化使用配套的 Train 参数，不在鉴定表上重新拟合。

默认使用同一个 EIC 绘图函数、480×480 画布、2 分钟窗口、原始强度纵轴、
10 ppm 的 nearest 提取。绘图参数和训练来源应保持一致，必要时可调整提取方法
和窗口。检测框通过实际坐标轴变换换算为 RT，修正后的分类框使用训练标注的
5% 横向/高度留白规则。每个新候选单独计算属性，不复用旧候选的属性。

MS2 与候选的默认关联容差是峰边界外 0.2 分钟，可用
`--ms2-rt-tolerance-min` 调整。多候选通过时优先选择包含 MS2 RT 的峰，再选择
峰顶最接近 MS2 RT 的候选；输出保留所有候选和通过数量供复核。
边界修正默认采用现有 guarded 方法，最多向检测边界外扩一个 MS1 扫描间隔。
至少需要三个扫描点。异常条目单独记录，其余条目继续运行。

## 输出

| 文件 | 内容 |
| --- | --- |
| `targets.csv` | 所有输入条目及状态、选中候选、关联峰编号、异常原因 |
| `candidates.csv` | 检测框、修正框、RT 边界、16 项属性、分类概率及 QC |
| `peaks.csv` | 去重后的复核通过峰和补回峰 |
| `rescued_peaks.csv` | 仅包含补回峰，可交给后续软件 |
| `summary.json` | 总体及各样本补峰数、状态统计、运行参数及模型来源 |
| `images/` | 按 MS2 RT 生成的 EIC 图像及坐标映射 JSON |

`rescued`：原表没有 MS1 特征，且通过检测、边界/信号 QC 和分类阈值。
`existing_validated`：原表已有特征，复核通过。
`duplicate_peak`：重复 MS2 条目关联到一个已经补回的峰，数量不重复增加。
`already_present`：关联的同一物理峰在其他输入条目中已经有 MS1 特征，不算新增。
没有检测、没有信号、分类未通过或运行异常的条目分别保留原因。

去重要求同一样本和文件、前体质量在提取容差内、峰顶相距不超过 0.02 分钟，
且 RT 区间 IoU ≥ 0.5。不同注释名仍保留在该峰的 `target_ids`，不会制造多个峰。
去重范围是本次传入的鉴定表；不自动检索另一张未传入的原始特征表。

`summary.json` 区分通过的缺失目标行数和去重后的补回峰数。
面积从未平滑的 MS1 EIC 梯形积分得到，单位为强度×分钟；它是本接口的积分值，
不代表与上游软件的面积定义相同。“补回”表示本工作流验证通过，不替代鉴定可信度评估。
每次使用新的输出目录，避免覆盖历史结果。

## Python 接口

```python
from lipidbench.workflows.ms2_rescue import (
    RescueOptions, read_identification_table, rescue_ms2_peaks,
)
from lipidbench.workflows.ms2_rescue_inference import TorchRescuePredictor

predictor = TorchRescuePredictor(
    "weights/best_detection.pt", "weights/best_seed.pt",
    "weights/attribute_preprocessing.json",
)
result = rescue_ms2_peaks(
    read_identification_table("targets.csv"), predictor,
    "results/ms2_rescue_run1",
    RescueOptions(detection_threshold=0.5, classification_threshold=0.5),
    base_dir="data/raw_mzML",
)
print(result.summary["unique_rescued_peaks"])
```

Python 示例中的 0.5 仅展示接口参数，实际运行应使用所选模型的配套阈值。
调用者也可实现 `RescuePredictor.detect` / `classify`，复用已有的模型服务。
