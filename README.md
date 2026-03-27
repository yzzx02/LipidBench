# LipidBench

这是一个用于脂质算法评估、EIC 数据构建与模型训练的工作区。

## 必读规则（每次动代码前）
1. 每次开始改动前，必须先阅读：
	- `PeakTruthLab/docs/CONVNEXT_FUSION_EXECUTION_PLAN.md`
	- 本 README 的本章节
2. 当前 EIC 最终数据集**唯一目录**：`PeakTruthLab/datasets/eic_images_flat/`
3. 当前最终特征表**唯一文件**：`PeakTruthLab/datasets/feature_table_final_10000.csv`
4. 严禁把 `results/` 作为最终训练数据集目录；`results/` 仅用于过程产物。
5. 任何对最终 CSV 的删改，必须同步更新 `eic_images_flat`（保证 `Feature_ID` 与 `png/json` 一一对应）。
6. 每次运行任务结束后，必须立即执行清理：删除本次产生的中间日志、临时脚本、临时调试产物与缓存文件；该步骤为强制收尾步骤，不得跳过。
7. 执行人（含 AI 代理）必须在每次任务完成前主动完成清理与复核，确保仓库只保留最终数据集与必要报告，不保留无关临时文件。
8. 任何会修改最终主表 `PeakTruthLab/datasets/feature_table_final_10000.csv` 的脚本，在写入前必须自动备份到 `PeakTruthLab/datasets/backups/`，备份文件名需包含时间戳，禁止覆盖历史备份。

## 运行后强制清理清单
1. 清理中间日志：删除本次运行生成的临时 `*.log`（保留明确约定的最终报告除外）。
2. 清理临时脚本与调试输出：删除一次性脚本、临时 CSV/JSON、重试残留文件。
3. 清理缓存：删除 `__pycache__/` 与临时缓存文件。
4. 结果收敛复核：确认最终仅使用以下目录与文件作为训练数据源：
	- `PeakTruthLab/datasets/eic_images_flat/`
	- `PeakTruthLab/datasets/feature_table_final_10000.csv`
5. 备份复核：若本次任务涉及主表改写，确认 `PeakTruthLab/datasets/backups/` 已生成新的时间戳备份文件。

## 目录角色（简版）
- `PeakTruthLab/datasets/eic_images_flat/`：最终训练图像与 json 元数据。
- `PeakTruthLab/datasets/feature_table_final_10000.csv`：最终标签/属性主表。
- `results/`：中间分析输出，不作为最终数据集来源。
