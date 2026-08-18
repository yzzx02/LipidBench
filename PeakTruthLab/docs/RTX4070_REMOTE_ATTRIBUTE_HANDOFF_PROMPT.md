# Prompt for the Codex on the mzML workstation

请在保存原始 mzML 文件的电脑上直接完成下面任务，不要只把命令发给我：

1. 进入或克隆 `yzzx02/LipidBench`，拉取并切换到分支 `agent/rtx4070-final-merge-split`。
2. 完整读取 `PeakTruthLab/docs/RTX4070_REMOTE_ATTRIBUTE_HANDOFF.md`、`PeakTruthLab/datasets/rtx4070_attribute_handoff_20260814/handoff_manifest.json` 和 `source_inventory.csv`，严格按文档顺序执行。
3. 由你检查环境、定位并核对 29 个原始 mzML、安装缺少的 Python 依赖、运行 `PeakTruthLab/scripts/data_prep/compute_rtx4070_attribute_handoff.py`、检查 QC 和校验和。
4. 任务范围仅是补齐冻结旧最终数据的属性：15,317 个 Seed jobs 和 18,580 个 peak jobs。不得修改人工标签、峰边界、legacy 13 属性、Train/Val/Test 划分或算法；不得合并新 4,500 条数据；不得启动训练或用 Test 调参。
5. 只有 `remote_attribute_qc.json` 的 `status` 为 `ok`，且行数、唯一 job_id、29 个数据源、回归检查和 SHA-256 全部通过，才可视为完成。否则停止并报告具体阻塞，不要伪造、补值或绕过检查。
6. 成功后新建分支 `codex/rtx4070-oldfinal-16attr-results`，只上传五个最终结果文件到 `PeakTruthLab/results/rtx4070_oldfinal_attribute_results_20260814/`：两个 16 属性 CSV、`source_resolution.csv`、`remote_attribute_qc.json`、`SHA256SUMS.txt`。不要上传原始 mzML 或中间 `parts`。
7. 提交并推送，创建以 `agent/rtx4070-final-merge-split` 为 base 的 Pull Request；PR 中写清命令、环境、backend、源目录、耗时、行数、QC、缺失/非有限值统计和输出哈希。把 PR 链接发给我后停止，不继续合并、划分或训练。
