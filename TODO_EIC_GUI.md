# LipidBench：EIC 提取 + 400×300 绘图 + PySide6 GUI（不含 ROI 检测/预测）

目标：在 LipidBench 里实现一个最小可用流程：从“feature 表 + mzML”生成每个 feature 的 EIC（XIC）曲线，并以固定像素 400×300 导出图片；再做一个 PySide6 小界面用于选择输入/运行/浏览结果。

## 0. 约束与约定
- **不做 ROI 检测**：不跑 QuanFormer/DETR 推理，不生成 bbox。
- **统一图像尺寸**：所有导出的 EIC 图片固定为 **400×300**（用于后续你要做训练时保持一致坐标系）。
- **输入数据**：
  - mzML：来自 `config.yaml` 的 `paths.input_dir`（默认 `./data/raw_mzML`）。
  - feature 表：CSV（优先复用现有 `Results/pyopenms/pyopenms_features.csv` 格式）。
- **RT 单位**：feature 表里 `RT/RTmin/RTmax` 当前在 LipidBench 的 loader 里会被规范为 **分钟**（见 `src/utils/data_io.py: load_pyopenms_results()`）。mzML 中 spectrum 的 RT 在 OpenMS 内部通常是 **秒**，实现时要显式换算/检测。

## 0.1 依赖与环境（建议先锁定）
LipidBench 当前根目录下没有 `requirements.txt`，建议你在实现前明确这些依赖：
- 必需：`pyopenms`、`numpy`、`pandas`、`matplotlib`、`PySide6`
- 可选（用于平滑）：`scipy`
- 可选（用于验证像素尺寸/快速读图）：`Pillow`

## 1) EIC 提取（pyOpenMS + feature 表）
### 1.1 明确 feature 表最小字段规范
最小必需列（CSV）：
- `Feature_ID`（可选；若没有也可以按行号生成）
- `mz`（目标 m/z）
- `RTmin`、`RTmax`（分钟；若缺失则用 `RT ± rt_window_min/2` 兜底）

建议兼容 LipidBench 已有 pyOpenMS 输出（示例在 `Results/pyopenms/pyopenms_features.csv`）：
- `Feature_ID,mz,mzmin,mzmax,RT,RTmin,RTmax,intensity`

### 1.2 选择 EIC 提取策略（保持可配置，但默认简单）
实现两个策略，默认用 A（更贴近 QuanFormer）：
- A. **最近点强度 + ppm 判定**（QuanFormer 同款思想）：
  - 对每个 MS1 scan：在 spectrum 的 m/z 数组中找离 `mz_target` 最近的点 `mz_nearest`；
  - 若 `|mz_nearest - mz_target| / mz_target * 1e6 <= ppm` 则取该点 intensity，否则 intensity=0。
- B. **ppm 窗口求和**（备选）：
  - 取落在 `[mz_target - tol_da, mz_target + tol_da]` 范围内的所有点 intensity 求和。

参数来源：优先复用 `config.yaml/common_params/mz_tolerance_ppm`（默认 10 ppm）。

### 1.3 实现核心提取函数（建议新增模块）
新增模块（建议）：
- `src/eic/extract_eic_pyopenms.py`

建议函数签名：
- `extract_eic_for_feature(exp, mz_target, ppm, rt_min_min, rt_max_min, ms_level=1) -> pd.DataFrame`
  - 返回列：`rt_min`（分钟）、`intensity`
- `extract_eic_batch(mzml_path, features_df, ppm, strategy, output_dir, smooth_sigma=None)`

实现要点：
- 用 `pyopenms.MzMLFile().load(str(mzml_path), exp)` 读取。
- 遍历 `exp` 的 spectrum：只处理 `MSLevel==1`。
- RT 获取：`rt_sec = spectrum.getRT()`；换算 `rt_min = rt_sec / 60.0`。
- 用 numpy 加速最近点查找：对 `mzs` 用 `np.searchsorted`。

### 1.4 输出规范
对每个 mzML 文件输出：
- `results/eic/<mzml_stem>/eic_curves.csv`（可选：合并存储所有 feature 的曲线，或每个 feature 单独一个 CSV）
- `results/eic/<mzml_stem>/images/<Feature_ID>.png`（或 jpg）

最小验收：给定一个 mzML + 一个 feature CSV，能够生成至少 1 张 EIC 图。

## 2) ROI 检测（暂不需要）
- 明确：当前阶段不实现任何 bbox/ROI 生成、预测、quantify。
- 为后续扩展留接口：输出文件夹结构里保留 `images/`，未来可以把检测模型的输入直接指向这里。

## 3) EIC 图像绘制（固定 400×300）
### 3.1 绘图规格（严格锁定像素）
使用 Matplotlib：
- `fig = Figure(figsize=(4, 3), dpi=100)` → **400×300**
- 单条折线：RT（分钟） vs intensity
- 标题建议：`<mzml_stem> | <Feature_ID> | m/z=<mz>`
- 轴标签：`RT (min)`、`Intensity`

### 3.2 平滑（可选）
- 可选参数：`smooth_sigma`（类似 QuanFormer 的 gaussian_filter）
- 若不想引入 SciPy，可先用简单移动平均；否则用 `scipy.ndimage.gaussian_filter1d`。

### 3.3 建议新增模块
- `src/eic/plot_eic.py`
  - `plot_eic(rt_min, intensity, out_path, width_px=400, height_px=300, dpi=100, ...)`

最小验收：输出图片像素严格为 400×300（用 Pillow/Qt 加载验证）。

## 4) PySide6 GUI（先不做预测）
### 4.1 最小界面功能
- 输入选择：
  - 选择 mzML 文件（或文件夹）
  - 选择 feature CSV
  - 选择输出目录（默认 `./results/eic`）
- 参数区（最少 3 个）：
  - `ppm`（默认来自 config 10）
  - `strategy`（nearest / sum_window）
  - `smooth_sigma`（可空）
- 按钮：`Run`、`Open Output Folder`
- 结果浏览：
  - 列表：Feature_ID
  - 右侧图片区：显示对应的 400×300 图（允许缩放显示，但不要改变输出文件本身）

### 4.2 线程/进度
- 提取和绘图应放到 `QThread` 或 `QtConcurrent`，避免卡 UI。
- 最小进度：显示当前处理到的 Feature_ID / 总数。

### 4.3 代码落点建议
- 新增：`src/gui/eic_viewer.py`（或 `gui/main.py`）
- 入口：可在项目根 `main.py` 增加一个 `--gui` 选项（可选；也可单独脚本启动）。

最小验收：GUI 能跑通一次生成，并在界面里点选 Feature_ID 查看图片。

## 5. 集成到现有 LipidBench（可选但推荐）
- 在 `config.yaml` 增加：
  - `paths.eic_output: ./results/eic`
  - `parameters.eic: { ppm: 10.0, strategy: nearest, smooth_sigma: null }`
- 在 `main.py` 的 registry 里增加一个 `eic` runner（如果你希望和 xcms/pyopenms 一样从 CLI 调）。

## 6. 开发顺序（建议）
1) 先实现 `extract_eic_batch()`（命令行可跑）
2) 再实现 `plot_eic()` 并固定 400×300
3) 最后做 GUI（复用第 1/2 步函数）

