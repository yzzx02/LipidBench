# 系统设计与开发说明书：LipidVision-DIA (脂质组学多通道视觉反褶积框架)

## 1. AI 助手角色定义与项目背景
**你的角色：** 你是一位精通 Python 数据工程、PyTorch 计算机视觉，且对液质联用 (LC-MS) 数据处理和生物信息学有深入研究的资深架构师。
**项目名称：** LipidVision-DIA
**技术栈要求：** 严格基于 Python 3.9.10。核心依赖库：`numpy`, `scipy`, `pyteomics`, `torch`, `torchvision`, `matplotlib`。
**核心目标：** [Image of Data-Independent Acquisition (DIA) mass spectrometry workflow] 构建一个自动化流水线。该流水线需要读取数据独立型采集 (DIA/SWATH) 的脂质组学数据 (`.mzML` 格式)，提取连续的 MS1 和 MS2 信号，利用数学插值解决扫描时间点不对齐的问题，并最终将它们构建为 3D 多通道张量 (类似 RGB 图像)。生成的张量将用于下游的深度学习实例分割模型，以解决脂质同分异构体共流出时的拆分难题。

## 2. 全局架构与目录结构
请在生成代码时，严格遵循并假设以下目录结构：
```text
LipidVision-DIA/
├── data/                   
├── src/                    
│   ├── data_parser/        # 负责解析 mzML 文件并提取 EIC (提取离子色谱图)
│   │   ├── __init__.py
│   │   └── mzml_reader.py  
│   ├── tensor_builder/     # 负责多通道信号的插值对齐与张量构建
│   │   ├── __init__.py
│   │   └── rgb_encoder.py  
│   ├── vision_model/       # 实例分割网络 (如 Mask R-CNN，暂不实现)
│   └── post_processor/     # 掩膜 (Mask) 映射与峰面积定量积分
└── requirements.txt

核心模块开发任务 (具体执行要求)
任务 1：实现 src/data_parser/mzml_reader.py
目标： 从 .mzML 文件中提取高质量的提取离子色谱图 (EIC) 序列。
核心要求：

使用 pyteomics.mzml 库进行高效解析。

输入参数： mzml_path (文件路径, str), target_mz (目标质荷比, float), mz_tol_ppm (质量偏差容忍度, float), rt_range (保留时间窗口元组: start_min, end_min), ms_level (质谱层级, int: 1 或 2)。

算法逻辑：

根据 target_mz 和 mz_tol_ppm 动态计算 mz_min 和 mz_max。

遍历谱图。根据给定的 ms level 和 scan start time (保留时间) 过滤无关数据。

必须使用 Numpy 向量化操作，将落在 [mz_min, mz_max] 窗口内的所有 m/z 对应的 intensity (强度) 进行加和。

输出： 返回一个包含两个一维 Numpy 数组的元组：(rt_array, intensity_array)。

任务 2：实现 src/tensor_builder/rgb_encoder.py
目标： 将异步扫描的 MS1 和 MS2 散点序列，强制对齐并堆叠为一个统一的 3D 视觉张量。
核心要求：

输入参数： rt_range (窗口元组), ms1_data (元组), ms2_channel_1_data (元组), ms2_channel_2_data (元组), num_pixels (图像宽度/时间点数，默认 128)。

算法逻辑：

创建标准伪时间轴： 生成一个均匀分布的 X 轴：standard_rt = np.linspace(rt_range[0], rt_range[1], num_pixels)。

插值对齐算法： 使用 scipy.interpolate.interp1d(kind='linear', bounds_error=False, fill_value=0)，将原始的 rt_array 和 intensity_array 映射到 standard_rt 上。

独立归一化 (极其关键)： 必须对每个插值后的通道进行独立的 Min-Max 归一化。公式：ch_norm = ch_interp / np.max(ch_interp)。注意处理除以零的异常情况。

张量堆叠： 将归一化后的三个 1D 数组堆叠成 (num_pixels, 3) 的形状。然后使用 np.tile 沿 Y 轴广播，形成最终的 2D 图像张量，形状应为 (Height, Width, Channels)，例如 (128, 128, 3)。

输出： 返回 (standard_rt, image_tensor)。

任务 3：了解 src/post_processor/quantifier.py 的设计意图 (无需在此次生成代码)
目标： 将视觉模型预测的 Mask 还原回原始 MS1 信号进行精确的峰面积积分。
预设逻辑：
1. 接收视觉模型输出的 2D 布尔 Mask，沿时间轴折叠为 1D 数组。
2. 找到该掩膜在伪时间轴 (standard_rt) 上的起点和终点坐标。
3. 将这些坐标反向映射回真实 MS1 数据的真实保留时间 (rt_array)。
4. 使用 scipy.integrate.simps 对原始 MS1 强度计算积分面积 (AUC)。

4. 强制代码规范 (AI 必须遵守)
类型提示 (Type Hinting)： 必须严格使用 Python 3.9 的类型提示规范 (如 List, Tuple, Optional, np.ndarray)。

文档字符串 (Docstrings)： 所有类和函数必须使用标准的 Google 风格中文注释说明。

异常处理机制： 如果在指定的保留时间窗口内没有找到该脂质 (即 mzml_reader 返回空列表)，interp1d 插值器必须能优雅地处理并返回全零数组，绝对不能抛出报错打断流水线。

性能优先： 在解析和堆叠张量时，绝对禁止使用低效的 Python for 循环处理底层数组，全面采用 numpy 向量化操作。

5. AI 首要执行动作 (First Action Required from AI)
请在回复的开头简要确认你已完全理解本《系统设计说明书》的要求。
随后，请直接、完整地生成 src/data_parser/mzml_reader.py 和 src/tensor_builder/rgb_encoder.py 这两个文件的代码。
在 rgb_encoder.py 文件的末尾，必须包含一个 if __name__ == "__main__": 测试代码块，使用模拟生成的 numpy 数组数据 (Dummy Data) 来测试并验证你写的这套处理流程是否能正常运行并打印出 Tensor 的形状。