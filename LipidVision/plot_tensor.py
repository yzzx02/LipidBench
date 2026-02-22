import numpy as np
import matplotlib.pyplot as plt

# 读取你刚生成的张量文件 (请确保路径正确)
tensor_path = "data/output_tensors_v2/SM_d34_1.npy" 
try:
    tensor = np.load(tensor_path)
    num_channels = tensor.shape[2]

    # 设置画图
    fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4))
    if num_channels == 1:
        axes = [axes]
        
    titles = ["Channel 1 (MS1)"] + [f"Channel {i+2} (MS2)" for i in range(num_channels-1)]
    colors = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges'] # 给不同通道上不同的伪彩

    for i in range(num_channels):
        # 提取单独的通道并画图
        ax = axes[i]
        im = ax.imshow(tensor[:, :, i], aspect='auto', cmap=colors[i % len(colors)])
        ax.set_title(titles[i])
        ax.set_xlabel("Time (standard_rt index)")
        ax.set_yticks([]) # 隐藏Y轴，因为是复制的
        
    plt.suptitle("Hyperspectral Tensor Slices for SM d34:1", fontsize=16)
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"找不到文件 {tensor_path}，请确认路径是否正确。")