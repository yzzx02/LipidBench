import pandas as pd

# 创建示例数据
df_original = pd.DataFrame({
    'A': [10, 20, 30, 40, 50],
    'B': [1, 2, 3, 4, 5]
}, index=[100, 101, 102, 103, 104])  # 自定义索引

print("=== 原始DataFrame ===")
print("数据:")
print(df_original)
print(f"索引: {df_original.index.tolist()}")
print()

# 使用 iloc[2:] 切片
df_slice = df_original.iloc[2:]
print("=== iloc[2:] 切片后（未重置索引）===")
print("数据:")
print(df_slice)
print(f"索引: {df_slice.index.tolist()}")
print("注意：索引仍然是 [102, 103, 104]，不是连续的 [0, 1, 2]")
print()

# 重置索引（drop=False，默认）
df_reset_keep = df_slice.reset_index()
print("=== reset_index() （保留原索引）===")
print("数据:")
print(df_reset_keep)
print(f"新索引: {df_reset_keep.index.tolist()}")
print("原索引变成了一列数据")
print()

# 重置索引（drop=True）
df_reset_drop = df_slice.reset_index(drop=True)
print("=== reset_index(drop=True) （丢弃原索引）===")
print("数据:")
print(df_reset_drop)
print(f"新索引: {df_reset_drop.index.tolist()}")
print("原索引被丢弃，新的索引是连续的 [0, 1, 2]")
print()

print("=== reset_index参数详解 ===")
print("reset_index(drop=False): 原索引变成新列，新索引从0开始")
print("reset_index(drop=True): 丢弃原索引，新索引从0开始")
print("drop=False 是默认值")
