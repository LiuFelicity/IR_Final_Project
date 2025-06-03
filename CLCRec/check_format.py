import os
import numpy as np

# 指定資料夾路徑
folder_path = './Data/movielens'

# 列出資料夾中所有 .npy 文件
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# 檢查每個 .npy 文件的格式
for npy_file in npy_files:
    file_path = os.path.join(folder_path, npy_file)
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"File: {npy_file}")
        print(f"Shape: {data.shape}")
        print(f"Data Type: {data.dtype}")
        print(f"Sample Data: {data[:5]}\n")  # 顯示前 5 筆資料
    except Exception as e:
        print(f"Error loading {npy_file}: {e}")