import numpy as np

"""
This file is for printing out the content of the dataset
"""

npy_path = './amazon/train.npy'
output_path = 'structure_user_item_train_dict.txt'

data = np.load(npy_path, allow_pickle=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f"Type: {type(data)}\n")
    if hasattr(data, 'shape'):
        f.write(f"Shape: {data.shape}\n")
    if isinstance(data, dict):
        f.write(f"Dict keys (first 20): {list(data.keys())[:20]}\n")
        f.write("First 20 items:\n")
        for idx, (k, v) in enumerate(data.items()):
            if idx >= 20:
                break
            f.write(f"{k}: {v}\n")
    elif hasattr(data, '__len__'):
        f.write("First 100 elements:\n")
        for i in range(min(100, len(data))):
            f.write(f"{i}: {data[i]}\n")
        f.write("\nLast 5 elements:\n")
        for i in range(len(data)-5, len(data)):
            f.write(f"{i}: {data[i]}\n")
    else:
        f.write(f"Data: {repr(data)}\n")