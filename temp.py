import numpy as np


data_path = "/home/knuvi/Desktop/song/Pointcept/data/3dgs_attribute_merge_pdis00001/train/scene0478_01"
coord = np.load(f"{data_path}/coord.npy")
color = np.load(f"{data_path}/color.npy")
features = np.load(f"{data_path}/features.npy")
print("Coord shape:", coord.shape)  # 예: (20221, 3)
print("Color shape:", color.shape)  # 예: (20221, 3)
print("Features shape:", features.shape)  # 예: (26318, 8) -> 문제 발생
assert len(coord) == len(color) == len(features), "Point count mismatch in raw data!"