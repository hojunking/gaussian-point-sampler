import numpy as np
import matplotlib.pyplot as plt

def load_pointcept_data(data_dir):
    """Pointcept 데이터를 로드 (가정된 함수)"""
    data = {}
    for key in ['coord', 'color', 'normal', 'segment20', 'segment200', 'instance']:
        file_path = f"{data_dir}/{key}.npy"
        data[key] = np.load(file_path)
    return data

def compute_label_distribution(labels, ignore_index=-1, max_label=20):
    """라벨 분포 계산"""
    distribution = {}
    valid_labels = labels[labels != ignore_index]
    for label in range(-1, max_label):
        count = np.sum(labels == label)
        distribution[label] = count / len(labels)
    return distribution

def compare_datasets(previous_dir, current_dir, scene_name="scene0010_00"):
    """
    이전 데이터셋과 현재 데이터셋 비교.
    
    Args:
        previous_dir (str): 이전 데이터셋 경로.
        current_dir (str): 현재 데이터셋 경로.
        scene_name (str): 장면 이름.
    """
    # 데이터 로드
    previous_data = load_pointcept_data(previous_dir)
    current_data = load_pointcept_data(current_dir)

    # 포인트 수 비교
    prev_points = len(previous_data['coord'])
    curr_points = len(current_data['coord'])
    print(f"\nPoint Count Comparison for {scene_name}:")
    print(f"  Previous: {prev_points}")
    print(f"  Current: {curr_points}")
    print(f"  Difference: {curr_points - prev_points}")

    # 라벨 분포 비교
    print("\nLabel Distribution Comparison:")
    for label_key in ['segment20', 'segment200', 'instance']:
        max_label = 20 if label_key == 'segment20' else 200 if label_key == 'segment200' else 1000
        prev_dist = compute_label_distribution(previous_data[label_key], max_label=max_label)
        curr_dist = compute_label_distribution(current_data[label_key], max_label=max_label)

        print(f"\n{label_key} Distribution:")
        print(f"  Previous -1 ratio: {prev_dist[-1]:.4f}")
        print(f"  Current -1 ratio: {curr_dist[-1]:.4f}")
        print(f"  Difference: {curr_dist[-1] - prev_dist[-1]:.4f}")

        # 히스토그램 시각화
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(list(prev_dist.keys()), list(prev_dist.values()))
        plt.title(f"Previous {label_key} Distribution")
        plt.xlabel("Label")
        plt.ylabel("Ratio")
        plt.subplot(1, 2, 2)
        plt.bar(list(curr_dist.keys()), list(curr_dist.values()))
        plt.title(f"Current {label_key} Distribution")
        plt.xlabel("Label")
        plt.ylabel("Ratio")
        plt.tight_layout()
        plt.show()

    # 색상 및 법선 통계 비교
    print("\nColor and Normal Statistics Comparison:")
    for key in ['color', 'normal']:
        prev_mean = np.mean(previous_data[key], axis=0)
        prev_std = np.std(previous_data[key], axis=0)
        curr_mean = np.mean(current_data[key], axis=0)
        curr_std = np.std(current_data[key], axis=0)
        print(f"\n{key}:")
        print(f"  Previous Mean: {prev_mean}")
        print(f"  Current Mean: {curr_mean}")
        print(f"  Mean Difference: {curr_mean - prev_mean}")
        print(f"  Previous Std: {prev_std}")
        print(f"  Current Std: {curr_std}")
        print(f"  Std Difference: {curr_std - prev_std}")

# 사용 예시
previous_dir = "/home/knuvi/Desktop/song/Pointcept/data/pdistance00005_scale05_keep-dup/train/scene0010_00"
current_dir = "/home/knuvi/Desktop/song/Pointcept/data/pdistance00005_scale05_modi-attr/train/scene0010_00"
compare_datasets(previous_dir, current_dir, "scene0010_00")