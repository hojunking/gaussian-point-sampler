import numpy as np
import os
import argparse
from tqdm import tqdm

def validate_scene(scene_dir):
    """
    단일 scene 디렉토리의 데이터를 검증.
    
    Args:
        scene_dir (str): 검증할 scene 디렉토리 경로.
    
    Returns:
        bool: 검증 성공 여부.
    """
    try:
        # 필수 파일 목록
        required_files = ['coord.npy', 'color.npy', 'normal.npy', 'segment20.npy', 'segment200.npy', 'instance.npy']
        
        # 파일 존재 여부 확인
        for file_name in required_files:
            file_path = os.path.join(scene_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Error: {file_name} not found in {scene_dir}")
                return False
        
        # 데이터 로드
        data = {}
        for file_name in required_files:
            file_path = os.path.join(scene_dir, file_name)
            data[file_name.split('.')[0]] = np.load(file_path)
        
        # 점 개수 확인
        num_points = len(data['coord'])
        for key in data:
            if len(data[key]) != num_points:
                print(f"Error: Point count mismatch in {scene_dir} - {key} has {len(data[key])} points, expected {num_points}")
                return False
        
        # 데이터 타입 확인
        expected_dtypes = {
            'coord': np.float32,
            'color': np.uint8,
            'normal': np.float32,
            'segment20': np.int64,
            'segment200': np.int64,
            'instance': np.int64
        }
        for key, expected_dtype in expected_dtypes.items():
            if data[key].dtype != expected_dtype:
                print(f"Error: Incorrect dtype in {scene_dir} - {key} has dtype {data[key].dtype}, expected {expected_dtype}")
                return False
        
        # 라벨 분포 확인
        print(f"\nLabel distribution for {scene_dir}:")
        unique_labels, counts = np.unique(data['segment20'], return_counts=True)
        print(f"segment20 labels: {dict(zip(unique_labels, counts))}")
        unique_labels, counts = np.unique(data['segment200'], return_counts=True)
        print(f"segment200 labels: {dict(zip(unique_labels, counts))}")
        unique_instances, counts = np.unique(data['instance'], return_counts=True)
        print(f"instance IDs: {dict(zip(unique_instances, counts))}")
        
        print(f"Validation passed for {scene_dir}: {num_points} points")
        return True
    
    except Exception as e:
        print(f"Error validating {scene_dir}: {e}")
        return False

def validate_data(root_dir, split):
    """
    주어진 루트 디렉토리와 split에 속한 모든 scene을 검증.
    
    Args:
        root_dir (str): 루트 디렉토리 (예: /home/knuvi/Desktop/song/Pointcept/data/scannet_merged).
        split (str): 'train' 또는 'val'.
    """
    split_dir = os.path.join(root_dir, split)
    if not os.path.exists(split_dir):
        print(f"Error: {split_dir} does not exist")
        return
    
    # scene 디렉토리 목록
    scenes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    print(f"Found {len(scenes)} scenes in {split_dir}")
    
    # 각 scene 검증
    success_count = 0
    for scene in tqdm(scenes, desc=f"Validating {split} scenes", unit="scene"):
        scene_dir = os.path.join(split_dir, scene)
        if validate_scene(scene_dir):
            success_count += 1
    
    print(f"\nValidation summary for {split}: {success_count}/{len(scenes)} scenes passed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Pointcept format data.")
    parser.add_argument(
        "--root_dir",
        default="/home/knuvi/Desktop/song/Pointcept/data/scannet_merged",
        help="Root directory of the processed data (default: /home/knuvi/Desktop/song/Pointcept/data/scannet_merged)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Split to validate (train or val, default: train)",
    )
    args = parser.parse_args()

    validate_data(args.root_dir, args.split)