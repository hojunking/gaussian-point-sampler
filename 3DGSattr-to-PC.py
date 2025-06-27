# PC-3DGS_fusion.py

import os, json
import argparse
import numpy as np
from tqdm import tqdm

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes

def GSattr_to_PC(pointcept_dir, path_3dgs, output_dir, k_neighbors=10):
    # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    
    # 2. 3DGS 데이터 로드
    points_3dgs, raw_features_3dgs = load_3dgs_data(path_3dgs)

    # 3. 3DGS 속성 전처리
    #print("Preprocessing 3DGS attributes...")
    features_3dgs = preprocess_3dgs_attributes(raw_features_3dgs)
    
    #print("Augmenting Pointcept points with 3DGS attributes...")
    features_pointcept  = augment_pointcept_with_3dgs_attributes(
        points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors
    )

    # 9. Pointcept 포맷으로 저장 (.npy 파일)
    save_dict = {
        'features': features_pointcept.astype(np.float32),
    }
    
    os.makedirs(output_dir, exist_ok=True)
    for key, value in save_dict.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), value)
        print(f"Saved {key}.npy to {output_dir}")

def process_single_scene(scene, split, input_root, output_root, path_3dgs_root, k_neighbors=5):
    try:
        pointcept_dir = os.path.join(input_root, split, scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        output_dir = os.path.join(output_root, split, scene)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"3DGS PLY file not found: {path_3dgs}, required for '3dgs' method.")
            return

        GSattr_to_PC(
            pointcept_dir, path_3dgs, output_dir, k_neighbors
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def process_scenes(input_root, output_root, split, scene_list, path_3dgs_root, k_neighbors=5, num_workers=1):
    print(f"Processing {split} scenes (Total: {len(scene_list)} scenes)")
    # 병렬 처리는 현재 코드에서 제외되었으나, 필요시 Pool 등을 사용하여 구현할 수 있습니다.
    for scene in tqdm(scene_list, desc=f"Processing {split} scenes", unit="scene"):
        process_single_scene(
            scene, split, input_root, output_root, path_3dgs_root, k_neighbors
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate boundary pseudo-labels for ScanNet scenes.")
    parser.add_argument(
        "--output_root",
        default="/home/knuvi/Desktop/song/point/data/scannet_boundary", # 출력 경로 변경 권장
        help="Output path for processed data",
    )
    parser.add_argument(
        "--data_type",
        default="samples100",
        choices=["full", "samples100"],
        help="Dataset type: 'full' or 'samples100'",
    )
    # ... (기존 다른 인자들은 그대로 유지)
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers for parallel processing (currently not implemented for simplicity)",
    )

    args = parser.parse_args()

    # Config 파일 로드
    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_root = config['input_root']
    path_3dgs_root = config['path_3dgs_root']
    meta_root = config['meta_root']
    prune_params = config['prune_params']
    k_neighbors = prune_params['k_neighbors']
    

    # Data type에 따른 메타 파일 선택
    if args.data_type == 'full':
        train_meta_file = "scannetv2_train.txt"
        val_meta_file = "scannetv2_val.txt"
    else:
        train_meta_file = "train100_samples.txt"
        val_meta_file = "valid20_samples.txt"

    # Train split 처리
    train_meta_file_path = os.path.join(meta_root, train_meta_file)
    if not os.path.exists(train_meta_file_path):
        raise FileNotFoundError(f"Train meta file not found: {train_meta_file_path}")
    
    with open(train_meta_file_path) as f:
        train_scenes = f.read().splitlines()

    process_scenes(
        input_root, args.output_root, 'train', train_scenes, path_3dgs_root, 
        k_neighbors, args.num_workers
    )

    # Val split 처리
    val_meta_file_path = os.path.join(meta_root, val_meta_file)
    if not os.path.exists(val_meta_file_path):
        raise FileNotFoundError(f"Val meta file not found: {val_meta_file_path}")
    
    with open(val_meta_file_path) as f:
        val_scenes = f.read().splitlines()

    process_scenes(
        input_root, args.output_root, 'val', val_scenes, path_3dgs_root,
        k_neighbors, args.num_workers
    )