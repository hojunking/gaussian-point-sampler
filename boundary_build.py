# PC-3DGS_fusion.py

import os, json
import argparse
import numpy as np
from tqdm import tqdm

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes
from boundary_utils import boundary_labeling

def boundary_labeling_with_3dgs(pointcept_dir, path_3dgs, output_dir, prune_methods=None, prune_params=None, k_neighbors=10):
    # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    
    # 2. 3DGS 데이터 로드
    points_3dgs, normals_3dgs, raw_features_3dgs = load_3dgs_data(path_3dgs)

    # 3. 3DGS 속성 전처리
    print("Preprocessing 3DGS attributes...")
    features_3dgs = preprocess_3dgs_attributes(raw_features_3dgs)
    
    # 6. 3DGS-attr transfer
    print("Augmenting Pointcept points with 3DGS attributes...")
    features_pointcept  = augment_pointcept_with_3dgs_attributes(
        points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors
    )
    
    boundary = boundary_labeling(
        points_pointcept, features_pointcept, prune_methods, prune_params
    )

    # 9. Pointcept 포맷으로 저장 (.npy 파일)
    save_dict = {
        'boundary': boundary.astype(np.int64),
    }
    os.makedirs(output_dir, exist_ok=True)
    for key, value in save_dict.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), value)
        print(f"Saved {key}.npy to {output_dir}")

def process_single_scene(scene, split, input_root, output_root, path_3dgs_root, prune_methods, prune_params, k_neighbors=5):
    try:
        pointcept_dir = os.path.join(input_root, split, scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        output_dir = os.path.join(output_root, split, scene)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"3DGS PLY file not found: {path_3dgs}")
            return

        boundary_labeling_with_3dgs(
            pointcept_dir, path_3dgs, output_dir, prune_methods, prune_params, k_neighbors
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def process_scenes(input_root, output_root, split, scene_list, path_3dgs_root, prune_methods=None, prune_params=None, k_neighbors=5, num_workers=1):

    print(f"Processing {split} scenes (Total: {len(scene_list)} scenes)")
    for scene in tqdm(scene_list, desc=f"Processing {split} scenes", unit="scene"):
        process_single_scene(
            scene, split, input_root, output_root, path_3dgs_root, prune_methods, prune_params, k_neighbors,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ScanNet scenes with 3DGS merging and save in Pointcept format with 3DGS attributes.")
    parser.add_argument(
        "--output_root",
        default="/home/knuvi/Desktop/song/Pointcept/data/scannet_merged",
        help="Output path for processed data",
    )
    parser.add_argument(
        "--data_type",
        default="samples100",
        choices=["full", "samples100"],
        help="Dataset type: 'full' or 'samples100'",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--attr_pruning_ratio",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Pruning ratios for scale, opacity, rotation (default: 0.0 0.0 0.0). Set ratio > 0 to enable pruning for that attribute.",
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

    # Prune methods 설정
    scale_ratio, rotation_ratio, opacity_ratio = args.attr_pruning_ratio
    prune_methods = {
        'scale': scale_ratio > 0,
        'scale_ratio': scale_ratio,
        'opacity': opacity_ratio > 0,
        'opacity_ratio': opacity_ratio,
        'rotation': rotation_ratio > 0,
        'rotation_ratio': rotation_ratio,
    }

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
        input_root,
        args.output_root,
        'train',
        train_scenes,
        path_3dgs_root,
        prune_methods,
        prune_params,
        k_neighbors=k_neighbors,
        num_workers=args.num_workers
    )

    # Val split 처리
    val_meta_file_path = os.path.join(meta_root, val_meta_file)
    if not os.path.exists(val_meta_file_path):
        raise FileNotFoundError(f"Val meta file not found: {val_meta_file_path}")
    
    with open(val_meta_file_path) as f:
        val_scenes = f.read().splitlines()

    process_scenes(
        input_root,
        args.output_root,
        'val',
        val_scenes,
        path_3dgs_root,
        prune_methods,
        prune_params,
        k_neighbors=k_neighbors,
        num_workers=args.num_workers
    )