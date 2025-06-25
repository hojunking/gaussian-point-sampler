# PC-3DGS_fusion.py

import os, json
import argparse
import numpy as np
from tqdm import tqdm

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes
from boundary_utils import boundary_labeling_with_3dgs, boundary_labeling_with_semantic_label, boundary_labeling_with_semantic_gaussian

def boundary_labeling(pointcept_dir, path_3dgs, output_dir, labeling_method, prune_methods=None, prune_params=None, k_neighbors=10):
    # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    labels_pointcept = pointcept_data['segment20']
    
    # 2. 3DGS 데이터 로드
    points_3dgs, _, raw_features_3dgs = load_3dgs_data(path_3dgs)

    # 3. 3DGS 속성 전처리
    print("Preprocessing 3DGS attributes...")
    features_3dgs = preprocess_3dgs_attributes(raw_features_3dgs)
    
    if labeling_method in ['3dgs', 'both']:
        # 6. 3DGS-attr transfer
        print("Augmenting Pointcept points with 3DGS attributes...")
        features_pointcept  = augment_pointcept_with_3dgs_attributes(
            points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors
        )
        
        boundary_3dgs = boundary_labeling_with_3dgs(
            points_pointcept, features_pointcept, labels_pointcept, prune_methods, prune_params
        )

    if labeling_method in ['label', 'both']:
        
        boundary_label = boundary_labeling_with_semantic_gaussian(
            points_pointcept, labels_pointcept, points_3dgs, features_3dgs, prune_methods, prune_params=prune_params
        )
        # boundary_label = boundary_labeling_with_semantic_label(
        #     points_pointcept, labels_pointcept, prune_params=prune_params
        # )

     # --- 최종 레이블 결정 및 병합 ---
    final_boundary = None
    if labeling_method == '3dgs':
        final_boundary = boundary_3dgs
    elif labeling_method == 'label':
        final_boundary = boundary_label
    elif labeling_method == 'both':
        if boundary_3dgs is not None and boundary_label is not None:
            print("\nMerging results from both methods (logical OR)...")
            final_boundary = np.logical_or(boundary_3dgs, boundary_label).astype(np.int64)
            print(f"Final merged boundary points: {np.sum(final_boundary)}")
        elif boundary_3dgs is not None:
            print("Warning: Only 3DGS-based label was generated. Using it as final.")
            final_boundary = boundary_3dgs
        elif boundary_label is not None:
            print("Warning: Only semantic-based label was generated. Using it as final.")
            final_boundary = boundary_label

    # 9. Pointcept 포맷으로 저장 (.npy 파일)
    save_dict = {
        'boundary': final_boundary.astype(np.int64),
    }
    
    os.makedirs(output_dir, exist_ok=True)
    for key, value in save_dict.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), value)
        print(f"Saved {key}.npy to {output_dir}")

def process_single_scene(scene, split, input_root, output_root, path_3dgs_root, labeling_method, prune_methods, prune_params, k_neighbors=5):
    try:
        pointcept_dir = os.path.join(input_root, split, scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        output_dir = os.path.join(output_root, split, scene)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs) and labeling_method in ['3dgs', 'both']:
            print(f"3DGS PLY file not found: {path_3dgs}, required for '3dgs' method.")
            return

        boundary_labeling(
            pointcept_dir, path_3dgs, output_dir, labeling_method, prune_methods, prune_params, k_neighbors
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def process_scenes(input_root, output_root, split, scene_list, path_3dgs_root, labeling_method, prune_methods=None, prune_params=None, k_neighbors=5, num_workers=1):
    print(f"Processing {split} scenes (Total: {len(scene_list)} scenes) with method: '{labeling_method}'")
    # 병렬 처리는 현재 코드에서 제외되었으나, 필요시 Pool 등을 사용하여 구현할 수 있습니다.
    for scene in tqdm(scene_list, desc=f"Processing {split} scenes", unit="scene"):
        process_single_scene(
            scene, split, input_root, output_root, path_3dgs_root, labeling_method, prune_methods, prune_params, k_neighbors
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate boundary pseudo-labels for ScanNet scenes.")
    parser.add_argument(
        "--output_root",
        default="/home/knuvi/Desktop/song/point/data/scannet_boundary", # 출력 경로 변경 권장
        help="Output path for processed data",
    )
    parser.add_argument(
        "--labeling_method",
        default="both",
        choices=["3dgs", "label", "both"],
        help="Method for boundary labeling: '3dgs' for feature-based, 'label' for semantic-based, 'both' for running both.",
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
    parser.add_argument(
        "--radius",
        default=0.06,
        type=float,
        help="Boundary radius for neighbor search (default: 0.06m)",
    )
    parser.add_argument(
        "--attr_pruning_ratio",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.0],
        help="Pruning ratios for scale, opacity, rotation (for 3dgs method)",
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
    prune_params['boundary_radius'] = args.radius  # 사용자 지정 반경 사용
    
    # prune_params에 boundary_radius가 없는 경우를 대비하여 기본값 설정
    if 'boundary_radius' not in prune_params:
        prune_params['boundary_radius'] = 0.06 # 6cm default
        print(f"Using default boundary_radius: {prune_params['boundary_radius']}m")

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
        input_root, args.output_root, 'train', train_scenes, path_3dgs_root, 
        args.labeling_method, prune_methods, prune_params, k_neighbors, args.num_workers
    )

    # Val split 처리
    val_meta_file_path = os.path.join(meta_root, val_meta_file)
    if not os.path.exists(val_meta_file_path):
        raise FileNotFoundError(f"Val meta file not found: {val_meta_file_path}")
    
    with open(val_meta_file_path) as f:
        val_scenes = f.read().splitlines()

    process_scenes(
        input_root, args.output_root, 'val', val_scenes, path_3dgs_root,
        args.labeling_method, prune_methods, prune_params, k_neighbors, args.num_workers
    )