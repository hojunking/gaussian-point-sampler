# PC-3DGS_fusion.py

import os, json
import argparse
import numpy as np
from tqdm import tqdm

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data, voxelize_3dgs, update_3dgs_attributes, fps_knn_sampling
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes, remove_duplicates, pdistance_pruning, pruning_3dgs_attr, select_3dgs_features, pruning_all_3dgs_attr

def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, prune_methods=None, prune_params=None, k_neighbors=10, ignore_threshold=0.6, voxel_size=0.02, use_features=('scale', 'opacity', 'rotation')):
    # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    colors_pointcept = pointcept_data['color']
    normals_pointcept = pointcept_data['normal']
    labels_pointcept = pointcept_data['segment20']
    labels200_pointcept = pointcept_data['segment200']
    instances_pointcept = pointcept_data['instance']
    
    # 2. 3DGS 데이터 로드
    points_3dgs, normals_3dgs, raw_features_3dgs = load_3dgs_data(path_3dgs)

    # 3. 3DGS 속성 전처리
    print("Preprocessing 3DGS attributes...")
    features_3dgs = preprocess_3dgs_attributes(raw_features_3dgs)
    
    # 4. 3DGS pdistance Pruning
    pdistance = prune_params['pointcept_max_distance']  !=  0 # voxel_size가 0이 아니면 pdistance 활성화
    # 4. 3DGS pdistance Pruning
    if pdistance:
        points_3dgs, features_3dgs = pdistance_pruning(
            points_3dgs, features_3dgs, points_pointcept, prune_params
        )

    # 6. 3DGS-attr transfer
    print("Augmenting Pointcept points with 3DGS attributes...")
    features_pointcept  = augment_pointcept_with_3dgs_attributes(
        points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors
    )
    
    # 5. 3DGS 데이터 Voxelization (옵션)
    voxelize = voxel_size != 0  # voxel_size가 0이 아니면 voxelize 활성화
    if voxelize:
        print("Applying Voxelization to 3DGS points...")
        points_3dgs, features_3dgs = voxelize_3dgs(
            points_3dgs, features_3dgs, voxel_size=voxel_size, k_neighbors_max=20)
    else:
        print("Applying FPS-kNN to 3DGS points...")
        points_3dgs, features_3dgs = fps_knn_sampling(
            points_3dgs, features_3dgs, sample_ratio=0.01, aggregation_method='mean')
    
     # 7. 3DGS-attr Pruning
    points_3dgs, features_3dgs = pruning_3dgs_attr(
        points_3dgs, features_3dgs, prune_methods, prune_params
    )

    features_pointcept, features_3dgs = select_3dgs_features(
        features_pointcept, features_3dgs, use_features=use_features
    )
    # 7. 중복 점 제거 (3DGS 점 제거)
    print("Removing duplicate points...")
    points_3dgs, features_3dgs = remove_duplicates(points_3dgs, features_3dgs, points_pointcept)
    
    # 8. Point cloud color, normals, labels transfer 
    print("Updating 3DGS attributes from Pointcept...")
    colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask = update_3dgs_attributes(
        points_3dgs, points_pointcept, colors_pointcept, normals_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept,
        k_neighbors=k_neighbors, use_label_consistency=True, ignore_threshold=ignore_threshold
    )
    # 라벨 일관성 필터링 적용
    points_3dgs = points_3dgs[mask]
    colors_3dgs = colors_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    labels_3dgs = labels_3dgs[mask]
    labels200_3dgs = labels200_3dgs[mask]
    instances_3dgs = instances_3dgs[mask]
    features_3dgs = features_3dgs[mask]  # 3DGS 속성도 필터링하여 동기화

    # 6.5. -1 라벨 3DGS 점 비율 출력
    ignore_count = np.sum(labels_3dgs == -1)
    total_count = len(labels_3dgs)
    ignore_ratio = ignore_count / total_count if total_count > 0 else 0
    print(f"3DGS points with label -1: {ignore_count}/{total_count} (Ratio: {ignore_ratio:.4f})")

    # 8. 병합
    print("Merging Pointcept and 3DGS points...")
    points_merged = np.vstack((points_pointcept, points_3dgs))
    colors_merged = np.vstack((colors_pointcept, colors_3dgs))
    normals_merged = np.vstack((normals_pointcept, normals_3dgs))
    labels_merged = np.concatenate((labels_pointcept, labels_3dgs))
    labels200_merged = np.concatenate((labels200_pointcept, labels200_3dgs))
    instances_merged = np.concatenate((instances_pointcept, instances_3dgs))
    features_merged = np.vstack((features_pointcept, features_3dgs))  # 3DGS 점의 원래 속성 유지
    print(f"Merged points: {len(points_merged)} (Pointcept: {len(points_pointcept)}, 3DGS: {len(points_3dgs)})")

    # # 8. 3DGS-attr Pruning
    # if any(ratio != 0 for ratio in (prune_methods.get('scale_ratio', 0), 
    #                                 prune_methods.get('rotation_ratio', 0), 
    #                                 prune_methods.get('opacity_ratio', 0))):
    #     print(f"Before pruning points: {len(points_merged)}")
    #     points_merged, features_merged, colors_merged, normals_merged, labels_merged, labels200_merged, instances_merged = pruning_all_3dgs_attr(
    #         points_merged, features_merged, prune_methods, prune_params, colors_merged, normals_merged, labels_merged, labels200_merged, instances_merged,
    #     )
    #     print(f"After pruning points: {len(points_merged)}")

    # 9. Pointcept 포맷으로 저장 (.npy 파일)
    save_dict = {
        'coord': points_merged.astype(np.float32),
        'color': colors_merged.astype(np.uint8),
        'normal': normals_merged.astype(np.float32),
        'segment20': labels_merged.astype(np.int64),
        'segment200': labels200_merged.astype(np.int64),
        'instance': instances_merged.astype(np.int64),
        'features': features_merged.astype(np.float32),  # features.npy로 저장
    }
    os.makedirs(output_dir, exist_ok=True)
    for key, value in save_dict.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), value)
        print(f"Saved {key}.npy to {output_dir}")

def process_single_scene(scene, split, input_root, output_root, path_3dgs_root, prune_methods, prune_params, k_neighbors=5, voxel_size=0.02, use_features=('scale', 'opacity', 'rotation')):
    """
    단일 scene을 처리하는 함수.
    
    Args:
        scene (str): 처리할 scene 이름.
        split (str): 처리할 데이터셋 split ('train' 또는 'val').
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리.
        output_root (str): 출력 디렉토리.
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 속성 전달에 사용할 이웃 점 개수.
        voxel_size (float): Voxel 크기 (기본값: 0.02m). 0이 아니면 voxelization 적용.
        use_features (tuple): 사용할 3DGS 속성 ('scale', 'opacity', 'rotation').
    """
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

        merge_pointcept_with_3dgs(
            pointcept_dir, path_3dgs, output_dir, prune_methods, prune_params, k_neighbors,
            voxel_size=voxel_size, use_features=use_features
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def process_scenes(input_root, output_root, split, scene_list, path_3dgs_root, prune_methods=None, prune_params=None, k_neighbors=5, num_workers=1, voxel_size=0.02, use_features=('scale', 'opacity', 'rotation')):
    """
    주어진 scene 목록을 처리하여 Pointcept 포맷으로 저장.
    
    Args:
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리 (예: scannet).
        output_root (str): 출력 디렉토리 (예: scannet_merged).
        split (str): 처리할 데이터셋 split ('train' 또는 'val').
        scene_list (list): 처리할 scene 이름 목록.
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 속성 전달에 사용할 이웃 점 개수.
        num_workers (int): 병렬 처리 작업자 수 (사용하지 않음).
        voxel_size (float): Voxel 크기 (기본값: 0.02m). 0이 아니면 voxelization 적용.
        use_features (tuple): 사용할 3DGS 속성 ('scale', 'opacity', 'rotation').
    """
    print(f"Processing {split} scenes (Total: {len(scene_list)} scenes)")
    for scene in tqdm(scene_list, desc=f"Processing {split} scenes", unit="scene"):
        process_single_scene(
            scene, split, input_root, output_root, path_3dgs_root, prune_methods, prune_params, k_neighbors,
            voxel_size=voxel_size, use_features=use_features
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
    parser.add_argument(
        "--pdistance",
        default=0.001,
        type=float,
        help="Pointcept distance threshold",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.04,
        type=float,
        help="Voxel size for 3DGS voxelization (default: 0.02m)",
    )
    parser.add_argument(
        "--use_features",
        nargs='+',
        help="3DGS features to transfer (default: scale opacity rotation)",
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
    prune_params['pointcept_max_distance'] = args.pdistance


    # Prune methods 설정
    scale_ratio, opacity_ratio, rotation_ratio = args.attr_pruning_ratio
    prune_methods = {
        'pointcept_distance': args.pdistance > 0,
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
        num_workers=args.num_workers,
        voxel_size=args.voxel_size,
        use_features=tuple(args.use_features)
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
        num_workers=args.num_workers,
        voxel_size=args.voxel_size,
        use_features=tuple(args.use_features)
    )