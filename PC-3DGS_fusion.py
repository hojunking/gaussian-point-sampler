# PC-3DGS_fusion.py

import os, json
import argparse
import numpy as np
from tqdm import tqdm

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data, voxelize_3dgs, update_3dgs_attributes
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes, remove_duplicates, prune_3dgs

def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, prune_methods=None, prune_params=None, k_neighbors=5, ignore_threshold=0.6, voxel_size=0.02, use_features=('scale', 'opacity', 'rotation')):
    """
    Pointcept Point Cloud와 3DGS Point Cloud를 병합하고, 3DGS 속성을 전달하여 PLY 파일로 저장.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/train/scene0000_00).
        path_3dgs (str): 3DGS Point Cloud 경로.
        output_dir (str): 출력 디렉토리 (예: scannet_merged/train).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 속성 전달에 사용할 이웃 점 개수.
        ignore_threshold (float): ignore_index(-1) 라벨의 비율이 이 값 이상이면 결과 라벨을 -1로 설정.
        voxel_size (float): Voxel 크기 (기본값: 0.02m). 0이 아니면 voxelization 적용.
        use_features (tuple): 사용할 3DGS 속성 ('scale', 'opacity', 'rotation').
    """
    # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    colors_pointcept = pointcept_data['color']
    normals_pointcept = pointcept_data['normal']
    labels_pointcept = pointcept_data['segment20']
    labels200_pointcept = pointcept_data['segment200']
    instances_pointcept = pointcept_data['instance']
    
    # 2. 3DGS 데이터 로드
    points_3dgs, normals_3dgs, vertex_data_3dgs = load_3dgs_data(path_3dgs)

    # 3. 3DGS 속성 전처리
    print("Preprocessing 3DGS attributes...")
    features_3dgs = preprocess_3dgs_attributes(vertex_data_3dgs)

    # 4. 3DGS 데이터 Pruning
    print("Pruning 3DGS points...")
    points_3dgs, normals_3dgs, vertex_data_3dgs, features_3dgs = prune_3dgs(
        vertex_data_3dgs, points_3dgs, normals_3dgs, features_3dgs, points_pointcept, normals_pointcept, prune_methods, prune_params
    )

    # 5. 3DGS 데이터 Voxelization (옵션)
    voxelize = voxel_size != 0  # voxel_size가 0이 아니면 voxelize 활성화
    if voxelize:
        print("Applying Voxelization to 3DGS points...")
        points_3dgs, normals_3dgs, vertex_data_3dgs, features_3dgs = voxelize_3dgs(
            points_3dgs, normals_3dgs, vertex_data_3dgs, features_3dgs, voxel_size=voxel_size, k_neighbors=5
        )
    else:
        print("Skipping Voxelization for 3DGS data.")

    print("Augmenting Pointcept points with 3DGS attributes...")
    features_pointcept = augment_pointcept_with_3dgs_attributes(
        points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors, use_features=use_features
    )

    # 6. 3DGS 점의 색상, 법선, 라벨을 복사
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

    # 7. 중복 점 제거 (3DGS 점 제거)
    print("Removing duplicate points...")
    points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, features_3dgs = remove_duplicates(
        points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, features_3dgs
    )

    # 8. 병합
    print("Merging Pointcept and 3DGS points...")
    points_merged = np.vstack((points_pointcept, points_3dgs))
    colors_merged = np.vstack((colors_pointcept, colors_3dgs))
    normals_merged = np.vstack((normals_pointcept, normals_3dgs))
    labels_merged = np.concatenate((labels_pointcept, labels_3dgs))
    labels200_merged = np.concatenate((labels200_pointcept, labels200_3dgs))
    instances_merged = np.concatenate((instances_pointcept, instances_3dgs))
    features_merged = np.vstack((features_pointcept, features_3dgs))  # 3DGS 점의 원래 속성 유지
    print(f"Final merged points: {len(points_merged)} (Pointcept: {len(points_pointcept)}, 3DGS: {len(points_3dgs)})")

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
        "--input_root",
        default="/home/knuvi/Desktop/song/Pointcept/data/scannet",
        help="Path to the ScanNet dataset root (default: /home/knuvi/Desktop/song/Pointcept/data/scannet)",
    )
    parser.add_argument(
        "--output_root",
        default="/home/knuvi/Desktop/song/Pointcept/data/scannet_merged",
        help="Output path for processed data (default: /home/knuvi/Desktop/song/Pointcept/data/scannet_merged)",
    )
    parser.add_argument(
        "--path_3dgs_root",
        default="/home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output",
        help="Path to the 3DGS dataset (default: /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output)",
    )
    parser.add_argument(
        "--meta_root",
        default="/home/knuvi/Desktop/song/gaussian-point-sampler/meta",
        help="Path to the meta directory containing train_100_samples.txt and val_samples.txt (default: /home/knuvi/Desktop/song/gaussian-point-sampler/meta)",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Dataset split to process (train or val, default: train)",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers for parallel processing (not used in sequential mode)",
    )
    parser.add_argument(
        "--scale_ratio",
        default=0,
        type=float,
        help="Ratio of points to prune based on scale (top X%). If > 0, scale pruning is enabled.",
    )
    parser.add_argument(
        "--opacity_ratio",
        default=0,
        type=float,
        help="Ratio of points to prune based on opacity (bottom X%). If > 0, opacity pruning is enabled.",
    )
    parser.add_argument(
        "--enable_density",
        action="store_true",
        help="Enable density-based pruning",
    )
    parser.add_argument(
        "--enable_normal",
        action="store_true",
        help="Enable normal-based pruning",
    )
    parser.add_argument(
        "--enable_sor",
        action="store_true",
        help="Enable SOR-based pruning",
    )
    parser.add_argument(
        "--k_neighbors",
        default=5,
        type=int,
        help="Number of neighbors for label voting and attribute transfer",
    )
    parser.add_argument(
        "--pdistance",
        default=0.001,
        type=float,
        help="Pointcept distance threshold. If specified, pointcept distance pruning is enabled.",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.02,
        type=float,
        help="Voxel size for 3DGS voxelization (default: 0.02m). Set to 0 to disable voxelization.",
    )
    parser.add_argument(
        "--use_features",
        nargs='+',
        default=['scale', 'opacity', 'rotation'],
        help="3DGS features to transfer (default: scale opacity rotation)",
    )

    args = parser.parse_args()

    # Split에 따라 적절한 메타 파일 읽기
    meta_file = "train100_samples.txt" if args.split == "train" else "valid20_samples.txt"
    meta_file_path = os.path.join(args.meta_root, meta_file)
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"Meta file not found: {meta_file_path}")

    with open(meta_file_path) as f:
        scenes = f.read().splitlines()

    # prune_methods 설정
    prune_methods = {
        'scale': args.scale_ratio > 0,
        'scale_ratio': args.scale_ratio,
        'opacity': args.opacity_ratio > 0,
        'opacity_ratio': args.opacity_ratio,
        'density': args.enable_density,
        'pointcept_distance': args.pdistance > 0,  # pdistance가 0이 아니면 pointcept_distance 활성화
        'normal': args.enable_normal,
        'sor': args.enable_sor
    }

    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    prune_params = config['prune_params']
    prune_params['pointcept_max_distance'] = args.pdistance
    print(f'Pointcept distance threshold: {prune_params["pointcept_max_distance"]}')

    process_scenes(
        args.input_root,
        args.output_root,
        args.split,
        scenes,
        args.path_3dgs_root,
        prune_methods,
        prune_params,
        k_neighbors=args.k_neighbors,
        num_workers=args.num_workers,
        voxel_size=args.voxel_size,
        use_features=tuple(args.use_features)
    )