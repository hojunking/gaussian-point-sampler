# process_pointcept_with_3dgs_attributes.py
import os
import numpy as np
import argparse
import json
from tqdm import tqdm
from attribute_utils import load_pointcept_data, load_3dgs_data_with_attributes, augment_pointcept_with_3dgs_attributes, preprocessing_attr
from pruning import prune_by_pointcept_distance

def merge_pointcept_with_3dgs_attributes(pointcept_dir, path_3dgs, output_dir, prune_methods=None, prune_params=None, k_neighbors=5, use_features=('scale',), aggregation='mean'):
    """
    Pointcept Point Cloud에 3DGS 속성을 전달하고 Pointcept 포맷으로 저장.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/train/scene0000_00).
        path_3dgs (str): 3DGS Point Cloud 경로.
        output_dir (str): 출력 디렉토리 (예: scannet_merged/train).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): KNN에서 사용할 이웃 점 개수.
        use_features (tuple): 사용할 features ('scale', 'opacity', 'rotation').
        aggregation (str): 속성 집계 방식 ('mean', 'max', 'median').
    """
    try:
        # 1. Pointcept 데이터 로드 (.npy 파일에서)
        pointcept_data = load_pointcept_data(pointcept_dir)
        points_pointcept = pointcept_data['coord']
        colors_pointcept = pointcept_data['color']
        normals_pointcept = pointcept_data['normal']
        labels_pointcept = pointcept_data['segment20']
        labels200_pointcept = pointcept_data['segment200']
        instances_pointcept = pointcept_data['instance']

        # 2. 3DGS 데이터 로드 (속성 포함)
        points_3dgs, _, raw_features_3dgs = load_3dgs_data_with_attributes(path_3dgs)
        features_3dgs = preprocessing_attr(raw_features_3dgs)

        # 3. 3DGS 점 pruning (Pointcept-Distance 기반)
        if prune_methods.get('pointcept_distance', False):
            pdistance_max = prune_params.get('pointcept_distance', 0.00008)
            mask = prune_by_pointcept_distance(points_3dgs, points_pointcept, pdistance_max=pdistance_max)
            points_3dgs = points_3dgs[mask]
            features_3dgs = features_3dgs[mask]
            print(f"Pruned 3DGS points: {np.sum(~mask)} points removed (pdistance_max={pdistance_max})")
        else:
            print("Skipping Pointcept-Distance pruning")

        # 4. Pointcept 점에 3DGS 속성 전달
        augmented_features = augment_pointcept_with_3dgs_attributes(
            points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors, use_features=use_features, aggregation=aggregation
        )

        # 5. Pointcept 포맷으로 저장 (.npy 파일)
        save_dict = {
            'coord': points_pointcept.astype(np.float32),
            'color': colors_pointcept.astype(np.uint8),
            'normal': normals_pointcept.astype(np.float32),
            'features': augmented_features.astype(np.float32),  # 선택된 features
            'segment20': labels_pointcept.astype(np.int64),
            'segment200': labels200_pointcept.astype(np.int64),
            'instance': instances_pointcept.astype(np.int64)
        }

        os.makedirs(output_dir, exist_ok=True)
        for key, value in save_dict.items():
            np.save(os.path.join(output_dir, f"{key}.npy"), value)
            print(f"Saved {key}.npy to {output_dir}")

    except Exception as e:
        print(f"Error processing scene: {e}")

def process_single_scene(scene, input_root, output_root, path_3dgs_root, split, prune_methods, prune_params, k_neighbors=5, use_features=('scale',), aggregation='mean'):
    """
    단일 scene을 처리하는 함수.
    
    Args:
        scene (str): 처리할 scene 이름.
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리.
        output_root (str): 출력 디렉토리.
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        split (str): 'train' 또는 'val' (데이터 분할).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): KNN에서 사용할 이웃 점 개수.
        use_features (tuple): 사용할 features ('scale', 'opacity', 'rotation').
        aggregation (str): 속성 집계 방식 ('mean', 'max', 'median').
    """
    try:
        # 입력 경로
        pointcept_dir = os.path.join(input_root, split, scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        # 출력 경로
        output_dir = os.path.join(output_root, split, scene)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"3DGS PLY file not found: {path_3dgs}")
            return

        # 병합 및 Pointcept 포맷으로 저장
        merge_pointcept_with_3dgs_attributes(
            pointcept_dir, path_3dgs, output_dir, prune_methods, prune_params, k_neighbors, use_features, aggregation
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def process_scenes(input_root, output_root, scene_list, path_3dgs_root, split, prune_methods=None, prune_params=None, k_neighbors=5, use_features=('scale',), aggregation='mean', num_workers=1):
    """
    주어진 scene 목록을 처리하여 Pointcept 포맷으로 저장.
    
    Args:
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리 (예: scannet).
        output_root (str): 출력 디렉토리 (예: scannet_merged).
        scene_list (list): 처리할 scene 이름 목록.
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        split (str): 'train' 또는 'val' (데이터 분할).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): KNN에서 사용할 이웃 점 개수.
        use_features (tuple): 사용할 features ('scale', 'opacity', 'rotation').
        aggregation (str): 속성 집계 방식 ('mean', 'max', 'median').
        num_workers (int): 병렬 처리 작업자 수 (사용하지 않음).
    """
    print(f"Processing {split} scenes (Total: {len(scene_list)} scenes)")
    # 순차적 처리
    for scene in tqdm(scene_list, desc=f"Processing {split} scenes", unit="scene"):
        process_single_scene(scene, input_root, output_root, path_3dgs_root, split, prune_methods, prune_params, k_neighbors, use_features, aggregation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ScanNet scenes with 3DGS attribute augmentation and save in Pointcept format.")
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
        help="Path to the meta directory containing train_100_samples.txt and val_20_samples.txt",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Data split to process: 'train' or 'val' (default: train)",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers for parallel processing (not used in sequential mode)",
    )
    parser.add_argument(
        "--enable_pointcept_distance",
        action="store_true",
        help="Enable Pointcept distance-based pruning",
    )
    parser.add_argument(
        "--k_neighbors",
        default=5,
        type=int,
        help="Number of neighbors for attribute augmentation",
    )
    parser.add_argument(
        "--use_features",
        nargs='+',
        default=['scale'],
        choices=['scale', 'opacity', 'rotation'],
        help="Features to use from 3DGS attributes (default: ['scale'])",
    )
    parser.add_argument(
        "--aggregation",
        default='mean',
        choices=['mean', 'max', 'median'],
        help="Aggregation method for attributes ('mean', 'max', 'median') (default: 'mean')",
    )
    args = parser.parse_args()

    # Scene 목록 로드
    if args.split == "train":
        scene_file = "train100_samples.txt"
    else:
        scene_file = "valid20_samples.txt"

    with open(os.path.join(args.meta_root, scene_file)) as f:
        scenes = f.read().splitlines()

    # Pruning 방법 설정
    prune_methods = {
        'pointcept_distance': args.enable_pointcept_distance,
    }

    # Pruning 파라미터 설정 (config.json에서 로드)
    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    prune_params = config.get('prune_params', {'pdistance_max': 0.00008})

    # Scenes 처리
    process_scenes(
        args.input_root,
        args.output_root,
        scenes,
        args.path_3dgs_root,
        args.split,
        prune_methods,
        prune_params,
        k_neighbors=args.k_neighbors,
        use_features=tuple(args.use_features),
        aggregation=args.aggregation,
        num_workers=args.num_workers
    )