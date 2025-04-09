# process_3dgs_with_pruning.py
import os
import numpy as np
import argparse
import json
from tqdm import tqdm
from utils import load_pointcept_data, save_ply
from attribute_utils import load_3dgs_data_with_attributes, update_3dgs_labels, augment_pointcept_with_3dgs_attributes, preprocessing_attr
from pruning import prune_by_pointcept_distance

def process_3dgs_with_pruning(pointcept_dir, path_3dgs, output_dir, prune_params=None, k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6, use_features=('scale',), aggregation='mean'):
    """
    Process 3DGS data by pruning based on Pointcept points and save in Pointcept format.
    
    Args:
        pointcept_dir (str): Pointcept data directory (e.g., scannet/train/scene0000_00).
        path_3dgs (str): Path to the 3DGS PLY file.
        output_dir (str): Output directory (e.g., scannet_3dgs/train).
        prune_params (dict): Pruning hyperparameters (e.g., pdistance_max).
        k_neighbors (int): Number of nearest neighbors for label mapping.
        use_label_consistency (bool): Whether to use label consistency filtering.
        ignore_threshold (float): Threshold for ignore_index (-1) ratio.
        use_features (tuple): Features to use from 3DGS attributes ('scale', 'opacity', 'rotation').
        aggregation (str): Aggregation method for attributes ('mean', 'max', 'median').
    """
    try:
        # 1. Pointcept 데이터 로드 (.npy 파일에서)
        pointcept_data = load_pointcept_data(pointcept_dir)
        points_pointcept = pointcept_data['coord']
        labels_pointcept = pointcept_data['segment20']
        labels200_pointcept = pointcept_data['segment200']
        instances_pointcept = pointcept_data['instance']

        # 2. 3DGS 데이터 로드 (속성 및 SH 계수 포함)
        points_3dgs, colors_3dgs, raw_features_3dgs = load_3dgs_data_with_attributes(path_3dgs)
        features_3dgs = preprocessing_attr(raw_features_3dgs)
        
        print(f"Loaded 3DGS data: {len(points_3dgs)} points")  # 디버깅 로그 추가

        # 3. 3DGS 점 pruning (Pointcept-Distance 기반)
        pdistance_max = prune_params.get('pdistance_max', 0.00008)
        mask_prune = prune_by_pointcept_distance(points_3dgs, points_pointcept, pdistance_max=pdistance_max)
        points_3dgs = points_3dgs[mask_prune]
        colors_3dgs = colors_3dgs[mask_prune]
        features_3dgs = features_3dgs[mask_prune]
        print(f"After pruning: {len(points_3dgs)} points")  # 디버깅 로그 추가

        # 4. Pruned 3DGS 점에 Pointcept 레이블 매핑 (KNN 사용)
        labels_3dgs, labels200_3dgs, instances_3dgs, mask_consistency = update_3dgs_labels(
            points_3dgs,
            points_pointcept,
            labels_pointcept,
            labels200_pointcept,
            instances_pointcept,
            k_neighbors=k_neighbors,
            use_label_consistency=use_label_consistency,
            ignore_threshold=ignore_threshold
        )

        # 일관성 체크 결과 적용
        points_3dgs = points_3dgs[mask_consistency]
        colors_3dgs = colors_3dgs[mask_consistency]
        features_3dgs = features_3dgs[mask_consistency]
        labels_3dgs = labels_3dgs[mask_consistency]
        labels200_3dgs = labels200_3dgs[mask_consistency]
        instances_3dgs = instances_3dgs[mask_consistency]
        print(f"Removed {np.sum(~mask_consistency)} points due to label inconsistency")
        print(f"Final 3DGS points after consistency check: {len(points_3dgs)} points")  # 디버깅 로그 추가

        # 5. 3DGS 데이터에 법선 추가 (기본값: 0, 0, 0)
        normals_3dgs = np.zeros((len(points_3dgs), 3), dtype=np.float32)  # [N, 3]

        # 6. 선택적 features 처리
        # augment_pointcept_with_3dgs_attributes를 사용하여 features 필터링
        # 여기서는 3DGS 포인트 자체에 대해 처리하므로 points_pointcept=points_3dgs로 설정
        filtered_features_3dgs = augment_pointcept_with_3dgs_attributes(
            points_3dgs, points_3dgs, features_3dgs, k_neighbors=1, use_features=use_features, aggregation=aggregation
        )
        print(f"Filtered features shape: {filtered_features_3dgs.shape}")  # 디버깅 로그 추가

        # 7. Pointcept 포맷으로 저장 (.npy 파일)
        save_dict = {
            'coord': points_3dgs.astype(np.float32),
            'color': (colors_3dgs * 255).astype(np.uint8),  # [0, 1] -> [0, 255]로 변환
            'normal': normals_3dgs.astype(np.float32),
            'features': filtered_features_3dgs.astype(np.float32),  # 선택된 features
            'segment20': labels_3dgs.astype(np.int64),
            'segment200': labels200_3dgs.astype(np.int64),
            'instance': instances_3dgs.astype(np.int64)
        }

        os.makedirs(output_dir, exist_ok=True)
        for key, value in save_dict.items():
            np.save(os.path.join(output_dir, f"{key}.npy"), value)
            print(f"Saved {key}.npy to {output_dir}")

        # 처리된 데이터 반환 (PLY 저장용)
        return points_3dgs, colors_3dgs, labels_3dgs

    except Exception as e:
        print(f"Error processing scene: {e}")
        return None, None, None

def process_single_scene(scene, input_root, output_root, path_3dgs_root, split, prune_params, k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6, use_features=('scale',), aggregation='mean'):
    """
    Process a single scene.
    
    Args:
        scene (str): Scene name.
        input_root (str): Input Pointcept data root directory.
        output_root (str): Output directory.
        path_3dgs_root (str): 3DGS data root directory.
        split (str): 'train' or 'val' (data split).
        prune_params (dict): Pruning hyperparameters.
        k_neighbors (int): Number of nearest neighbors for label mapping.
        use_label_consistency (bool): Whether to use label consistency filtering.
        ignore_threshold (float): Threshold for ignore_index (-1) ratio.
        use_features (tuple): Features to use from 3DGS attributes ('scale', 'opacity', 'rotation').
        aggregation (str): Aggregation method for attributes ('mean', 'max', 'median').
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

        # 3DGS 데이터 처리 및 Pointcept 포맷으로 저장
        process_3dgs_with_pruning(
            pointcept_dir, path_3dgs, output_dir, prune_params, k_neighbors, use_label_consistency, ignore_threshold, use_features, aggregation
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def process_single_scene_for_visualization(scene, input_root, output_root, path_3dgs_root, split, exp, prune_params, k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6, use_features=('scale',), aggregation='mean'):
    """
    Process a single scene and save as PLY for visualization.
    
    Args:
        scene (str): Scene name.
        input_root (str): Input Pointcept data root directory.
        output_root (str): Output directory.
        path_3dgs_root (str): 3DGS data root directory.
        split (str): 'train' or 'val' (data split).
        exp (str): Experiment name.
        prune_params (dict): Pruning hyperparameters.
        k_neighbors (int): Number of nearest neighbors for label mapping.
        use_label_consistency (bool): Whether to use label consistency filtering.
        ignore_threshold (float): Threshold for ignore_index (-1) ratio.
        use_features (tuple): Features to use from 3DGS attributes ('scale', 'opacity', 'rotation').
        aggregation (str): Aggregation method for attributes ('mean', 'max', 'median').
    """
    try:
        # 입력 경로
        pointcept_dir = os.path.join(input_root, split, scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        # 출력 경로
        output_dir = os.path.join(output_root, exp)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"3DGS PLY file not found: {path_3dgs}")
            return

        # 3DGS 데이터 처리
        points_3dgs, colors_3dgs, labels_3dgs = process_3dgs_with_pruning(
            pointcept_dir, path_3dgs, output_dir, prune_params, k_neighbors, use_label_consistency, ignore_threshold, use_features, aggregation
        )

        if points_3dgs is None:
            print("Processing failed, skipping PLY generation.")
            return

        # 레이블 분포 출력
        unique_labels, label_counts = np.unique(labels_3dgs, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique_labels, label_counts):
            print(f"Label {label}: {count} points")

        # 색상 분포 출력
        print("Label colors distribution:")
        print(f"Red: min={colors_3dgs[:, 0].min()}, max={colors_3dgs[:, 0].max()}")
        print(f"Green: min={colors_3dgs[:, 1].min()}, max={colors_3dgs[:, 1].max()}")
        print(f"Blue: min={colors_3dgs[:, 2].min()}, max={colors_3dgs[:, 2].max()}")

        # PLY 파일로 저장
        os.makedirs(output_dir, exist_ok=True)
        output_ply_path = os.path.join(output_dir, f"{exp}_3dgs.ply")
        save_ply(
            points_3dgs, (colors_3dgs * 255).astype(np.uint8), labels_3dgs, output_ply_path,
            save_separate_labels=False
        )

    except Exception as e:
        print(f"Error processing scene for visualization {scene}: {e}")

def process_scenes(input_root, output_root, scene_list, path_3dgs_root, split, prune_params=None, k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6, use_features=('scale',), aggregation='mean', num_workers=1):
    """
    Process a list of scenes and save in Pointcept format.
    
    Args:
        input_root (str): Input Pointcept data root directory (e.g., scannet).
        output_root (str): Output directory (e.g., scannet_3dgs).
        scene_list (list): List of scene names to process.
        path_3dgs_root (str): 3DGS data root directory.
        split (str): 'train' or 'val' (data split).
        prune_params (dict): Pruning hyperparameters.
        k_neighbors (int): Number of nearest neighbors for label mapping.
        use_label_consistency (bool): Whether to use label consistency filtering.
        ignore_threshold (float): Threshold for ignore_index (-1) ratio.
        use_features (tuple): Features to use from 3DGS attributes ('scale', 'opacity', 'rotation').
        aggregation (str): Aggregation method for attributes ('mean', 'max', 'median').
        num_workers (int): Number of workers for parallel processing (not used).
    """
    print(f"Processing {split} scenes (Total: {len(scene_list)} scenes)")
    # 순차적 처리
    for scene in tqdm(scene_list, desc=f"Processing {split} scenes", unit="scene"):
        process_single_scene(scene, input_root, output_root, path_3dgs_root, split, prune_params, k_neighbors, use_label_consistency, ignore_threshold, use_features, aggregation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 3DGS data with pruning and save in Pointcept format or PLY for visualization.")
    parser.add_argument(
        "--input_root",
        default="/home/knuvi/Desktop/song/Pointcept/data/scannet",
        help="Path to the ScanNet dataset root (default: /home/knuvi/Desktop/song/Pointcept/data/scannet)",
    )
    parser.add_argument(
        "--output_root",
        default="/home/knuvi/Desktop/song/Pointcept/data/scannet_3dgs",
        help="Output path for processed data (default: /home/knuvi/Desktop/song/Pointcept/data/scannet_3dgs)",
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
        "--k_neighbors",
        default=5,
        type=int,
        help="Number of neighbors for label mapping",
    )
    parser.add_argument(
        "--use_label_consistency",
        default=True,
        type=bool,
        help="Whether to use label consistency filtering",
    )
    parser.add_argument(
        "--ignore_threshold",
        default=0.6,
        type=float,
        help="Threshold for ignore_index (-1) ratio",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Process a single scene and save as PLY for visualization",
    )
    parser.add_argument(
        "--scene",
        default="scene0011_00",
        help="Scene to process for visualization (e.g., scene0011_00)",
    )
    parser.add_argument(
        "--exp",
        default="",
        help="Experiment name for visualization",
    )
    parser.add_argument(
        "--pdistance_max",
        default=None,
        type=float,
        help="Override pdistance_max for pruning (default: use value from config.json)",
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

    # Pruning 파라미터 설정 (config.json에서 로드)
    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    prune_params = config.get('prune_params', {'pdistance_max': 0.00008})

    # pdistance_max 오버라이드
    if args.pdistance_max is not None:
        prune_params['pdistance_max'] = args.pdistance_max
        print(f"Overriding pdistance_max to {args.pdistance_max}")

    if args.visualize:
        # 단일 Scene을 처리하고 PLY로 저장
        process_single_scene_for_visualization(
            args.scene,
            args.input_root,
            args.output_root,
            args.path_3dgs_root,
            args.split,
            args.exp if args.exp else args.scene,
            prune_params,
            k_neighbors=args.k_neighbors,
            use_label_consistency=args.use_label_consistency,
            ignore_threshold=args.ignore_threshold,
            use_features=tuple(args.use_features),
            aggregation=args.aggregation
        )
    else:
        # Scene 목록 로드
        if args.split == "train":
            scene_file = "train100_samples.txt"
        else:
            scene_file = "valid20_samples.txt"

        with open(os.path.join(args.meta_root, scene_file)) as f:
            scenes = f.read().splitlines()

        # Scenes 처리
        process_scenes(
            args.input_root,
            args.output_root,
            scenes,
            args.path_3dgs_root,
            args.split,
            prune_params,
            k_neighbors=args.k_neighbors,
            use_label_consistency=args.use_label_consistency,
            ignore_threshold=args.ignore_threshold,
            use_features=tuple(args.use_features),
            aggregation=args.aggregation,
            num_workers=args.num_workers
        )