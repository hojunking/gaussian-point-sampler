import numpy as np
import argparse
from pruning import prune_3dgs
from utils import load_pointcept_data, load_3dgs_data, update_3dgs_attributes, remove_duplicates, remove_outliers_3dgs
from tqdm import tqdm
import os

def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, prune_methods=None, prune_params=None, k_neighbors=5, use_label_consistency=True):
    """
    Pointcept Point Cloud와 3DGS Point Cloud를 병합하고 Pointcept 포맷으로 저장.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/train/scene0000_00).
        path_3dgs (str): 3DGS Point Cloud 경로.
        output_dir (str): 출력 디렉토리 (예: scannet_merged/train/scene0000_00).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
    """
    # 기본값 설정
    if prune_methods is None:
        prune_methods = {
            'scale': False,
            'scale_ratio': 0.0,
            'opacity': False,
            'opacity_ratio': 0.0,
            'density': False,
            'pointcept_distance': False,
            'normal': False,
            'sor': True
        }
    if prune_params is None:
        prune_params = {
            'density_eps': 0.1,
            'density_min_points': 10,
            'pointcept_max_distance': 0.2,
            'normal_k_neighbors': 10,
            'normal_cos_threshold': 0.8,
            'sor_nb_neighbors': 20,
            'sor_std_ratio': 2.0
        }

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

    # 3. 3DGS 점 pruning (normals_pointcept 전달)
    points_3dgs, normals_3dgs, vertex_data_3dgs = prune_3dgs(
        vertex_data_3dgs, points_3dgs, normals_3dgs, points_pointcept, normals_pointcept, prune_methods, prune_params
    )

    # 4. 3DGS 점의 색상과 라벨을 복사
    colors_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask = update_3dgs_attributes(
        points_3dgs, points_pointcept, colors_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept,
        k_neighbors=k_neighbors, use_label_consistency=use_label_consistency
    )

    # 라벨 일관성 필터링 적용
    points_3dgs = points_3dgs[mask]
    colors_3dgs = colors_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    labels_3dgs = labels_3dgs[mask]
    labels200_3dgs = labels200_3dgs[mask]
    instances_3dgs = instances_3dgs[mask]

    # 5. 중복 점 제거 (3DGS 점 제거)
    points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs = remove_duplicates(
        points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs
    )

    # 6. 3DGS 점에 Outlier 제거 적용
    points_3dgs, colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs = remove_outliers_3dgs(
        points_3dgs, colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs
    )

    # 7. 병합
    points_merged = np.vstack((points_pointcept, points_3dgs))
    colors_merged = np.vstack((colors_pointcept, colors_3dgs))
    normals_merged = np.vstack((normals_pointcept, normals_3dgs))
    labels_merged = np.concatenate((labels_pointcept, labels_3dgs))
    labels200_merged = np.concatenate((labels200_pointcept, labels200_3dgs))
    instances_merged = np.concatenate((instances_pointcept, instances_3dgs))
    print(f"Final merged points: {len(points_merged)} (Pointcept: {len(points_pointcept)}, 3DGS: {len(points_3dgs)})")

    # 8. Pointcept 포맷으로 저장
    save_dict = {
        'coord': points_merged.astype(np.float32),
        'color': colors_merged.astype(np.uint8),
        'normal': normals_merged.astype(np.float32),
        'segment20': labels_merged.astype(np.int64),
        'segment200': labels200_merged.astype(np.int64),
        'instance': instances_merged.astype(np.int64)
    }
    os.makedirs(output_dir, exist_ok=True)
    for key, value in save_dict.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), value)
        print(f"Saved {key}.npy to {output_dir}")

def process_single_scene(scene, input_root, output_root, path_3dgs_root, prune_methods, prune_params, k_neighbors=5, use_label_consistency=True):
    """
    단일 scene을 처리하는 함수.
    
    Args:
        scene (str): 처리할 scene 이름.
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리.
        output_root (str): 출력 디렉토리.
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
    """
    try:
        # 입력 경로
        pointcept_dir = os.path.join(input_root, "train", scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        # 출력 경로
        output_dir = os.path.join(output_root, "train", scene)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"3DGS PLY file not found: {path_3dgs}")
            return

        # 병합 및 Pointcept 포맷으로 저장
        merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, prune_methods, prune_params, k_neighbors, use_label_consistency)
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def process_scenes(input_root, output_root, scene_list, path_3dgs_root, prune_methods=None, prune_params=None, k_neighbors=5, use_label_consistency=True, num_workers=1):
    """
    주어진 scene 목록을 처리하여 Pointcept 포맷으로 저장.
    
    Args:
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리 (예: scannet).
        output_root (str): 출력 디렉토리 (예: scannet_merged).
        scene_list (list): 처리할 scene 이름 목록.
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
        num_workers (int): 병렬 처리 작업자 수 (사용하지 않음).
    """
    print(f"Processing train scenes (Total: {len(scene_list)} scenes)")
    # 순차적 처리
    for scene in tqdm(scene_list, desc="Processing train scenes", unit="scene"):
        process_single_scene(scene, input_root, output_root, path_3dgs_root, prune_methods, prune_params, k_neighbors, use_label_consistency)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ScanNet scenes with 3DGS merging and save in Pointcept format.")
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
        help="Path to the meta directory containing train_100_samples.txt (default: /home/knuvi/Desktop/song/gaussian-point-sampler/meta)",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of workers for parallel processing (not used in sequential mode)",
    )
    parser.add_argument(
        "--scale_ratio",
        default=0.3,
        type=float,
        help="Ratio of points to prune based on scale (top X%). If > 0, scale pruning is enabled.",
    )
    parser.add_argument(
        "--opacity_ratio",
        default=0.5,
        type=float,
        help="Ratio of points to prune based on opacity (bottom X%). If > 0, opacity pruning is enabled.",
    )
    parser.add_argument(
        "--enable_density",
        action="store_true",
        help="Enable density-based pruning",
    )
    parser.add_argument(
        "--enable_pointcept_distance",
        action="store_true",
        help="Enable Pointcept distance-based pruning",
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
        help="Number of neighbors for label voting and consistency check",
    )
    parser.add_argument(
        "--use_label_consistency",
        action="store_true",
        default=True,
        help="Use label consistency filtering",
    )
    args = parser.parse_args()

    # Scene 목록 로드
    with open(os.path.join(args.meta_root, "train100_samples.txt")) as f:
        train_scenes = f.read().splitlines()

    # Pruning 방법 설정
    prune_methods = {
        'scale': args.scale_ratio > 0,
        'scale_ratio': args.scale_ratio,
        'opacity': args.opacity_ratio > 0,
        'opacity_ratio': args.opacity_ratio,
        'density': args.enable_density,
        'pointcept_distance': args.enable_pointcept_distance,
        'normal': args.enable_normal,
        'sor': args.enable_sor
    }

    # Pruning 하이퍼파라미터 설정
    prune_params = {
        'density_eps': 0.05,
        'density_min_points': 50,
        'pointcept_max_distance': 0.0001,
        'normal_k_neighbors': 5,
        'normal_cos_threshold': 0.9,
        'sor_nb_neighbors': 40,
        'sor_std_ratio': 0.1
    }

    # Train scenes 처리
    process_scenes(
        args.input_root,
        args.output_root,
        train_scenes,
        args.path_3dgs_root,
        prune_methods,
        prune_params,
        k_neighbors=args.k_neighbors,
        use_label_consistency=args.use_label_consistency,
        num_workers=args.num_workers
    )