import numpy as np
import argparse, json
from pruning import prune_3dgs
from utils import load_pointcept_data, load_3dgs_data, update_3dgs_attributes, remove_duplicates, save_ply, voxelize_3dgs
import os


def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, exp, prune_methods=None, prune_params=None, k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6, filter_ignore_label=False, voxelize=True, voxel_size=0.02):

    """
    Pointcept Point Cloud와 3DGS Point Cloud를 병합하고 PLY 파일로 저장.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/train/scene0000_00).
        path_3dgs (str): 3DGS Point Cloud 경로.
        output_dir (str): 출력 디렉토리 (예: scannet_merged/train).
        exp (str): 실험 이름 (예: scene0000_00).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
        ignore_threshold (float): ignore_index(-1) 라벨의 비율이 이 값 이상이면 결과 라벨을 -1로 설정.
        filter_ignore_label (bool): -1 라벨 3DGS 점을 필터링할지 여부.
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

    # 3. 3DGS 데이터 Voxelization (옵션)
    if voxelize:
        points_3dgs, normals_3dgs, vertex_data_3dgs = voxelize_3dgs(
            points_3dgs, normals_3dgs, vertex_data_3dgs, voxel_size=voxel_size
        )
    else:
        print("Skipping Voxelization for 3DGS data.")

    # 4. 3DGS 점 pruning (normals_pointcept 전달)
    points_3dgs, normals_3dgs, vertex_data_3dgs = prune_3dgs(
        vertex_data_3dgs, points_3dgs, normals_3dgs, points_pointcept, normals_pointcept, prune_methods, prune_params
    )

    # 5. 3DGS 점의 색상, 법선, 라벨을 복사
    colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask = update_3dgs_attributes(
        points_3dgs, points_pointcept, colors_pointcept, normals_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept,
        k_neighbors=k_neighbors, use_label_consistency=use_label_consistency, ignore_threshold=ignore_threshold
    )

    # 라벨 일관성 필터링 적용
    points_3dgs = points_3dgs[mask]
    colors_3dgs = colors_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    labels_3dgs = labels_3dgs[mask]
    labels200_3dgs = labels200_3dgs[mask]
    instances_3dgs = instances_3dgs[mask]

    # 5.5. -1 라벨 3DGS 점 비율 출력
    ignore_count = np.sum(labels_3dgs == -1)
    total_count = len(labels_3dgs)
    ignore_ratio = ignore_count / total_count if total_count > 0 else 0
    print(f"3DGS points with label -1: {ignore_count}/{total_count} (Ratio: {ignore_ratio:.4f})")

    # 6. 중복 점 제거 (3DGS 점 제거)
    points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs = remove_duplicates(
        points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs
    )

    # 6. 병합
    points_merged = np.vstack((points_pointcept, points_3dgs))
    colors_merged = np.vstack((colors_pointcept, colors_3dgs))
    labels_merged = np.concatenate((labels_pointcept, labels_3dgs))
    print(f"Final merged points: {len(points_merged)} (Pointcept: {len(points_pointcept)}, 3DGS: {len(points_3dgs)})")

    # 7. PLY 파일로 저장
    output_dir_path = os.path.join(output_dir, exp)
    os.makedirs(output_dir_path, exist_ok=True)
    output_ply_path = os.path.join(output_dir_path, f"{exp}.ply")
    save_ply(
        points_merged, colors_merged, labels_merged, output_ply_path,
        save_separate_labels=True,
        points_pointcept=points_pointcept,
        colors_pointcept=colors_pointcept,
        points_3dgs=points_3dgs
    )

    # 3DGS 점 별도 저장 (라벨 포함)
    dgs_ply_path = os.path.join(output_dir_path, f"{exp}_3dgs.ply")
    save_ply(
        points_3dgs, colors_3dgs, labels_3dgs, dgs_ply_path,
        save_separate_labels=False
    )

def process_single_scene(scene, input_root, output_root, exp, path_3dgs_root, prune_methods, prune_params, k_neighbors=5, use_label_consistency=True, voxelize=True, voxel_size=0.02):
    """
    단일 scene을 처리하는 함수.
    
    Args:
        scene (str): 처리할 scene 이름.
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리.
        output_root (str): 출력 디렉토리.
        exp (str): 실험 이름.
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
    """
    try:
        # 입력 경로
        pointcept_dir = os.path.join(input_root, "val", scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        # 출력 경로
        output_dir = os.path.join(output_root)

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"3DGS PLY file not found: {path_3dgs}")
            return

        # 병합 및 PLY 파일로 저장
        merge_pointcept_with_3dgs(
            pointcept_dir, path_3dgs, output_dir, exp, prune_methods, prune_params,
            k_neighbors=k_neighbors, use_label_consistency=use_label_consistency, voxelize=True, voxel_size=0.02
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single ScanNet scene with 3DGS merging and save as PLY for visualization.")
    parser.add_argument(
        "--scene",
        default="scene0011_00",
        help="Scene to process (e.g., scene0011_00)",
    )
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
    parser.add_argument(
        "--exp",
        default="",
        help="Experiment name",
    )
    parser.add_argument(
        "--pdistance",
        default=0.001,
        type=float,
        help="Pointcept distance threshold",
    )
    parser.add_argument(
        "--voxelize",
        action="store_true",
        default=True,
        help="Apply voxelization to 3DGS data (default: True)",
    )
    parser.add_argument(
        "--voxel_size",
        default=0.02,
        type=float,
        help="Voxel size for 3DGS voxelization (default: 0.02m)",
    )
    args = parser.parse_args()

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

    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    prune_params = config['prune_params']
    prune_params['pointcept_max_distance'] = args.pdistance
    print(f'Pointcept distance threshold: {prune_params["pointcept_max_distance"]}')

    # 단일 Scene 처리
    process_single_scene(
        args.scene,
        args.input_root,
        args.output_root,
        args.exp,
        args.path_3dgs_root,
        prune_methods,
        prune_params,
        k_neighbors=args.k_neighbors,
        use_label_consistency=args.use_label_consistency,
        voxelize=args.voxelize,
        voxel_size=args.voxel_size
    )