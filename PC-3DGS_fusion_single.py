import os, json
import argparse
import numpy as np
from plyfile import PlyData, PlyElement

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data, update_3dgs_attributes, voxelize_3dgs, save_ply
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes, remove_duplicates, pruning_3dgs_attr, pdistance_pruning

def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, exp, prune_methods=None, prune_params=None, k_neighbors=5, ignore_threshold=0.6, voxel_size=0.02, use_features=('scale', 'opacity', 'rotation')):
    """
    Pointcept Point Cloud와 3DGS Point Cloud를 병합하고, 3DGS 속성을 전달하여 PLY 파일로 저장.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/val/scene0011_00).
        path_3dgs (str): 3DGS Point Cloud 경로.
        output_dir (str): 출력 디렉토리 (예: test_samples10).
        exp (str): 실험 이름 (파일 저장 시 사용).
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
    points_3dgs, normals_3dgs, raw_features_3dgs = load_3dgs_data(path_3dgs)

    # 3. 3DGS 속성 전처리
    print("Preprocessing 3DGS attributes...")
    features_3dgs = preprocess_3dgs_attributes(raw_features_3dgs)

    # 4. 3DGS pdistance Pruning
    print("Pruning 3DGS points with pdistance...")
    points_3dgs, features_3dgs = pdistance_pruning(
        points_3dgs, features_3dgs, points_pointcept, prune_params
    )

    # 5. 3DGS 데이터 Voxelization (옵션)
    voxelize = voxel_size != 0  # voxel_size가 0이 아니면 voxelize 활성화
    if voxelize:
        print("Applying Voxelization to 3DGS points...")
        points_3dgs, features_3dgs = voxelize_3dgs(
            points_3dgs, features_3dgs, voxel_size=voxel_size, k_neighbors=k_neighbors
        )
    else:
        print("Skipping Voxelization for 3DGS data.")

    # 6. 3DGS-attr transfer
    print("Augmenting Pointcept points with 3DGS attributes...")
    features_pointcept, filtered_features_3dgs = augment_pointcept_with_3dgs_attributes(
        points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors, use_features=use_features
    )
    
    # 7. 3DGS-attr Pruning
    print("Pruning 3DGS points with attributes...")
    points_3dgs, features_3dgs = pruning_3dgs_attr(
        points_3dgs, filtered_features_3dgs, features_3dgs, prune_methods, prune_params
    )

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

    # 8.5. -1 라벨 3DGS 점 비율 출력
    ignore_count = np.sum(labels_3dgs == -1)
    total_count = len(labels_3dgs)
    ignore_ratio = ignore_count / total_count if total_count > 0 else 0
    print(f"3DGS points with label -1: {ignore_count}/{total_count} (Ratio: {ignore_ratio:.4f})")

    # 9. 중복 점 제거 (3DGS 점 제거)
    print("Removing duplicate points...")
    points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, features_3dgs = remove_duplicates(
        points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, features_3dgs
    )

    # 10. 병합
    print("Merging Pointcept and 3DGS points...")
    points_merged = np.vstack((points_pointcept, points_3dgs))
    colors_merged = np.vstack((colors_pointcept, colors_3dgs))
    normals_merged = np.vstack((normals_pointcept, normals_3dgs))
    labels_merged = np.concatenate((labels_pointcept, labels_3dgs))
    labels200_merged = np.concatenate((labels200_pointcept, labels200_3dgs))
    instances_merged = np.concatenate((instances_pointcept, instances_3dgs))
    features_merged = np.vstack((features_pointcept, features_3dgs))  # 3DGS 점의 원래 속성 유지
    print(f"Final merged points: {len(points_merged)} (Pointcept: {len(points_pointcept)}, 3DGS: {len(points_3dgs)})")

    # 11. PLY 파일로 저장
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

def process_single_scene(scene, input_root, output_root, exp, path_3dgs_root, prune_methods, prune_params, k_neighbors=5, ignore_threshold=0.6, voxel_size=0.02, use_features=('scale', 'opacity', 'rotation')):
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
        k_neighbors (int): 라벨 복사 및 속성 전달에 사용할 이웃 점 개수.
        ignore_threshold (float): ignore_index(-1) 라벨의 비율이 이 값 이상이면 결과 라벨을 -1로 설정.
        voxel_size (float): Voxel 크기 (기본값: 0.02m). 0이 아니면 voxelization 적용.
        use_features (tuple): 사용할 3DGS 속성 ('scale', 'opacity', 'rotation').
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
            k_neighbors=k_neighbors, ignore_threshold=ignore_threshold, voxel_size=voxel_size, use_features=use_features
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
        "--output_root",
        default="/home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10",
        help="Output path for processed data (default: /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10)",
    )
    parser.add_argument(
        "--exp",
        default="pdistance00005_vox002",
        help="Experiment name",
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
        "--rotation_ratio",
        default=0,
        type=float,
        help="Ratio of points to prune based on rotation (bottom X%). If > 0, rotation pruning is enabled.",
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
        "--ignore_threshold",
        default=0.6,
        type=float,
        help="Threshold for ignore_index(-1) label ratio",
    )
    parser.add_argument(
        "--use_features",
        nargs='+',
        default=['scale', 'opacity', 'rotation'],
        help="3DGS features to transfer (default: scale opacity rotation)",
    )

    args = parser.parse_args()

    # Config 파일 로드
    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    input_root = config['input_root']
    path_3dgs_root = config['path_3dgs_root']
    prune_params = config['prune_params']
    k_neighbors = prune_params['k_neighbors']
    prune_params['pointcept_max_distance'] = args.pdistance

    # Prune methods 설정
    prune_methods = {
        'scale': args.scale_ratio > 0,
        'scale_ratio': args.scale_ratio,
        'rotation': args.rotation_ratio > 0,
        'rotation_ratio': args.rotation_ratio,
        'opacity': args.opacity_ratio > 0,
        'opacity_ratio': args.opacity_ratio,
        'pointcept_distance': args.pdistance > 0,
    }

    # 단일 Scene 처리
    process_single_scene(
        args.scene,
        input_root,
        args.output_root,
        args.exp,
        path_3dgs_root,
        prune_methods,
        prune_params,
        k_neighbors=k_neighbors,
        ignore_threshold=args.ignore_threshold,
        voxel_size=args.voxel_size,
        use_features=tuple(args.use_features)
    )