import os, json
import argparse
import numpy as np

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes
from boundary_utils import boundary_labeling, save_boundary_ply

def boudnary_labeling_with_3dgs(pointcept_dir, path_3dgs, output_dir, exp, prune_methods=None, prune_params=None, k_neighbors=5):
     # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    labels_pointcept = pointcept_data['segment20']
    
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
    
    output_dir_path = os.path.join(output_dir, exp)
    os.makedirs(output_dir_path, exist_ok=True)
    output_ply_path = os.path.join(output_dir_path, f"{exp}.ply")
    save_boundary_ply(points_pointcept, labels_pointcept, boundary, output_ply_path) 


def process_single_scene(scene, input_root, output_root, exp, path_3dgs_root, prune_methods, prune_params, k_neighbors=5):
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
        boudnary_labeling_with_3dgs(
            pointcept_dir, path_3dgs, output_dir, exp, prune_methods, prune_params,
            k_neighbors=k_neighbors
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
    scale_ratio, rotation_ratio, opacity_ratio  = args.attr_pruning_ratio

    prune_methods = {
        'scale': scale_ratio > 0,
        'scale_ratio': scale_ratio,
        'opacity': opacity_ratio > 0,
        'opacity_ratio': opacity_ratio,
        'rotation': rotation_ratio > 0,
        'rotation_ratio': rotation_ratio,
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
    )