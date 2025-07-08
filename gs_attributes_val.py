


import os, json
import argparse
import numpy as np

# 기존 모듈 임포트
from utils import load_pointcept_data, load_3dgs_data
from fusion_utils import augment_pointcept_with_3dgs_attributes, preprocess_3dgs_attributes
from attribute_utils import save_scale_rot_opacity_ply

def visual_3dgs_attr(pointcept_dir, path_3dgs, output_dir, exp, k_neighbors=5):
    # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    labels_pointcept = pointcept_data['segment20']

    # 2. 3DGS 데이터 로드
    points_3dgs, raw_features_3dgs = load_3dgs_data(path_3dgs)

    # 3. 3DGS 속성 전처리
    print("Preprocessing 3DGS attributes...")
    features_3dgs = preprocess_3dgs_attributes(raw_features_3dgs)

    # 6. 3DGS-attr transfer
    print("Augmenting Pointcept points with 3DGS attributes...")
    features_pointcept  = augment_pointcept_with_3dgs_attributes(
        points_pointcept, points_3dgs, features_3dgs, k_neighbors=k_neighbors
    )

    print("Saving attributes to PLY files...")
    output_dir_path = os.path.join(output_dir, exp)
    os.makedirs(output_dir_path, exist_ok=True)
    
    save_scale_rot_opacity_ply(points_pointcept, features_pointcept,
                                labels_pointcept, exp,
                                output_dir_path, exclude_labels=[-1, 0, 1])
    #save_scale_rot_opacity_ply(points_3dgs, features_3dgs, labels_pointcept, exp, output_dir_path)

    

def process_single_scene(scene, input_root, output_root, exp, path_3dgs_root,k_neighbors=5):
    try:
        # 입력 경로
        pointcept_dir = os.path.join(input_root, "val", scene)
        path_3dgs = os.path.join(path_3dgs_root, scene, "point_cloud.ply")
        # 출력 경로

        if not os.path.exists(os.path.join(pointcept_dir, "coord.npy")):
            print(f"Pointcept data not found: {pointcept_dir}")
            return
        if not os.path.exists(path_3dgs):
            print(f"3DGS PLY file not found: {path_3dgs}, required for '3dgs' method.")
            return
        # 병합 및 PLY 파일로 저장
        visual_3dgs_attr(pointcept_dir, path_3dgs, output_root, exp, k_neighbors=k_neighbors)
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
        "--path_3dgs_root",
        default="/home/knuvi/Desktop/song/gaussian-point-sampler/test/pup_gs",
        help="Output path for processed data (default: /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10)",
    )
    parser.add_argument(
        "--exp",
        default="pdistance00005_vox002",
        help="Experiment name",
    )
    args = parser.parse_args()

    # Config 파일 로드
    config_path = './config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    input_root = config['input_root']

    #path_3dgs_root = config['path_3dgs_root']
    path_3dgs_root = args.path_3dgs_root  # 사용자 지정 3DGS 경로 사용
    
    meta_root = config['meta_root']
    k_neighbors = 10

    # 단일 Scene 처리
    process_single_scene(
        args.scene,
        input_root,
        args.output_root,
        args.exp,
        path_3dgs_root,
        k_neighbors=k_neighbors,
    )