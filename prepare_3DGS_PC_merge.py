import os
import shutil
import glob
import argparse
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

def copy_scene_data(scene_path, raw_data_path, output_path, output_root_path_3dgs):
    #print("working?")
    """
    각 Scene 데이터를 지정된 디렉터리 구조로 복사.
    
    Args:
        scene_path (str): 원본 Scene 폴더 경로 (예: raw_data_path/scans/scene0000_00).
        raw_data_path (str): 원본 ScanNet 데이터 루트 경로.
        output_path (str): 출력 루트 경로.
        output_root_path_3dgs (str): 3DGS 데이터 루트 경로.
    """
    # Scene ID 추출
    scene_id = os.path.basename(scene_path)  # 예: scene0000_00

    # 출력 디렉터리 경로 설정
    output_scene_path = os.path.join(output_path, "scans", scene_id)
    output_3dgs_path = os.path.join(output_path, "3dgs_output", scene_id)

    # 출력 디렉터리 생성
    os.makedirs(output_scene_path, exist_ok=True)
    os.makedirs(output_3dgs_path, exist_ok=True)

    # 1. ScanNet 필수 파일 복사
    required_files = [
        f"{scene_id}_vh_clean_2.ply",
        f"{scene_id}_vh_clean_2.labels.ply",
        f"{scene_id}_vh_clean_2.0.010000.segs.json",
        f"{scene_id}.aggregation.json",
        f"{scene_id}.txt"
    ]

    for file_name in required_files:
        src_file = os.path.join(scene_path, file_name)
        dst_file = os.path.join(output_scene_path, file_name)
        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")
        else:
            print(f"Warning: {src_file} does not exist, skipping...")

    # 2. 3DGS Point Cloud 복사
    src_3dgs_file = os.path.join(
        output_root_path_3dgs, "3dgs_pc_scannet", scene_id,
        "point_cloud", "iteration_10000", "point_cloud.ply"
    )
    dst_3dgs_file = os.path.join(output_3dgs_path, "point_cloud.ply")

    if os.path.exists(src_3dgs_file):
        shutil.copy2(src_3dgs_file, dst_3dgs_file)
        print(f"Copied {src_3dgs_file} to {dst_3dgs_file}")
    else:
        print(f"Warning: {src_3dgs_file} does not exist, skipping...")

def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="Prepare ScanNet and 3DGS data with specified directory structure.")
    parser.add_argument(
        "--raw_data_path",
        #required=True,
        help="Path to the raw ScanNet dataset (containing scans/ folder).",
        default="/media/knuvi/Desk SSD/hojun"
    )
    parser.add_argument(
        "--output_root_path_3dgs",
        #required=True,
        help="Path to the 3DGS output root (containing 3dgs_pc_scannet/ folder).",
        default="/media/knuvi/Desk SSD/hojun"
    )
    parser.add_argument(
        "--output_path",
        #required=True,
        help="Output path where the restructured data will be saved.",
        default="/home/knuvi/Desktop/song/data/3dgs_scans"
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Number of workers for parallel processing."
    )
    args = parser.parse_args()

    # 경로 설정
    raw_data_path = args.raw_data_path
    output_root_path_3dgs = args.output_root_path_3dgs
    output_path = args.output_path

    # ScanNet Scene 목록 로드
    scene_paths = sorted(glob.glob(os.path.join(raw_data_path, "scans", "scene*")))
    #print(scene_paths)
    # 출력 디렉터리 초기화
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists, removing...")
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    # 병렬 처리로 Scene 데이터 복사
    print("Copying data to new directory structure...")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        executor.map(
            copy_scene_data,
            scene_paths,
            repeat(raw_data_path),
            repeat(output_path),
            repeat(output_root_path_3dgs)
        )

    print(f"Data preparation completed! Output saved to {output_path}")

if __name__ == "__main__":
    main()