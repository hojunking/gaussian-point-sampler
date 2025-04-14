import numpy as np
import argparse, json
from merge_utils import merge_pointcept_with_3dgs  # merge_utils에서 가져오기
from utils import load_pointcept_data
from tqdm import tqdm
import os

def process_single_scene(scene, input_root, output_root, split, path_3dgs_root, prune_methods, prune_params, k_neighbors=5, use_label_consistency=False, use_features=('scale', 'opacity', 'rotation'), aggregation='mean'):
    """
    단일 scene을 처리하는 함수.
    
    Args:
        scene (str): 처리할 scene 이름.
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리.
        output_root (str): 출력 디렉토리.
        split (str): 데이터 분할 (train/val).
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
        use_features (tuple): 3DGS 속성 전이에 사용할 features ('scale', 'opacity', 'rotation').
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
        merge_pointcept_with_3dgs(
            pointcept_dir, path_3dgs, output_dir, prune_methods, prune_params, k_neighbors, use_label_consistency,
            use_features=use_features, aggregation=aggregation
        )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

def process_scenes(input_root, output_root, scene_list, split, path_3dgs_root, prune_methods=None, prune_params=None, k_neighbors=5, use_label_consistency=True, num_workers=1, use_features=('scale', 'opacity', 'rotation'), aggregation='mean'):
    """
    주어진 scene 목록을 처리하여 Pointcept 포맷으로 저장.
    
    Args:
        input_root (str): 입력 Pointcept 데이터의 루트 디렉토리 (예: scannet).
        output_root (str): 출력 디렉토리 (예: scannet_merged).
        scene_list (list): 처리할 scene 이름 목록.
        split (str): 데이터 분할 (train/val).
        path_3dgs_root (str): 3DGS 데이터의 루트 디렉토리.
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
        num_workers (int): 병렬 처리 작업자 수 (사용하지 않음).
        use_features (tuple): 3DGS 속성 전이에 사용할 features ('scale', 'opacity', 'rotation').
        aggregation (str): 속성 집계 방식 ('mean', 'max', 'median').
    """
    print(f"Processing {split} scenes (Total: {len(scene_list)} scenes)")
    # 순차적 처리
    for scene in tqdm(scene_list, desc=f"Processing {split} scenes", unit="scene"):
        process_single_scene(
            scene, input_root, output_root, split, path_3dgs_root, prune_methods, prune_params, k_neighbors, use_label_consistency,
            use_features=use_features, aggregation=aggregation
        )

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
        "--split",
        default="train",
        choices=["train", "val"],
        help="Data split to process (train/val) (default: train)",
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
        default=6,
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
        "--pdistance",
        default=0.001,
        type=float,
        help="Pointcept distance for pruning",
    )
    parser.add_argument(
        "--use_features",
        nargs='+',
        default=['scale', 'opacity', 'rotation'],
        choices=['scale', 'opacity', 'rotation'],
        help="Features to use for 3DGS attribute transfer (default: scale opacity rotation)",
    )
    parser.add_argument(
        "--aggregation",
        default='mean',
        choices=['mean', 'max', 'median'],
        help="Aggregation method for 3DGS attribute transfer (default: mean)",
    )

    args = parser.parse_args()

    # Scene 목록 로드
    with open(os.path.join(args.meta_root, f"{args.split}100_samples.txt")) as f:
        scenes = f.read().splitlines()

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
    p_distance = prune_params['pointcept_max_distance'] = args.pdistance
    print(f'p_distance: {p_distance}')
    
    # Scenes 처리
    process_scenes(
        args.input_root,
        args.output_root,
        scenes,
        args.split,
        args.path_3dgs_root,
        prune_methods,
        prune_params,
        k_neighbors=args.k_neighbors,
        use_label_consistency=args.use_label_consistency,
        num_workers=args.num_workers,
        use_features=tuple(args.use_features),  # 리스트를 튜플로 변환
        aggregation=args.aggregation
    )