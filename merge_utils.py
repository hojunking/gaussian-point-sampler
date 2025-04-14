import os
import numpy as np
import open3d as o3d
from plyfile import PlyData
from sklearn.neighbors import NearestNeighbors

# utils.py와 attribute_utils.py에서 필요한 함수 가져오기
from utils import load_pointcept_data, remove_duplicates, update_3dgs_attributes
from attribute_utils import preprocessing_attr, quaternion_to_direction, augment_pointcept_with_3dgs_attributes
from pruning import sigmoid

def load_3dgs_data_for_merge(path_3dgs):
    """
    3DGS 데이터를 로드 (color, normal 제외).
    
    Args:
        path_3dgs (str): 3DGS PLY 파일 경로.
    
    Returns:
        tuple: (points_3dgs, raw_features_3dgs)
            - points_3dgs (np.ndarray): 3DGS 점 좌표 [N, 3].
            - raw_features_3dgs (np.ndarray): 3DGS 속성 [N, 8] (scale_x, scale_y, scale_z, opacity, rot_w, rot_x, rot_y, rot_z).
    """
    with open(path_3dgs, 'rb') as f:
        ply_data_3dgs = PlyData.read(f)
    vertex_data_3dgs = ply_data_3dgs['vertex']
    
    # 좌표
    points_3dgs = np.stack([vertex_data_3dgs['x'], vertex_data_3dgs['y'], vertex_data_3dgs['z']], axis=-1)
    
    # 속성
    scales_3dgs = np.stack([vertex_data_3dgs['scale_0'], vertex_data_3dgs['scale_1'], vertex_data_3dgs['scale_2']], axis=-1)
    opacity_3dgs = vertex_data_3dgs['opacity']
    rotation_3dgs = np.stack([vertex_data_3dgs['rot_0'], vertex_data_3dgs['rot_1'], vertex_data_3dgs['rot_2'], vertex_data_3dgs['rot_3']], axis=-1)
    raw_features_3dgs = np.hstack((scales_3dgs, opacity_3dgs[:, np.newaxis], rotation_3dgs))
    
    print(f"Loaded 3DGS data from {path_3dgs}: {points_3dgs.shape[0]} points")
    return points_3dgs, raw_features_3dgs

def prune_3dgs_for_merge(points_3dgs, features_3dgs_processed, points_pointcept, prune_methods, prune_params):
    """
    3DGS 점에 대해 다양한 pruning 방법을 적용 (merge용).
    
    Args:
        points_3dgs (np.ndarray): 3DGS 점 좌표 [N, 3].
        features_3dgs_processed (np.ndarray): 전처리된 3DGS 속성 [N, D].
        points_pointcept (np.ndarray): Pointcept 점 좌표 [M, 3].
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
    
    Returns:
        tuple: Pruning된 (points_3dgs, features_3dgs_processed, mask).
    """
    print(f"Initial 3DGS points: {len(points_3dgs)}")
    num_points = len(points_3dgs)
    mask = np.ones(num_points, dtype=bool)  # 초기 mask: 모든 점 포함
    
    # Pointcept Distance Pruning
    if prune_methods.get('pointcept_distance', False):
        pcd_pointcept = o3d.geometry.PointCloud()
        pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
        tree = o3d.geometry.KDTreeFlann(pcd_pointcept)
        
        # 벡터화된 거리 계산
        distances = np.zeros(num_points, dtype=np.float64)
        for i in range(num_points):
            _, _, dist = tree.search_knn_vector_3d(points_3dgs[i], 1)
            distances[i] = dist[0]
        
        distance_mask = distances <= prune_params['pointcept_max_distance']
        mask = mask & distance_mask  # 전체 mask 업데이트
        print(f"Pointcept Distance Pruning: Before {num_points} points, Pruned {np.sum(~distance_mask)} points with distance > {prune_params['pointcept_max_distance']:.5f}, After {np.sum(mask)} points")
        
        # 데이터 필터링
        points_3dgs = points_3dgs[distance_mask]
        features_3dgs_processed = features_3dgs_processed[distance_mask]
        
        if len(points_3dgs) == 0:
            print("No 3DGS points remaining after Pointcept Distance Pruning. Skipping further pruning.")
            return points_3dgs, features_3dgs_processed, mask

    # Scale-based Pruning
    if prune_methods.get('scale', False) and features_3dgs_processed.shape[0] > 0:
        scales = features_3dgs_processed[:, 0:3]
        scale_magnitudes = np.linalg.norm(scales, axis=-1)
        threshold = np.percentile(scale_magnitudes, 100 * (1 - prune_methods['scale_ratio']))
        scale_mask = scale_magnitudes <= threshold
        
        # 전체 mask에서 현재 데이터에 해당하는 부분만 업데이트
        current_mask = mask[mask]  # 현재 남아있는 점들에 대한 mask
        current_mask = current_mask & scale_mask
        mask[mask] = current_mask  # 업데이트된 mask를 원래 mask에 반영
        
        print(f"Scale Pruning: Pruned {np.sum(~scale_mask)} points with scale > {threshold:.4f}, Remaining {np.sum(scale_mask)} points")
        
        points_3dgs = points_3dgs[scale_mask]
        features_3dgs_processed = features_3dgs_processed[scale_mask]

    # Opacity-based Pruning
    if prune_methods.get('opacity', False) and features_3dgs_processed.shape[0] > 0:
        opacities = features_3dgs_processed[:, 3]
        threshold = np.percentile(opacities, 100 * prune_methods['opacity_ratio'])
        opacity_mask = opacities >= threshold
        
        # 전체 mask에서 현재 데이터에 해당하는 부분만 업데이트
        current_mask = mask[mask]  # 현재 남아있는 점들에 대한 mask
        current_mask = current_mask & opacity_mask
        mask[mask] = current_mask  # 업데이트된 mask를 원래 mask에 반영
        
        print(f"Opacity Pruning: Pruned {np.sum(~opacity_mask)} points with opacity < {threshold:.4f}, Remaining {np.sum(opacity_mask)} points")
        
        points_3dgs = points_3dgs[opacity_mask]
        features_3dgs_processed = features_3dgs_processed[opacity_mask]

    print(f"Final 3DGS points after pruning: {len(points_3dgs)} (Pruned {num_points - len(points_3dgs)} points in total)")
    return points_3dgs, features_3dgs_processed, mask

def merge_pointcept_with_3dgs(pointcept_dir, path_3dgs, output_dir, prune_methods=None, prune_params=None, k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6, filter_ignore_label=False, use_features=('scale', 'opacity', 'rotation'), aggregation='mean'):
    """
    Pointcept Point Cloud와 3DGS Point Cloud를 병합하고 PLY 파일로 저장.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/train/scene0000_00).
        path_3dgs (str): 3DGS Point Cloud 경로.
        output_dir (str): 출력 디렉토리 (예: scannet_merged/train).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
        ignore_threshold (float): ignore_index(-1) 라벨의 비율이 이 값 이상이면 결과 라벨을 -1로 설정.
        filter_ignore_label (bool): -1 라벨 3DGS 점을 필터링할지 여부.
        use_features (tuple): 3DGS 속성 전이에 사용할 features ('scale', 'opacity', 'rotation').
        aggregation (str): 속성 집계 방식 ('mean', 'max', 'median').
    """
    # 1. Pointcept 데이터 로드 (.npy 파일에서)
    pointcept_data = load_pointcept_data(pointcept_dir)
    points_pointcept = pointcept_data['coord']
    colors_pointcept = pointcept_data['color']
    normals_pointcept = pointcept_data['normal']
    labels_pointcept = pointcept_data['segment20']
    labels200_pointcept = pointcept_data['segment200']
    instances_pointcept = pointcept_data['instance']

    # 2. 3DGS 데이터 로드 (속성 포함)
    points_3dgs, raw_features_3dgs = load_3dgs_data_for_merge(path_3dgs)
    features_3dgs = preprocessing_attr(raw_features_3dgs)

    # Rotation을 미리 방향 벡터로 변환
    features_3dgs_processed = features_3dgs.copy()
    if 'rotation' in use_features:
        direction_3dgs = quaternion_to_direction(features_3dgs[:, 4:8])  # [N, 3]
        features_3dgs_processed = np.hstack((features_3dgs[:, 0:4], direction_3dgs))  # [N, 7] (scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z)
    else:
        features_3dgs_processed = features_3dgs[:, 0:4]  # rotation 제외, [N, 4] (scale_x, scale_y, scale_z, opacity)

    # 3. 3DGS 점 pruning
    points_3dgs, features_3dgs_processed, mask = prune_3dgs_for_merge(
        points_3dgs, features_3dgs_processed, points_pointcept, prune_methods, prune_params
    )

    # 4. 3DGS 점의 색상, 법선, 라벨을 복사
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
    features_3dgs_processed = features_3dgs_processed[mask]  # 속성도 필터링

    # 4.5. -1 라벨 3DGS 점 비율 출력
    ignore_count = np.sum(labels_3dgs == -1)
    total_count = len(labels_3dgs)
    ignore_ratio = ignore_count / total_count if total_count > 0 else 0
    print(f"3DGS points with label -1: {ignore_count}/{total_count} (Ratio: {ignore_ratio:.4f})")

    # 5. Pointcept 점에 3DGS 속성 전달
    augmented_features_pointcept = augment_pointcept_with_3dgs_attributes(
        points_pointcept, points_3dgs, features_3dgs_processed, k_neighbors=k_neighbors, use_features=use_features, aggregation=aggregation
    )

    # 6. 3DGS 점의 속성 준비 (Pointcept 점과 동일한 차원으로 맞춤)
    selected_features_3dgs = []
    feature_dim = features_3dgs_processed.shape[1]  # 동적으로 차원 확인
    if 'scale' in use_features:
        if feature_dim >= 3:
            selected_features_3dgs.append(features_3dgs_processed[:, 0:3])  # scale_x, scale_y, scale_z
        else:
            print(f"Warning: features_3dgs_processed has dimension {feature_dim}, cannot extract scale (requires at least 3 dimensions)")
    if 'opacity' in use_features:
        if feature_dim >= 4:
            selected_features_3dgs.append(features_3dgs_processed[:, 3:4])  # opacity
        else:
            print(f"Warning: features_3dgs_processed has dimension {feature_dim}, cannot extract opacity (requires at least 4 dimensions)")
    if 'rotation' in use_features:
        if feature_dim >= 7:
            selected_features_3dgs.append(features_3dgs_processed[:, 4:7])  # dir_x, dir_y, dir_z
        else:
            print(f"Warning: features_3dgs_processed has dimension {feature_dim}, cannot extract rotation (requires at least 7 dimensions)")
    if selected_features_3dgs:
        augmented_features_3dgs = np.concatenate(selected_features_3dgs, axis=-1)  # [N, D]
    else:
        augmented_features_3dgs = np.zeros((len(points_3dgs), 0), dtype=np.float32)

    # 7. 중복 점 제거 (3DGS 점 제거)
    points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, augmented_features_3dgs = remove_duplicates(
        points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, augmented_features_3dgs
    )

    # 8. 병합
    points_merged = np.vstack((points_pointcept, points_3dgs))
    colors_merged = np.vstack((colors_pointcept, colors_3dgs))
    normals_merged = np.vstack((normals_pointcept, normals_3dgs))
    labels_merged = np.concatenate((labels_pointcept, labels_3dgs))
    labels200_merged = np.concatenate((labels200_pointcept, labels200_3dgs))
    instances_merged = np.concatenate((instances_pointcept, instances_3dgs))
    augmented_features_merged = np.vstack((augmented_features_pointcept, augmented_features_3dgs))
    print(f"Final merged points: {len(points_merged)} (Pointcept: {len(points_pointcept)}, 3DGS: {len(points_3dgs)})")

    # 9. Pointcept 포맷으로 저장 (.npy 파일)
    save_dict = {
        'coord': points_merged.astype(np.float32),
        'color': colors_merged.astype(np.uint8),
        'normal': normals_merged.astype(np.float32),
        'segment20': labels_merged.astype(np.int64),
        'segment200': labels200_merged.astype(np.int64),
        'instance': instances_merged.astype(np.int64),
        'features': augmented_features_merged.astype(np.float32)
    }
    os.makedirs(output_dir, exist_ok=True)
    for key, value in save_dict.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), value)
        print(f"Saved {key}.npy to {output_dir}")