# fusion_utils.py
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils import process_rotation


def preprocess_3dgs_attributes(raw_features_3dgs, normalize_scale=True):
    
    # 속성 분리
    scale = raw_features_3dgs[:, 0:3]  # [N, 3] (scale_x, scale_y, scale_z)
    opacity = raw_features_3dgs[:, 3:4]  # [N, 1] (opacity)
    rotation = raw_features_3dgs[:, 4:8]  # [N, 4] (rot_w, rot_x, rot_y, rot_z)

    # Scale: 로그 변환 해제
    scale_processed = np.exp(scale)  # log(scale) -> scale
    scale_processed = np.nan_to_num(scale_processed, nan=1e-6, posinf=1e-6, neginf=1e-6)
    scale_processed = np.maximum(scale_processed, 1e-6)  # 양수 보장

    # # 추가 로그 변환으로 긴 꼬리 완화
    # scale_processed = np.log1p(scale_processed)
    # if normalize_scale:
    #     scale_min = scale_processed.min()
    #     scale_max = scale_processed.max()
    #     if scale_max > scale_min:  # 분모가 0이 되지 않도록
    #         scale_processed = (scale_processed - scale_min) / (scale_max - scale_min)
    #     else:
    #         scale_processed = np.zeros_like(scale_processed)  # 모든 값이 동일하면 0으로 설정
    # Opacity: 시그모이드 변환
    opacity_processed = 1 / (1 + np.exp(-opacity))  # [-∞, ∞] -> [0, 1]
    opacity_processed = np.nan_to_num(opacity_processed, nan=0.0, posinf=1.0, neginf=0.0)
    opacity_processed = np.clip(opacity_processed, 0.0, 1.0)  # [0, 1]로 클리핑

    # # 추가 로그 변환으로 긴 꼬리 완화
    # opacity_processed = np.log1p(opacity_processed)
    # opacity_min = opacity_processed.min()
    # opacity_max = opacity_processed.max()
    # if opacity_max > opacity_min:
    #     opacity_processed = (opacity_processed - opacity_min) / (opacity_max - opacity_min)
    # else:
    #     opacity_processed = np.zeros_like(opacity_processed)
    
    # 후처리된 속성 결합
    features_3dgs = np.hstack((scale_processed, opacity_processed, rotation))  # [N, 8]
    return features_3dgs


def quaternion_to_direction(quaternions):
    """(N, 4) 쿼터니언 배열을 (N, 3) 방향 벡터 배열로 변환합니다."""
    # 쿼터니언 정규화 (입력이 이미 정규화되었다고 가정하면 생략 가능)
    norm = np.linalg.norm(quaternions, axis=1, keepdims=True)
    # 0으로 나누는 것을 방지
    safe_norm = np.where(norm == 0, 1, norm)
    quaternions = quaternions / safe_norm

    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # 기준 벡터 [0, 0, 1]에 회전 적용
    directions = np.zeros((len(quaternions), 3))
    directions[:, 0] = 2 * (x * z + w * y)
    directions[:, 1] = 2 * (y * z - w * x)
    directions[:, 2] = 1 - 2 * (x * x + y * y)

    # 최종 방향 벡터 정규화
    dir_norm = np.linalg.norm(directions, axis=1, keepdims=True)
    safe_dir_norm = np.where(dir_norm == 0, 1, dir_norm)
    return directions / safe_dir_norm

def get_weighted_direction(neighbor_quaternions, weights):
    neighbor_directions = quaternion_to_direction(neighbor_quaternions)  # 결과 shape: [k, 3]

    # 2. 방향 벡터들에 대해 가중 평균을 계산합니다.
    # weights [k] -> [k, 1]로 만들어 브로드캐스팅 후 가중합 계산
    weighted_direction = np.sum(neighbor_directions * weights[:, np.newaxis], axis=0)

    # 3. 최종 방향 벡터를 정규화하여 반환합니다.
    final_norm = np.linalg.norm(weighted_direction)
    if final_norm > 1e-8:
        return weighted_direction / final_norm
    else:
        # 모든 벡터가 상쇄되어 0이 된 경우, 기본 방향 벡터를 반환합니다.
        return np.array([0, 0, 1], dtype=np.float32)


def augment_pointcept_with_3dgs_attributes(points_pointcept, points_3dgs, features_3dgs, k_neighbors=5):
    
    # KNN으로 이웃 찾기
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points_3dgs)
    distances, indices = nbrs.kneighbors(points_pointcept)

    weights = 1.0 / (distances + 1e-8) 
    weights = weights / np.sum(weights, axis=1, keepdims=True) # [M, k]

    # 사용할 features 선택 (Pointcept에 전달용)
    selected_features_pointcept = []
    feature_idx = 0

    # Scale 속성 (Pointcept에 전달)
    scale_3dgs = features_3dgs[:, feature_idx:feature_idx + 3]  # [N, 3]
    scale_neighbors = scale_3dgs[indices]  # [M, k_neighbors, 3]

    scale_augmented = np.sum(scale_neighbors * weights[..., np.newaxis], axis=1)  # [M, 3]
    selected_features_pointcept.append(scale_augmented)
    feature_idx += 3

    # Opacity 속성 (Pointcept에 전달)
    opacity_3dgs = features_3dgs[:, feature_idx:feature_idx + 1]  # [N, 1]
    opacity_neighbors = opacity_3dgs[indices]  # [M, k_neighbors, 1]

    opacity_augmented = np.sum(opacity_neighbors * weights[..., np.newaxis], axis=1)  # [M, 1]
    selected_features_pointcept.append(opacity_augmented)
    feature_idx += 1

    # Rotation 속성 (Pointcept에 전달)
    rotation_3dgs = features_3dgs[:, feature_idx:feature_idx + 4]  # [N, 4] (Quaternion)
    rotation_neighbors = rotation_3dgs[indices]  # [M, k_neighbors, 4]
    rotation_augmented = np.zeros((len(points_pointcept), 3), dtype=np.float32)  # [M, 3] 초기화
    for j in range(len(points_pointcept)):
        rotation_augmented[j] = get_weighted_direction(
            neighbor_quaternions=rotation_neighbors[j], 
            weights=weights[j]
        )
    selected_features_pointcept.append(rotation_augmented)
    
    # 선택된 features 연결
    if selected_features_pointcept:
        augmented_features = np.concatenate(selected_features_pointcept, axis=-1)  # [M, D]
    else:
        augmented_features = np.zeros((len(points_pointcept), 0), dtype=np.float32)

    return augmented_features

def select_3dgs_features(features_pointcept, features_3dgs, use_features=('scale',)):

    # 사용할 features 선택 (Pointcept에 전달용)
    selected_features_pointcept = []
    selected_features_3dgs = []
    feature_idx = 0

    if 'scale' in use_features:
        selected_features_3dgs.append(features_3dgs[:, feature_idx:feature_idx + 3])
        selected_features_pointcept.append(features_pointcept[:, feature_idx:feature_idx + 3] )
        feature_idx += 3

    if 'opacity' in use_features:
        selected_features_3dgs.append(features_3dgs[:, feature_idx:feature_idx + 1])
        selected_features_pointcept.append(features_pointcept[:, feature_idx:feature_idx + 1])
        feature_idx += 1

    if 'rotation' in use_features:
        selected_features_3dgs.append(features_3dgs[:, feature_idx:feature_idx + 3])
        selected_features_pointcept.append(features_pointcept[:, feature_idx:feature_idx + 3])

    # 비어있을 경우 기본 배열 반환
    if not selected_features_pointcept:
        filtered_pointcept_features = np.zeros((features_pointcept.shape[0], 0), dtype=np.float32)
    else:
        filtered_pointcept_features = np.concatenate(selected_features_pointcept, axis=-1)  # [M, D]

    if not selected_features_3dgs:
        filtered_features_3dgs = np.zeros((features_3dgs.shape[0], 0), dtype=np.float32)
    else:
        filtered_features_3dgs = np.concatenate(selected_features_3dgs, axis=-1)  # [N, D]

    return filtered_pointcept_features, filtered_features_3dgs


def remove_duplicates(points_3dgs, features_3dgs, points_pointcept):
    # KNN을 사용하여 Pointcept 점과 3DGS 점 간의 거리 계산
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points_pointcept)
    distances, _ = nbrs.kneighbors(points_3dgs)

    # 거리가 0.001보다 작은 경우 중복으로 간주
    mask = distances.flatten() > 0.001
    points_3dgs = points_3dgs[mask]
    features_3dgs = features_3dgs[mask]

    return points_3dgs, features_3dgs

def pdistance_pruning(points_3dgs, features_3dgs, points_pointcept, prune_params):
    
    print(f"Before pdistance pruning, 3DGS points: {len(points_3dgs)}")
    
    mask = np.ones(len(points_3dgs), dtype=bool)
    
    # 1. Pointcept Distance Pruning (최우선 적용)
    pcd_pointcept = o3d.geometry.PointCloud()
    pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
    tree = o3d.geometry.KDTreeFlann(pcd_pointcept)
    distances = np.array([tree.search_knn_vector_3d(point, 1)[2][0] for point in points_3dgs])
    distance_mask = distances <= prune_params['pointcept_max_distance']
    mask = mask & distance_mask
    print(f"Pointcept Distance Pruning: Before {len(points_3dgs)} points, Pruned {np.sum(~distance_mask)} points with distance > {prune_params['pointcept_max_distance']:.5f}, After {np.sum(mask)} points")
    
    # Pointcept Distance Pruning 후 점 업데이트
    points_3dgs = points_3dgs[mask]
    features_3dgs = features_3dgs[mask]

    return points_3dgs, features_3dgs


def pruning_all_3dgs_attr(points_3dgs, features_3dgs, prune_methods, prune_params, colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs):
    print(f"Before 3DGS-attr pruning, 3DGS points: {len(points_3dgs)}")
    mask = np.ones(len(points_3dgs), dtype=bool)  # 최종 마스크 초기화
    prev_pruned = 0  # 이전 단계까지 Pruned된 점 수 초기화

    # 1. Scale-based Pruning
    if prune_methods.get('scale', False):
        scales = features_3dgs[:, 0:3]  # 전처리된 scale 값 사용
        if prune_methods.get('scale_ratio', 0.0) > 0:
            scale_magnitudes = np.linalg.norm(scales, axis=-1)  # 전체 점에 대해 계산
            threshold = np.percentile(scale_magnitudes, 100 * (1 - prune_methods['scale_ratio']))
            scale_ratio_mask = scale_magnitudes <= threshold  # 하위 비율 제거
            mask = mask & scale_ratio_mask
            curr_pruned = np.sum(~mask)  # 현재까지 Pruned된 점 수
            print(f"Scale Ratio Pruning: Pruned {curr_pruned} points with scale > {threshold:.4f}, Remaining {np.sum(mask)} points")
            prev_pruned = curr_pruned  # 이전 Pruned 점 수 업데이트

    # 2. Opacity-based Pruning
    if prune_methods.get('opacity', False):
        opacities = features_3dgs[:, 3]  # 전처리된 opacity 값 사용
        if prune_methods.get('opacity_ratio', 0.0) > 0:
            threshold = np.percentile(opacities, 100 * prune_methods['opacity_ratio'])
            opacity_ratio_mask = opacities >= threshold  # 하위 비율 제거
            mask = mask & opacity_ratio_mask
            curr_pruned = np.sum(~mask)  # 현재까지 Pruned된 점 수
            additional_pruned = curr_pruned - prev_pruned  # 추가로 Pruned된 점 수
            print(f"Opacity Ratio Pruning: Pruned {additional_pruned} points with opacity < {threshold:.4f}, Remaining {np.sum(mask)} points")
            prev_pruned = curr_pruned  # 이전 Pruned 점 수 업데이트

    # 3. Rotation Consistency-based Pruning
    if prune_methods.get('rotation', False):
        # Rotation 추출
        rotation_3dgs = features_3dgs[:, -3:]  # [N, 3], [0, 1] 범위

        # KNN으로 Consistency Score 계산
        k_neighbors = prune_params.get('k_neighbors', 5)  # 기본값 5
        knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(points_3dgs)
        distances, indices = knn.kneighbors(points_3dgs)

        consistency_scores = np.zeros(points_3dgs.shape[0])
        for i in range(points_3dgs.shape[0]):
            neighbor_idx = indices[i, 1:]  # 자기 자신 제외
            rot_i = rotation_3dgs[i]
            rot_neighbors = rotation_3dgs[neighbor_idx]
            cos_sim = np.sum(rot_i * rot_neighbors, axis=1) / (
                np.linalg.norm(rot_i) * np.linalg.norm(rot_neighbors, axis=1) + 1e-8
            )
            consistency_scores[i] = np.mean(cos_sim)

        # Rotation Consistency 기반 Pruning
        if prune_methods.get('rotation_ratio', 0.0) > 0:
            threshold = np.percentile(consistency_scores, 100 * prune_methods['rotation_ratio'])
            rotation_ratio_mask = consistency_scores >= threshold  # 하위 비율 제거
            mask = mask & rotation_ratio_mask
            curr_pruned = np.sum(~mask)  # 현재까지 Pruned된 점 수
            additional_pruned = curr_pruned - prev_pruned  # 추가로 Pruned된 점 수
            print(f"Rotation Consistency Pruning: Pruned {additional_pruned} points with consistency < {threshold:.4f}, Remaining {np.sum(mask)} points")
            prev_pruned = curr_pruned  # 이전 Pruned 점 수 업데이트

    # 최종 점 업데이트
    points_3dgs = points_3dgs[mask]
    colors_3dgs = colors_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    labels_3dgs = labels_3dgs[mask]
    labels200_3dgs = labels200_3dgs[mask]
    instances_3dgs = instances_3dgs[mask]
    features_3dgs = features_3dgs[mask]  # features_3dgs도 업데이트

    print(f"Final 3DGS points after pruning: {len(points_3dgs)} (Pruned {len(mask) - np.sum(mask)} points in total)")
    return points_3dgs, features_3dgs, colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs

def pruning_3dgs_attr(points_3dgs, features_3dgs, prune_methods, prune_params):
    print(f"Before 3DGS-attr pruning, 3DGS points: {len(points_3dgs)}")
    mask = np.ones(len(points_3dgs), dtype=bool)  # 최종 마스크 초기화
    prev_pruned = 0  # 이전 단계까지 Pruned된 점 수 초기화

    # 1. Scale-based Pruning
    if prune_methods.get('scale', False):
        scales = features_3dgs[:, 0:3]  # 전처리된 scale 값 사용
        if prune_methods.get('scale_ratio', 0.0) > 0:
            scale_magnitudes = np.linalg.norm(scales, axis=-1)  # 전체 점에 대해 계산
            threshold = np.percentile(scale_magnitudes, 100 * (1 - prune_methods['scale_ratio']))
            scale_ratio_mask = scale_magnitudes <= threshold  # 하위 비율 제거
            mask = mask & scale_ratio_mask
            curr_pruned = np.sum(~mask)  # 현재까지 Pruned된 점 수
            print(f"Scale Ratio Pruning: Pruned {curr_pruned} points with scale > {threshold:.4f}, Remaining {np.sum(mask)} points")
            prev_pruned = curr_pruned  # 이전 Pruned 점 수 업데이트

    # 2. Opacity-based Pruning
    if prune_methods.get('opacity', False):
        opacities = features_3dgs[:, 3]  # 전처리된 opacity 값 사용
        if prune_methods.get('opacity_ratio', 0.0) > 0:
            threshold = np.percentile(opacities, 100 * prune_methods['opacity_ratio'])
            opacity_ratio_mask = opacities >= threshold  # 하위 비율 제거
            mask = mask & opacity_ratio_mask
            curr_pruned = np.sum(~mask)  # 현재까지 Pruned된 점 수
            additional_pruned = curr_pruned - prev_pruned  # 추가로 Pruned된 점 수
            print(f"Opacity Ratio Pruning: Pruned {additional_pruned} points with opacity < {threshold:.4f}, Remaining {np.sum(mask)} points")
            prev_pruned = curr_pruned  # 이전 Pruned 점 수 업데이트

    # 3. Rotation Consistency-based Pruning
    if prune_methods.get('rotation', False):
        # Rotation 추출
        rotation_3dgs = features_3dgs[:, -3:]  # [N, 3], [0, 1] 범위

        # KNN으로 Consistency Score 계산
        k_neighbors = prune_params.get('k_neighbors', 5)  # 기본값 5
        knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(points_3dgs)
        distances, indices = knn.kneighbors(points_3dgs)

        consistency_scores = np.zeros(points_3dgs.shape[0])
        for i in range(points_3dgs.shape[0]):
            neighbor_idx = indices[i, 1:]  # 자기 자신 제외
            rot_i = rotation_3dgs[i]
            rot_neighbors = rotation_3dgs[neighbor_idx]
            cos_sim = np.sum(rot_i * rot_neighbors, axis=1) / (
                np.linalg.norm(rot_i) * np.linalg.norm(rot_neighbors, axis=1) + 1e-8
            )
            consistency_scores[i] = np.mean(cos_sim)

        # Rotation Consistency 기반 Pruning
        if prune_methods.get('rotation_ratio', 0.0) > 0:
            threshold = np.percentile(consistency_scores, 100 * prune_methods['rotation_ratio'])
            rotation_ratio_mask = consistency_scores >= threshold  # 하위 비율 제거
            mask = mask & rotation_ratio_mask
            curr_pruned = np.sum(~mask)  # 현재까지 Pruned된 점 수
            additional_pruned = curr_pruned - prev_pruned  # 추가로 Pruned된 점 수
            print(f"Rotation Consistency Pruning: Pruned {additional_pruned} points with consistency < {threshold:.4f}, Remaining {np.sum(mask)} points")
            prev_pruned = curr_pruned  # 이전 Pruned 점 수 업데이트

    # 최종 점 업데이트
    points_3dgs = points_3dgs[mask]
    features_3dgs = features_3dgs[mask]  # features_3dgs도 업데이트

    print(f"Final 3DGS points after pruning: {len(points_3dgs)} (Pruned {len(mask) - np.sum(mask)} points in total)")
    return points_3dgs, features_3dgs