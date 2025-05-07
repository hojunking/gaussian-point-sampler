# fusion_utils.py
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def preprocess_3dgs_attributes(raw_features_3dgs, normalize_scale=True):
    """
    3DGS 속성을 전처리하여 사용 가능한 형태로 변환.
    
    Args:
        vertex_data_3dgs: 3DGS PLY 파일의 vertex 데이터.
        normalize_scale (bool): Scale 값을 [0, 1]로 정규화할지 여부.
    
    Returns:
        np.ndarray: 전처리된 3DGS 속성 [N, 7] (scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z).
    """

    # 속성 분리
    scale = raw_features_3dgs[:, 0:3]  # [N, 3] (scale_x, scale_y, scale_z)
    opacity = raw_features_3dgs[:, 3:4]  # [N, 1] (opacity)
    rotation = raw_features_3dgs[:, 4:8]  # [N, 4] (rot_w, rot_x, rot_y, rot_z)

    # Scale: 로그 변환 해제
    scale_processed = np.exp(scale)  # log(scale) -> scale
    scale_processed = np.nan_to_num(scale_processed, nan=1e-6, posinf=1e-6, neginf=1e-6)
    scale_processed = np.maximum(scale_processed, 1e-6)  # 양수 보장

    # 추가 로그 변환으로 긴 꼬리 완화
    scale_processed = np.log1p(scale_processed)
    if normalize_scale:
        scale_min = scale_processed.min()
        scale_max = scale_processed.max()
        if scale_max > scale_min:  # 분모가 0이 되지 않도록
            scale_processed = (scale_processed - scale_min) / (scale_max - scale_min)
        else:
            scale_processed = np.zeros_like(scale_processed)  # 모든 값이 동일하면 0으로 설정

    # Opacity: 시그모이드 변환
    opacity_processed = 1 / (1 + np.exp(-opacity))  # [-∞, ∞] -> [0, 1]
    opacity_processed = np.nan_to_num(opacity_processed, nan=0.0, posinf=1.0, neginf=0.0)
    opacity_processed = np.clip(opacity_processed, 0.0, 1.0)  # [0, 1]로 클리핑

    # 추가 로그 변환으로 긴 꼬리 완화
    opacity_processed = np.log1p(opacity_processed)
    opacity_min = opacity_processed.min()
    opacity_max = opacity_processed.max()
    if opacity_max > opacity_min:
        opacity_processed = (opacity_processed - opacity_min) / (opacity_max - opacity_min)
    else:
        opacity_processed = np.zeros_like(opacity_processed)

    # 후처리된 속성 결합
    features_3dgs = np.hstack((scale_processed, opacity_processed, rotation))  # [N, 8]
    return features_3dgs

def augment_pointcept_with_3dgs_attributes(points_pointcept, points_3dgs, features_3dgs, k_neighbors=5, use_features=('scale',)):
    # KNN으로 이웃 찾기
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points_3dgs)
    distances, indices = nbrs.kneighbors(points_pointcept)

    # 사용할 features 선택 (Pointcept에 전달용)
    selected_features_pointcept = []
    selected_features_3dgs = []
    feature_idx = 0

    if 'scale' in use_features:
        # Scale 속성 (Pointcept에 전달)
        scale_3dgs = features_3dgs[:, feature_idx:feature_idx + 3]  # [N, 3]
        scale_neighbors = scale_3dgs[indices]  # [M, k_neighbors, 3]
        scale_augmented = np.mean(scale_neighbors, axis=1)  # [M, 3]
        selected_features_pointcept.append(scale_augmented)
        # Scale 속성 (3DGS 필터링)
        selected_features_3dgs.append(scale_3dgs)
        feature_idx += 3

    if 'opacity' in use_features:
        # Opacity 속성 (Pointcept에 전달)
        opacity_3dgs = features_3dgs[:, feature_idx:feature_idx + 1]  # [N, 1]
        opacity_neighbors = opacity_3dgs[indices]  # [M, k_neighbors, 1]
        opacity_augmented = np.mean(opacity_neighbors, axis=1)  # [M, 1]
        selected_features_pointcept.append(opacity_augmented)
        # Opacity 속성 (3DGS 필터링)
        selected_features_3dgs.append(opacity_3dgs)
        feature_idx += 1

    if 'rotation' in use_features:
        # Rotation 속성 (Pointcept에 전달)
        rotation_3dgs = features_3dgs[:, feature_idx:feature_idx + 3]  # [N, 3]
        rotation_neighbors = rotation_3dgs[indices]  # [M, k_neighbors, 3]
        rotation_augmented = np.mean(rotation_neighbors, axis=1)  # [M, 3]
        # 방향 벡터 정규화
        norm = np.linalg.norm(rotation_augmented, axis=1, keepdims=True)
        rotation_augmented = np.divide(rotation_augmented, norm, where=norm != 0, out=np.zeros_like(rotation_augmented))
        selected_features_pointcept.append(rotation_augmented)
        # Rotation 속성 (3DGS 필터링)
        selected_features_3dgs.append(rotation_3dgs)

    # 선택된 features 연결
    if selected_features_pointcept:
        augmented_features = np.concatenate(selected_features_pointcept, axis=-1)  # [M, D]
        filtered_features_3dgs = np.concatenate(selected_features_3dgs, axis=-1)  # [N, D]
    else:
        augmented_features = np.zeros((len(points_pointcept), 0), dtype=np.float32)
        filtered_features_3dgs = np.zeros((len(points_3dgs), 0), dtype=np.float32)

    return augmented_features, filtered_features_3dgs


def remove_duplicates(points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, features_3dgs):
    """
    3DGS 점과 Pointcept 점 간의 중복 점 제거.
    
    Args:
        points_3dgs (np.ndarray): 3DGS 점 좌표.
        points_pointcept (np.ndarray): Pointcept 점 좌표.
        normals_3dgs (np.ndarray): 3DGS 점 법선.
        labels_3dgs (np.ndarray): 3DGS 점 segment20 라벨.
        labels200_3dgs (np.ndarray): 3DGS 점 segment200 라벨.
        instances_3dgs (np.ndarray): 3DGS 점 instance ID.
        colors_3dgs (np.ndarray): 3DGS 점 색상.
        return_indices (bool): 필터링된 인덱스를 반환할지 여부 (기본값: False).
    
    Returns:
        tuple: (points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, [indices])
               indices는 return_indices=True일 때만 반환됨.
    """
    # KNN을 사용하여 Pointcept 점과 3DGS 점 간의 거리 계산
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points_pointcept)
    distances, _ = nbrs.kneighbors(points_3dgs)

    # 거리가 0.001보다 작은 경우 중복으로 간주
    mask = distances.flatten() > 0.001
    points_3dgs = points_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    labels_3dgs = labels_3dgs[mask]
    labels200_3dgs = labels200_3dgs[mask]
    instances_3dgs = instances_3dgs[mask]
    colors_3dgs = colors_3dgs[mask]
    features_3dgs = features_3dgs[mask]
    return points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs, features_3dgs

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

# def pruning_3dgs_attr(points_3dgs, filtered_features_3dgs, features_3dgs, prune_methods, prune_params):
#     scores = {}

#     # 1. Scale-based Score
#     if prune_params.get('w_scale', 0.0) > 0 and features_3dgs is not None:
#         scales = features_3dgs[:, 0:3]
#         scale_magnitudes = np.linalg.norm(scales, axis=-1)
#         scores['scale'] = 1 - (scale_magnitudes - scale_magnitudes.min()) / (scale_magnitudes.max() - scale_magnitudes.min() + 1e-8)

#     # 2. Opacity-based Score
#     if prune_params.get('w_opacity', 0.0) > 0 and features_3dgs is not None:
#         beta = 8.0  # 스케일링 팩터
#         scores['opacity'] = beta * features_3dgs[:, 3]  # 이미 정규화된 값 스케일링
#         #scores['opacity'] = np.clip(scores['opacity'], 0, 1)  # 범위 보장
    
#     # 3. Rotation Consistency-based Score
#     if prune_params.get('w_rotation', 0.0) > 0 and features_3dgs is not None:
#         rotation_3dgs = features_3dgs[:, -3:]
#         rotation_3dgs = rotation_3dgs * 2 - 1

#         k_neighbors = prune_params.get('k_neighbors', 5)
#         knn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(points_3dgs)
#         distances, indices = knn.kneighbors(points_3dgs)

#         consistency_scores = np.zeros(points_3dgs.shape[0])
#         for i in range(points_3dgs.shape[0]):
#             neighbor_idx = indices[i, 1:]
#             rot_i = rotation_3dgs[i]
#             rot_neighbors = rotation_3dgs[neighbor_idx]
#             cos_sim = np.sum(rot_i * rot_neighbors, axis=1) / (
#                 np.linalg.norm(rot_i) * np.linalg.norm(rot_neighbors, axis=1) + 1e-8
#             )
#             consistency_scores[i] = np.mean(cos_sim)

#         scores['rotation'] = (consistency_scores - consistency_scores.min()) / (consistency_scores.max() - consistency_scores.min() + 1e-8)

#     return scores

# def pruning_3dgs_attr_hybrid(points_3dgs, filtered_features_3dgs, features_3dgs, prune_methods, prune_params):
#     print(f"Before 3DGS-attr hybrid pruning, 3DGS points: {len(points_3dgs)}")
    
#     scores = pruning_3dgs_attr(points_3dgs, filtered_features_3dgs, features_3dgs, prune_methods, prune_params)

#     # 분포 정규화
#     for method in scores:
#         score = scores[method]
#         mean = np.mean(score)
#         std = np.std(score) + 1e-8
#         scores[method] = (score - mean) / std
#         scores[method] = (scores[method] - scores[method].min()) / (scores[method].max() - scores[method].min() + 1e-8)
    
#     weights = {
#         'scale': prune_params.get('w_scale', 0.0),
#         'opacity': prune_params.get('w_opacity', 0.0),
#         'rotation': prune_params.get('w_rotation', 0.0)
#     }

#     total_weight = sum(weights.values())
#     if total_weight > 0:
#         for key in weights:
#             weights[key] /= total_weight
#     else:
#         raise ValueError("At least one pruning method must be enabled with non-zero weight.")

#     importance_scores = np.zeros(len(points_3dgs))
#     for method, score in scores.items():
#         importance_scores += weights[method] * score

#     pruning_ratio = prune_params.get('pruning_ratio', 0.0)
#     if pruning_ratio > 0:
#         threshold = np.percentile(importance_scores, 100 * pruning_ratio)
#         mask = importance_scores >= threshold
#         print(f"Hybrid Pruning: Pruned {np.sum(~mask)} points with importance score < {threshold:.4f}, Remaining {np.sum(mask)} points")
        
#         pruned_indices = np.where(~mask)[0]
#         if len(pruned_indices) > 0:
#             contrib = {}
#             for method in weights:
#                 if method in scores:
#                     contrib[method] = weights[method] * scores[method][pruned_indices] / (importance_scores[pruned_indices] + 1e-8)
#                 else:
#                     contrib[method] = np.zeros(len(pruned_indices))
            
#             avg_contrib = {method: np.mean(contrib[method]) for method in contrib}
#             total_contrib = sum(avg_contrib.values())
#             if total_contrib > 0:
#                 contrib_percent = {method: (avg_contrib[method] / total_contrib) * 100 for method in avg_contrib}
#             else:
#                 contrib_percent = {method: 0.0 for method in avg_contrib}
            
#             print(f"Pruning Contribution - Scale: {contrib_percent['scale']:.1f}%, Opacity: {contrib_percent['opacity']:.1f}%, Rotation: {contrib_percent['rotation']:.1f}%")
#         else:
#             print("No points were pruned, skipping contribution analysis.")
#     else:
#         mask = np.ones(len(points_3dgs), dtype=bool)
#         print("No final pruning applied (pruning_ratio = 0).")

#     points_3dgs = points_3dgs[mask]
#     filtered_features_3dgs = filtered_features_3dgs[mask]
#     features_3dgs = features_3dgs[mask]

#     print(f"Final 3DGS points after hybrid pruning: {len(points_3dgs)} (Pruned {len(mask) - np.sum(mask)} points in total)")
#     return points_3dgs, filtered_features_3dgs

def pruning_3dgs_attr(points_3dgs, filtered_features_3dgs, features_3dgs, prune_methods, prune_params):
    print(f"Before 3DGS-attr pruning, 3DGS points: {len(points_3dgs)}")
    mask = np.ones(len(points_3dgs), dtype=bool)  # 최종 마스크 초기화
    prev_pruned = 0  # 이전 단계까지 Pruned된 점 수 초기화

    # 1. Scale-based Pruning
    if prune_methods.get('scale', False) and features_3dgs is not None:
        scales = features_3dgs[:, 0:3]  # 전처리된 scale 값 사용
        if prune_methods.get('scale_ratio', 0.0) > 0:
            scale_magnitudes = np.linalg.norm(scales, axis=-1)  # 전체 점에 대해 계산
            threshold = np.percentile(scale_magnitudes, 100 * (1 - prune_methods['scale_ratio']))
            scale_ratio_mask = scale_magnitudes <= threshold  # 상위 비율 제거
            mask = mask & scale_ratio_mask
            curr_pruned = np.sum(~mask)  # 현재까지 Pruned된 점 수
            print(f"Scale Ratio Pruning: Pruned {curr_pruned} points with scale > {threshold:.4f}, Remaining {np.sum(mask)} points")
            prev_pruned = curr_pruned  # 이전 Pruned 점 수 업데이트

    # 2. Opacity-based Pruning
    if prune_methods.get('opacity', False) and features_3dgs is not None:
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
    if prune_methods.get('rotation', False) and features_3dgs is not None:
        # Rotation 추출
        rotation_3dgs = features_3dgs[:, -3:]  # [N, 3], [0, 1] 범위
        rotation_3dgs = rotation_3dgs * 2 - 1  # [-1, 1]로 복원

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
    filtered_features_3dgs = filtered_features_3dgs[mask]
    features_3dgs = features_3dgs[mask]  # features_3dgs도 업데이트

    print(f"Final 3DGS points after pruning: {len(points_3dgs)} (Pruned {len(mask) - np.sum(mask)} points in total)")
    return points_3dgs, filtered_features_3dgs