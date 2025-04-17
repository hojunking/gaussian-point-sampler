# fusion_utils.py
import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors

def quaternion_to_direction(quaternion):
    """
    쿼터니언을 방향 벡터(3차원)로 변환.
    
    Args:
        quaternion (np.ndarray): 쿼터니언 배열 [N, 4] (w, x, y, z).
    
    Returns:
        np.ndarray: 방향 벡터 배열 [N, 3].
    """
    # 쿼터니언 정규화
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = np.divide(quaternion, norm, where=norm != 0, out=np.zeros_like(quaternion))
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # 회전 행렬로 변환
    rot_matrix = np.zeros((len(quaternion), 3, 3))
    rot_matrix[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_matrix[:, 0, 1] = 2 * (x * y - w * z)
    rot_matrix[:, 0, 2] = 2 * (x * z + w * y)
    rot_matrix[:, 1, 0] = 2 * (x * y + w * z)
    rot_matrix[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_matrix[:, 1, 2] = 2 * (y * z - w * x)
    rot_matrix[:, 2, 0] = 2 * (x * z - w * y)
    rot_matrix[:, 2, 1] = 2 * (y * z + w * x)
    rot_matrix[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    # 기준 방향 벡터 (1, 0, 0), (0, 1, 0), (0, 0, 1)에 대해 방향 계산
    base_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    directions = np.zeros((len(quaternion), 3))
    for i in range(3):
        direction = np.einsum('ijk,k->ij', rot_matrix, base_vectors[i])
        directions += direction
    directions /= 3  # 평균

    # 방향 벡터 정규화
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.divide(directions, norm, where=norm != 0, out=np.zeros_like(directions))
    return directions

def augment_pointcept_with_3dgs_attributes(points_pointcept, points_3dgs, features_3dgs, k_neighbors=5, use_features=('scale',)):
    """
    Pointcept 점에 3DGS 속성을 전달 (KNN 사용).
    
    Args:
        points_pointcept (np.ndarray): Pointcept 점 좌표 [M, 3].
        points_3dgs (np.ndarray): 3DGS 점 좌표 [N, 3].
        features_3dgs (np.ndarray): 3DGS 속성 [N, 7] (scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z).
        k_neighbors (int): KNN에서 사용할 이웃 점 개수.
        use_features (tuple): 사용할 features ('scale', 'opacity', 'rotation').
    
    Returns:
        np.ndarray: Pointcept 점에 전달된 속성 [M, D].
    """
    # KNN으로 이웃 찾기
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points_3dgs)
    distances, indices = nbrs.kneighbors(points_pointcept)

    # 사용할 features 선택 (aggregation은 mean으로 고정)
    selected_features = []
    feature_idx = 0
    if 'scale' in use_features:
        # Scale 속성
        scale_3dgs = features_3dgs[:, feature_idx:feature_idx + 3]  # [N, 3]
        scale_neighbors = scale_3dgs[indices]  # [M, k_neighbors, 3]
        scale_augmented = np.mean(scale_neighbors, axis=1)  # [M, 3]
        selected_features.append(scale_augmented)
        feature_idx += 3

    if 'opacity' in use_features:
        # Opacity 속성
        opacity_3dgs = features_3dgs[:, feature_idx:feature_idx + 1]  # [N, 1]
        opacity_neighbors = opacity_3dgs[indices]  # [M, k_neighbors, 1]
        opacity_augmented = np.mean(opacity_neighbors, axis=1)  # [M, 1]
        selected_features.append(opacity_augmented)
        feature_idx += 1

    if 'rotation' in use_features:
        # Rotation 속성 (이미 방향 벡터로 변환된 상태)
        rotation_3dgs = features_3dgs[:, feature_idx:feature_idx + 3]  # [N, 3]
        rotation_neighbors = rotation_3dgs[indices]  # [M, k_neighbors, 3]
        rotation_augmented = np.mean(rotation_neighbors, axis=1)  # [M, 3]
        # 방향 벡터 정규화
        norm = np.linalg.norm(rotation_augmented, axis=1, keepdims=True)
        rotation_augmented = np.divide(rotation_augmented, norm, where=norm != 0, out=np.zeros_like(rotation_augmented))
        selected_features.append(rotation_augmented)

    # 선택된 features 연결
    if selected_features:
        augmented_features = np.concatenate(selected_features, axis=-1)  # [M, D]
    else:
        augmented_features = np.zeros((len(points_pointcept), 0), dtype=np.float32)

    return augmented_features

def preprocess_3dgs_attributes(vertex_data_3dgs, normalize_scale=True):
    """
    3DGS 속성을 전처리하여 사용 가능한 형태로 변환.
    
    Args:
        vertex_data_3dgs: 3DGS PLY 파일의 vertex 데이터.
        normalize_scale (bool): Scale 값을 [0, 1]로 정규화할지 여부.
    
    Returns:
        np.ndarray: 전처리된 3DGS 속성 [N, 7] (scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z).
    """
    # 속성 추출
    raw_features_3dgs = np.hstack([
        np.vstack([vertex_data_3dgs[f'scale_{i}'] for i in range(3)]).T,  # [N, 3]
        vertex_data_3dgs['opacity'][:, None],  # [N, 1]
        np.vstack([vertex_data_3dgs[f'rot_{i}'] for i in range(4)]).T  # [N, 4]
    ])

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
    opacity_processed = (opacity_processed - opacity_processed.min()) / (opacity_processed.max() - opacity_processed.min())

    # Rotation: 쿼터니언을 방향 벡터로 변환
    rotation_processed = quaternion_to_direction(rotation)  # [N, 3] (dir_x, dir_y, dir_z)

    # 후처리된 속성 결합
    features_3dgs = np.hstack((scale_processed, opacity_processed, rotation_processed))  # [N, 7]
    return features_3dgs

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

def prune_3dgs(vertex_data_3dgs, points_3dgs, normals_3dgs, features_3dgs, points_pointcept, normals_pointcept, prune_methods, prune_params):
    """
    3DGS 점에 대해 다양한 pruning 방법을 적용.
    
    Args:
        vertex_data_3dgs: 3DGS PLY 파일의 vertex 데이터.
        points_3dgs (np.ndarray): 3DGS 점 좌표 (N, 3).
        normals_3dgs (np.ndarray): 3DGS 점 법선 (N, 3).
        features_3dgs (np.ndarray): 전처리된 3DGS 속성 [N, 7] (scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z).
        points_pointcept (np.ndarray): Pointcept 점 좌표 (M, 3).
        normals_pointcept (np.ndarray): Pointcept 점 법선 (M, 3).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
    
    Returns:
        tuple: Pruning된 (points_3dgs, normals_3dgs, vertex_data_3dgs, features_3dgs).
    """
    print(f"Initial 3DGS points: {len(points_3dgs)}")
    
    # 마스크 초기화
    mask = np.ones(len(points_3dgs), dtype=bool)
    
    # 1. Pointcept Distance Pruning (최우선 적용)
    if prune_methods.get('pointcept_distance', False):
        pcd_pointcept = o3d.geometry.PointCloud()
        pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
        tree = o3d.geometry.KDTreeFlann(pcd_pointcept)
        distances = np.array([tree.search_knn_vector_3d(point, 1)[2][0] for point in points_3dgs])
        distance_mask = distances <= prune_params['pointcept_max_distance']
        mask = mask & distance_mask
        print(f"Pointcept Distance Pruning: Before {len(points_3dgs)} points, Pruned {np.sum(~distance_mask)} points with distance > {prune_params['pointcept_max_distance']:.5f}, After {np.sum(mask)} points")
    
    # Pointcept Distance Pruning 후 점 업데이트
    points_3dgs = points_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    if vertex_data_3dgs is not None:
        if isinstance(vertex_data_3dgs, dict):
            vertex_data_3dgs = {key: value[mask] for key, value in vertex_data_3dgs.items()}
        else:
            print("Warning: vertex_data_3dgs is not a dict, treating as numpy array.")
            vertex_data_3dgs = vertex_data_3dgs[mask]
    features_3dgs = features_3dgs[mask] if features_3dgs is not None else None
    
    # 점이 0개일 경우 조기 종료
    if len(points_3dgs) == 0:
        print("No 3DGS points remaining after Pointcept Distance Pruning. Skipping further pruning.")
        return points_3dgs, normals_3dgs, vertex_data_3dgs, features_3dgs
    
    mask = np.ones(len(points_3dgs), dtype=bool)  # 마스크 초기화
    
    # 2. Scale-based Pruning
    if prune_methods.get('scale', False) and features_3dgs is not None:
        scales = features_3dgs[:, 0:3]  # 전처리된 scale 값 사용
        scale_lower_threshold = prune_params.get('scale_lower_threshold', 0.01)
        scale_upper_threshold = prune_params.get('scale_upper_threshold', 0.9)
        scale_threshold_mask = np.all((scales >= scale_lower_threshold) & (scales <= scale_upper_threshold), axis=1)
        mask = mask & scale_threshold_mask
        print(f"Scale Threshold Pruning: Pruned {np.sum(~scale_threshold_mask)} points with scale outside [{scale_lower_threshold:.4f}, {scale_upper_threshold:.4f}], Remaining {np.sum(mask)} points")

        if prune_methods.get('scale_ratio', 0.0) > 0:
            scale_magnitudes = np.linalg.norm(scales[mask], axis=-1)
            threshold = np.percentile(scale_magnitudes, 100 * (1 - prune_methods['scale_ratio']))
            scale_ratio_mask = scale_magnitudes <= threshold
            temp_mask = np.ones(len(mask), dtype=bool)
            temp_mask[mask] = scale_ratio_mask
            mask = mask & temp_mask
            print(f"Scale Ratio Pruning: Pruned {np.sum(~scale_ratio_mask)} points with scale > {threshold:.4f}, Remaining {np.sum(mask)} points")
    
    # 3. Opacity-based Pruning
    if prune_methods.get('opacity', False) and features_3dgs is not None:
        opacities = features_3dgs[:, 3]  # 전처리된 opacity 값 사용
        opacity_lower_threshold = prune_params.get('opacity_lower_threshold', 0.01)
        opacity_upper_threshold = prune_params.get('opacity_upper_threshold', 0.9)
        opacity_threshold_mask = (opacities >= opacity_lower_threshold) & (opacities <= opacity_upper_threshold)
        mask = mask & opacity_threshold_mask
        print(f"Opacity Threshold Pruning: Pruned {np.sum(~opacity_threshold_mask)} points with opacity outside [{opacity_lower_threshold:.4f}, {opacity_upper_threshold:.4f}], Remaining {np.sum(mask)} points")

        if prune_methods.get('opacity_ratio', 0.0) > 0:
            threshold = np.percentile(opacities[mask], 100 * prune_methods['opacity_ratio'])
            opacity_ratio_mask = opacities[mask] >= threshold
            temp_mask = np.ones(len(mask), dtype=bool)
            temp_mask[mask] = opacity_ratio_mask
            mask = mask & temp_mask
            print(f"Opacity Ratio Pruning: Pruned {np.sum(~opacity_ratio_mask)} points with opacity < {threshold:.4f}, Remaining {np.sum(mask)} points")
    
    # 4. Density-based Pruning
    if prune_methods.get('density', False):
        pcd_3dgs = o3d.geometry.PointCloud()
        pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
        pcd_3dgs, ind = pcd_3dgs.remove_radius_outlier(
            nb_points=prune_params['density_min_points'],
            radius=prune_params['density_eps']
        )
        density_mask = np.zeros(len(points_3dgs), dtype=bool)
        density_mask[ind] = True
        mask = mask & density_mask
        print(f"Density Pruning: Pruned {np.sum(~density_mask)} points with density < {prune_params['density_min_points']} points within {prune_params['density_eps']:.4f}, Remaining {np.sum(mask)} points")
    
    # 5. Normal-based Pruning (Pointcept와 함께 고려)
    if prune_methods.get('normal', False):
        pcd_merged = o3d.geometry.PointCloud()
        pcd_merged.points = o3d.utility.Vector3dVector(np.vstack((points_pointcept, points_3dgs)))
        pcd_merged.normals = o3d.utility.Vector3dVector(np.vstack((normals_pointcept, normals_3dgs)))
        pcd_merged.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(prune_params['normal_k_neighbors']))
        pcd_merged.orient_normals_consistent_tangent_plane(k=prune_params['normal_k_neighbors'])
        normals_estimated = np.asarray(pcd_merged.normals)
        normals_estimated_3dgs = normals_estimated[len(points_pointcept):]
        
        norm_estimated = np.linalg.norm(normals_estimated_3dgs, axis=-1)
        print(f"Estimated normals norm: min={norm_estimated.min():.4f}, max={norm_estimated.max():.4f}, zero_count={np.sum(norm_estimated == 0)}")
        
        if np.all(norm_estimated == 0):
            print("All estimated normals are zero. Estimating normals for 3DGS points only...")
            pcd_3dgs = o3d.geometry.PointCloud()
            pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
            pcd_3dgs.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(prune_params['normal_k_neighbors']))
            pcd_3dgs.orient_normals_consistent_tangent_plane(k=prune_params['normal_k_neighbors'])
            normals_estimated_3dgs = np.asarray(pcd_3dgs.normals)
            norm_estimated = np.linalg.norm(normals_estimated_3dgs, axis=-1)
            print(f"After re-estimation - Estimated normals norm: min={norm_estimated.min():.4f}, max={norm_estimated.max():.4f}, zero_count={np.sum(norm_estimated == 0)}")
        
        norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
        print(f"3DGS normals norm (in prune_3dgs): min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
        valid_norm_mask = (norm_3dgs > 0) & (norm_estimated > 0)
        print(f"Points with valid norms: {np.sum(valid_norm_mask)} / {len(normals_3dgs)}")
        
        cos_sim = np.zeros(len(normals_3dgs))
        if np.sum(valid_norm_mask) > 0:
            cos_sim[valid_norm_mask] = np.abs(np.sum(normals_3dgs[valid_norm_mask] * normals_estimated_3dgs[valid_norm_mask], axis=-1) / (
                np.linalg.norm(normals_3dgs[valid_norm_mask], axis=-1) * np.linalg.norm(normals_estimated_3dgs[valid_norm_mask], axis=-1)
            ))
        print(f"Cosine similarity: min={cos_sim.min():.4f}, max={cos_sim.max():.4f}, mean={cos_sim.mean():.4f}")
        print(f"Points with cos_sim >= {prune_params['normal_cos_threshold']:.4f}: {np.sum(cos_sim >= prune_params['normal_cos_threshold'])}")
        
        normal_mask = cos_sim >= prune_params['normal_cos_threshold']
        mask = mask & normal_mask
        print(f"Normal Pruning (with Pointcept): Before {np.sum(mask)} points, Pruned {np.sum(~normal_mask)} points with cosine similarity < {prune_params['normal_cos_threshold']:.4f}, After {np.sum(mask)} points")
    
    # 6. SOR-based Pruning
    if prune_methods.get('sor', False):
        pcd_3dgs = o3d.geometry.PointCloud()
        pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
        pcd_3dgs, ind = pcd_3dgs.remove_statistical_outlier(
            nb_neighbors=prune_params['sor_nb_neighbors'],
            std_ratio=prune_params['sor_std_ratio']
        )
        sor_mask = np.zeros(len(points_3dgs), dtype=bool)
        sor_mask[ind] = True
        mask = mask & sor_mask
        print(f"SOR Pruning: Pruned {np.sum(~sor_mask)} points, Remaining {np.sum(mask)} points")
    
    # 최종 점 업데이트
    points_3dgs = points_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    if vertex_data_3dgs is not None:
        if isinstance(vertex_data_3dgs, dict):
            vertex_data_3dgs = {key: value[mask] for key, value in vertex_data_3dgs.items()}
        else:
            print("Warning: vertex_data_3dgs is not a dict, treating as numpy array.")
            vertex_data_3dgs = vertex_data_3dgs[mask]
    features_3dgs = features_3dgs[mask] if features_3dgs is not None else None
    
    print(f"Final 3DGS points after pruning: {len(points_3dgs)} (Pruned {len(mask) - np.sum(mask)} points in total)")
    return points_3dgs, normals_3dgs, vertex_data_3dgs, features_3dgs