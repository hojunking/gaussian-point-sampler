import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

# Pruning 함수들
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prune_3dgs_by_pointcept_distance(points_3dgs, normals_3dgs, vertex_data_3dgs, points_pointcept, max_distance=0.2):
    """
    Pointcept 점과의 거리 기반 Pruning (너무 멀리 떨어진 점 제거).
    
    Args:
        points_3dgs (np.ndarray): 3DGS 점 좌표 (N, 3).
        normals_3dgs (np.ndarray): 3DGS 점 법선 (N, 3).
        vertex_data_3dgs (PlyData): 3DGS 점의 원본 데이터.
        points_pointcept (np.ndarray): Pointcept 점 좌표 (M, 3).
        max_distance (float): 최대 허용 거리.
    
    Returns:
        tuple: Pruning된 (points_3dgs, normals_3dgs, vertex_data_3dgs).
    """
    num_points_before = len(points_3dgs)
    pcd_pointcept = o3d.geometry.PointCloud()
    pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
    tree = o3d.geometry.KDTreeFlann(pcd_pointcept)
    
    mask = np.ones(len(points_3dgs), dtype=bool)
    for i, point in enumerate(points_3dgs):
        [k, idx, dist] = tree.search_knn_vector_3d(point, 1)
        if k > 0 and dist[0] > max_distance ** 2:  # dist는 제곱 거리
            mask[i] = False
    
    points_3dgs = points_3dgs[mask]
    normals_3dgs = normals_3dgs[mask]
    vertex_data_3dgs = vertex_data_3dgs[mask]
    
    num_points_pruned = num_points_before - len(points_3dgs)
    print(f"Pointcept Distance Pruning: Before {num_points_before} points, Pruned {num_points_pruned} points with distance > {max_distance}, After {len(points_3dgs)} points")
    return points_3dgs, normals_3dgs, vertex_data_3dgs

def prune_3dgs_by_sor(points_3dgs, normals_3dgs, sor_nb_neighbors=20, sor_std_ratio=2.0):
    """
    Statistical Outlier Removal (SOR)을 사용하여 3DGS 점의 이상치를 제거.
    
    Args:
        points_3dgs: 3DGS 점의 좌표 (N, 3).
        normals_3dgs: 3DGS 점의 법선 (N, 3).
        sor_nb_neighbors (int): 이웃 점 수.
        sor_std_ratio (float): 표준편차 비율.
    
    Returns:
        points_3dgs: pruning된 좌표.
        normals_3dgs: pruning된 법선.
        mask: 제거된 점의 인덱스.
    """
    num_points_before = len(points_3dgs)
    pcd_3dgs = o3d.geometry.PointCloud()
    pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
    pcd_3dgs.normals = o3d.utility.Vector3dVector(normals_3dgs)
    pcd_3dgs, ind = pcd_3dgs.remove_statistical_outlier(nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio)
    num_points_pruned = num_points_before - len(ind)
    print(f"SOR Pruning: Before {num_points_before} points, Pruned {num_points_pruned} points, After {len(ind)} points")
    
    mask = np.ones(num_points_before, dtype=bool)
    mask[ind] = False  # 제거된 점: True, 유지된 점: False
    return np.asarray(pcd_3dgs.points), np.asarray(pcd_3dgs.normals), mask


def prune_by_pointcept_distance(points_3dgs, points_pointcept, pdistance_max=0.00008):
    """
    Prune 3DGS points based on distance to the nearest Pointcept point.
    
    Args:
        points_3dgs (np.ndarray): 3DGS points [N, 3].
        points_pointcept (np.ndarray): Pointcept points [M, 3].
        pdistance_max (float): Maximum allowed distance to the nearest Pointcept point (squared distance).
    
    Returns:
        np.ndarray: Boolean mask indicating which 3DGS points to keep.
    """
    num_points_before = len(points_3dgs)
    pcd_pointcept = o3d.geometry.PointCloud()
    pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
    tree = o3d.geometry.KDTreeFlann(pcd_pointcept)
    
    # 각 3DGS 포인트에서 가장 가까운 Pointcept 포인트까지의 제곱 거리 계산
    distances = np.array([tree.search_knn_vector_3d(point, 1)[2][0] for point in points_3dgs])
    
    # 제곱 거리 기준으로 Pruning
    mask = distances <= pdistance_max
    
    num_points_pruned = num_points_before - np.sum(mask)
    print(f"Pointcept Distance Pruning: Before {num_points_before} points, Pruned {num_points_pruned} points with squared distance > {pdistance_max:.5f}, After {np.sum(mask)} points")
    return mask
def prune_3dgs(vertex_data_3dgs, points_3dgs, normals_3dgs, points_pointcept, normals_pointcept, prune_methods, prune_params):
    """
    3DGS 점에 대해 다양한 pruning 방법을 적용.
    
    Args:
        vertex_data_3dgs: 3DGS PLY 파일의 vertex 데이터.
        points_3dgs (np.ndarray): 3DGS 점 좌표 (N, 3).
        normals_3dgs (np.ndarray): 3DGS 점 법선 (N, 3).
        points_pointcept (np.ndarray): Pointcept 점 좌표 (M, 3).
        normals_pointcept (np.ndarray): Pointcept 점 법선 (M, 3).
        prune_methods (dict): 적용할 pruning 방법 및 ratio.
        prune_params (dict): pruning 하이퍼파라미터.
    
    Returns:
        tuple: Pruning된 (points_3dgs, normals_3dgs, vertex_data_3dgs).
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
    vertex_data_3dgs = vertex_data_3dgs[mask] if vertex_data_3dgs is not None else None
    
    # 점이 0개일 경우 조기 종료
    if len(points_3dgs) == 0:
        print("No 3DGS points remaining after Pointcept Distance Pruning. Skipping further pruning.")
        return points_3dgs, normals_3dgs, vertex_data_3dgs
    
    mask = np.ones(len(points_3dgs), dtype=bool)  # 마스크 초기화
    
    # 2. Scale-based Pruning
    if prune_methods.get('scale', False) and vertex_data_3dgs is not None:
        scales = np.stack([vertex_data_3dgs[f'scale_{i}'] for i in range(3)], axis=-1)
        scale_magnitudes = np.linalg.norm(scales, axis=-1)
        threshold = np.percentile(scale_magnitudes, 100 * (1 - prune_methods['scale_ratio']))
        scale_mask = scale_magnitudes <= threshold
        mask = mask & scale_mask
        print(f"Scale Pruning: Pruned {np.sum(~scale_mask)} points with scale > {threshold:.4f}, Remaining {np.sum(mask)} points")
    
    # 3. Opacity-based Pruning
    if prune_methods.get('opacity', False) and vertex_data_3dgs is not None:
        opacities = sigmoid(vertex_data_3dgs['opacity'])
        threshold = np.percentile(opacities, 100 * prune_methods['opacity_ratio'])
        opacity_mask = opacities >= threshold
        mask = mask & opacity_mask
        print(f"Opacity Pruning: Pruned {np.sum(~opacity_mask)} points with opacity < {threshold:.4f}, Remaining {np.sum(mask)} points")
    
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
        # 병합된 Point Cloud 생성
        pcd_merged = o3d.geometry.PointCloud()
        pcd_merged.points = o3d.utility.Vector3dVector(np.vstack((points_pointcept, points_3dgs)))
        pcd_merged.normals = o3d.utility.Vector3dVector(np.vstack((normals_pointcept, normals_3dgs)))
        
        # 병합된 Point Cloud에서 법선 추정
        pcd_merged.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(prune_params['normal_k_neighbors'])
        )
        pcd_merged.orient_normals_consistent_tangent_plane(k=prune_params['normal_k_neighbors'])
        normals_estimated = np.asarray(pcd_merged.normals)
        # 3DGS 점에 해당하는 법선 추출
        normals_estimated_3dgs = normals_estimated[len(points_pointcept):]
        
        # 추정된 법선 벡터 크기 확인
        norm_estimated = np.linalg.norm(normals_estimated_3dgs, axis=-1)
        print(f"Estimated normals norm: min={norm_estimated.min():.4f}, max={norm_estimated.max():.4f}, zero_count={np.sum(norm_estimated == 0)}")
        
        # 추정된 법선이 모두 0인 경우, 3DGS 점만으로 다시 추정
        if np.all(norm_estimated == 0):
            print("All estimated normals are zero. Estimating normals for 3DGS points only...")
            pcd_3dgs = o3d.geometry.PointCloud()
            pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
            pcd_3dgs.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(prune_params['normal_k_neighbors'])
            )
            pcd_3dgs.orient_normals_consistent_tangent_plane(k=prune_params['normal_k_neighbors'])
            normals_estimated_3dgs = np.asarray(pcd_3dgs.normals)
            norm_estimated = np.linalg.norm(normals_estimated_3dgs, axis=-1)
            print(f"After re-estimation - Estimated normals norm: min={norm_estimated.min():.4f}, max={norm_estimated.max():.4f}, zero_count={np.sum(norm_estimated == 0)}")
        
        # 법선 벡터 크기 확인 및 필터링
        norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
        print(f"3DGS normals norm (in prune_3dgs): min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
        valid_norm_mask = (norm_3dgs > 0) & (norm_estimated > 0)
        print(f"Points with valid norms: {np.sum(valid_norm_mask)} / {len(normals_3dgs)}")
        
        # 코사인 유사도 계산 (방향 무시)
        cos_sim = np.zeros(len(normals_3dgs))
        if np.sum(valid_norm_mask) > 0:
            cos_sim[valid_norm_mask] = np.abs(np.sum(normals_3dgs[valid_norm_mask] * normals_estimated_3dgs[valid_norm_mask], axis=-1) / (
                np.linalg.norm(normals_3dgs[valid_norm_mask], axis=-1) * np.linalg.norm(normals_estimated_3dgs[valid_norm_mask], axis=-1)
            ))
        # 코사인 유사도 분포 출력
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
    vertex_data_3dgs = vertex_data_3dgs[mask] if vertex_data_3dgs is not None else None
    
    print(f"Final 3DGS points after pruning: {len(points_3dgs)} (Pruned {len(mask) - np.sum(mask)} points in total)")
    return points_3dgs, normals_3dgs, vertex_data_3dgs