# boundary_utils.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm

def boundary_labeling_with_3dgs(points: np.ndarray,
                      features: np.ndarray,
                      labels_pointcept,
                      prune_methods: dict,
                      prune_params: dict,
                      ) -> np.ndarray:
    N = points.shape[0]
    mask = np.ones(N, dtype=bool)

    # 1) Scale-based pruning: 단순화된 '클래스별 전역 하위 퍼센타일' 로직으로 대체
    if prune_methods.get('scale', False):
        print("\nStarting class-global pruning based on scale attribute...")
        
        # --- 하이퍼파라미터 ---
        ratio = prune_methods.get('scale_ratio', 0.0)
        excluded_classes = prune_params.get('excluded_classes', [1]) # floor
        min_class_points = prune_params.get('min_class_points', 20)   # 최소 20개 이상의 포인트가 있는 클래스만 분석
        # 전체 포인트의 스케일 크기를 미리 계산
        scales = np.linalg.norm(features[:, 0:3], axis=1)
        # 최종 스케일 마스크 초기화
        mask_scale = np.zeros(N, dtype=bool)
        
        # 씬에 존재하는 모든 유니크한 레이블을 찾음
        unique_labels = np.unique(labels_pointcept)

        # 각 유니크 레이블(클래스 그룹)에 대해 루프 실행
        for class_id in tqdm(unique_labels, desc="Analyzing each class group"):
            # 제외할 클래스인지 확인
            if class_id in excluded_classes:
                print(f"Skipping class {class_id} as it is in the excluded list.")
                continue
            # 현재 클래스에 해당하는 모든 포인트의 '전체 인덱스'를 찾음
            group_indices = np.where(labels_pointcept == class_id)[0]
            # 해당 클래스의 포인트가 너무 적으면 통계적 의미가 없으므로 건너뜀
            if len(group_indices) < min_class_points:
                continue
            # 현재 클래스 그룹의 스케일 값들을 가져옴
            group_scales = scales[group_indices]
            
            # 이 클래스 그룹 전체에서 하위 %에 해당하는 스케일 임계값 계산
            threshold = np.percentile(group_scales, 100 * (1.0 - ratio))
            
            is_edge_in_group = group_scales <= threshold
            # 경계로 판별된 포인트들의 '원래' 전체 인덱스를 가져옴
            edge_original_indices = group_indices[is_edge_in_group]
            
            # 최종 스케일 마스크에 경계 정보를 기록
            mask_scale[edge_original_indices] = True
            
        print(f"Pruning complete. Found {np.sum(mask_scale)} points as edges (bottom {100 * (1.0 - ratio)}% within each object class).")
        mask &= mask_scale


    # 2) Opacity-based pruning
    if prune_methods.get('opacity', False):
        ratio = prune_methods.get('opacity_ratio', 0.0)
        if ratio > 0:
            opacities = features[:, 3]
            thr        = np.percentile(opacities, 100 * ratio)
            mask_op    = (opacities >= thr)
            print(f"[Opacity] threshold={thr:.4f}, pruned={np.sum(~mask_op)}")
            mask &= mask_op

    # 3) Rotation-consistency pruning
    if prune_methods.get('rotation', False):
        ratio = prune_methods.get('rotation_ratio', 0.0)
        if ratio > 0:
            rots = features[:, -3:]  # (rx, ry, rz)
            k    = prune_params.get('k_neighbors', 5)

            # KNN 찾기
            knn = NearestNeighbors(n_neighbors=k+1).fit(points)
            _, idxs = knn.kneighbors(points)

            # 각 포인트별 이웃 간 평균 코사인 유사도 계산
            consistency = np.zeros(N, dtype=np.float32)
            for i in range(N):
                neigh = idxs[i, 1:]  # self 제외
                v0    = rots[i]
                v1    = rots[neigh]
                cosim = np.sum(v0 * v1, axis=1) / (
                            np.linalg.norm(v0) * np.linalg.norm(v1, axis=1) + 1e-8
                        )
                consistency[i] = np.mean(cosim)

            thr = np.percentile(consistency, 100 * ratio)
            mask_rot = (consistency >= thr)
            print(f"[Rotation] threshold={thr:.4f}, pruned={np.sum(~mask_rot)}")
            mask &= mask_rot

    # 최종 레이블: True → 1(boundary), False → 0(non-boundary)
    labels = np.zeros(N, dtype=np.int64)
    labels[mask] = 1

    return labels

def boundary_labeling_with_semantic_label(points, labels, prune_params):
    boundary_radius = prune_params.get('boundary_radius', 0.06) # 기본값 6cm
    
    if points is None or labels is None:
        raise ValueError("Points and labels cannot be None.")
    if len(points) != len(labels):
        raise ValueError("Points and labels must have the same length.")

    print("\nStarting boundary labeling based on semantic labels (BFANet method)...")
    
    # KD-Tree를 사용하여 효율적인 최근접 이웃 검색
    print("Building KD-Tree for efficient neighbor search...")
    tree = KDTree(points)
    
    # 각 점에 대해 반경 내 이웃을 쿼리
    print(f"Querying neighbors within radius {boundary_radius}m...")
    # query_radius는 각 포인트에 대한 이웃 인덱스 리스트의 리스트를 반환합니다.
    indices_list = tree.query_radius(points, r=boundary_radius)
    
    boundary = np.zeros(len(points), dtype=np.int8)
    
    print("Labeling boundary points...")
    for i in tqdm(range(len(points)), desc="Processing points"):
        # 현재 포인트의 레이블
        current_label = labels[i]
        
        # 이웃 포인트들의 인덱스
        neighbor_indices = indices_list[i]
        
        # 이웃이 자기 자신 뿐이라면 경계가 아님
        if len(neighbor_indices) <= 1:
            continue
            
        # 이웃 포인트들의 레이블
        neighbor_labels = labels[neighbor_indices]
        
        # 이웃 레이블 중에 현재 레이블과 다른 것이 하나라도 있으면 경계로 지정
        if np.any(neighbor_labels != current_label):
            boundary[i] = 1
            
    print(f"Boundary labeling complete. Found {np.sum(boundary)} boundary points.")
    return boundary

# 헬퍼 함수: 의미론적 경계점을 찾는 로직
def find_semantic_boundary_points(points, labels, radius):
    """주변에 다른 레이블을 가진 포인트가 있는 '의미론적 경계점'을 찾습니다."""
    tree = KDTree(points)
    indices_list = tree.query_radius(points, r=radius)
    boundary_mask = np.zeros(len(points), dtype=bool)
    # tqdm을 사용하여 진행 상황 표시
    for i in range(len(points)):
        # 여러 레이블이 존재하면 경계로 판단
        if len(np.unique(labels[indices_list[i]])) > 1:
            boundary_mask[i] = True
    return np.where(boundary_mask)[0]

# 새로운 함수 시그니처로 변경
def boundary_labeling_with_semantic_gaussian(
    points: np.ndarray,
    labels_pointcept: np.ndarray,
    points_3dgs: np.ndarray,
    features_3dgs: np.ndarray,
    prune_methods: dict,
    prune_params: dict
) -> np.ndarray:
    """
    의미론적 경계점 주변의 3DGS 가우시안만 필터링하여 모호성을 검사하는 최적화된 방법입니다.
    """
    N_points = points.shape[0]
    
    # --- 하이퍼파라미터 ---
    semantic_radius = prune_params.get('boundary_radius', 0.06)
    gaussian_search_radius = prune_params.get('gaussian_search_radius', 0.02)
    sigma_multiplier = prune_params.get('sigma_multiplier', 1.0)
    scale_ratio = prune_methods.get('scale_ratio', 0.0)


    print("\nStarting OPTIMIZED ambiguity-based boundary detection with separate 3DGS data...")

    # --- 1단계: 원본 포인트 클라우드에서 의미론적 경계점 찾기 ---
    semantic_boundary_indices = find_semantic_boundary_points(points, labels_pointcept, semantic_radius)
    
    if len(semantic_boundary_indices) == 0:
        print("No semantic boundary points found. Returning empty edge map.")
        return np.zeros(N_points, dtype=np.int64)
    print(f"Found {len(semantic_boundary_indices)} semantic boundary points to start the search.")
    
    # --- 2단계: 경계점 주변의 '후보' 가우시안 필터링 ---
    # 실제 3DGS 가우시안 중심점에 대한 KD-Tree 생성
    gaussian_tree = KDTree(points_3dgs)
    
    # 경계점들 주변의 가우시안 인덱스를 찾음 (중복 포함)
    candidate_gaussian_indices_nested = gaussian_tree.query_radius(
        points[semantic_boundary_indices], r=gaussian_search_radius
    )
    # 중첩 리스트를 펼치고, 중복을 제거하여 최종 후보 가우시안 인덱스 집합을 만듦
    candidate_gaussian_indices = np.unique(np.concatenate(candidate_gaussian_indices_nested))
    print(f"Filtered down to {len(candidate_gaussian_indices)} candidate Gaussians for final check.")

    scales_3dgs_magnitude = np.linalg.norm(features_3dgs[:, 0:3], axis=1)
    scale_threshold = np.percentile(scales_3dgs_magnitude, 100 * (1.0 - scale_ratio))
    final_candidate_indices = [idx for idx in candidate_gaussian_indices
        if scales_3dgs_magnitude[idx] <= scale_threshold
    ]
    # --- 3단계: 후보 가우시안의 모호성 검사 및 최종 레이블링 ---
    point_tree = KDTree(points)
    radii_3dgs = np.max(np.abs(features_3dgs[:, 0:3]), axis=1) * sigma_multiplier
    
    final_edge_mask = np.zeros(N_points, dtype=bool)

    # 필터링된 '후보' 가우시안들에 대해서만 모호성 검사
    for g_idx in tqdm(final_candidate_indices, desc="Final check on candidate Gaussians"):
        gaussian_center = points_3dgs[g_idx:g_idx+1]
        gaussian_radius = radii_3dgs[g_idx]
        # 가우시안의 영향권 내에 있는 '원본 포인트'들의 인덱스를 찾음
        point_indices_in_gaussian = point_tree.query_radius(gaussian_center, r=gaussian_radius)[0]
        
        if len(point_indices_in_gaussian) < 2:
            continue
            
        associated_labels = labels_pointcept[point_indices_in_gaussian]
        
        if len(np.unique(associated_labels)) > 1:
            # 이 가우시안은 '모호함'이 확정됨. 관련된 모든 원본 포인트를 경계로 지정.
            final_edge_mask[point_indices_in_gaussian] = True
            
    print(f"Analysis complete. Found {np.sum(final_edge_mask)} total boundary points.")

    labels = np.zeros(N_points, dtype=np.int64)
    labels[final_edge_mask] = 1
    return labels
    
def save_boundary_ply(points_pointcept: np.ndarray,
                      labels_pointcept: np.ndarray,
                      boundary: np.ndarray,
                      output_ply_path: str) -> None:
    SCANNET_20_COLORS = [
    [174, 199, 232],  # 0 - wall
    [152, 223, 138],  # 1 - floor
    [31, 119, 180],   # 2 - cabinet
    [255, 187, 120],  # 3 - bed
    [188, 189, 34],   # 4 - chair
    [140, 86, 75],    # 5 - sofa
    [255, 152, 150],  # 6 - table
    [214, 39, 40],    # 7 - door
    [197, 176, 213],  # 8 - window
    [148, 103, 189],  # 9 - bookshelf
    [196, 156, 148],  # 10 - picture
    [23, 190, 207],   # 11 - counter
    [178, 76, 76],    # 12 - desk
    [247, 182, 210],  # 13 - curtain
    [66, 188, 102],   # 14 - refrigerator
    [219, 219, 141],  # 15 - shower curtain
    [140, 57, 197],   # 16 - toilet
    [202, 185, 52],   # 17 - sink
    [51, 176, 203],   # 18 - bathtub
    [200, 54, 131]    # 19 - other furniture
]
    
    """
    Save the boundary labels to a PLY file.
    """
    """
    Save a colored PLY where:
      - boundary==1 → white
      - boundary==0 → semantic color from labels_pointcept
    """
    N = points_pointcept.shape[0]
    # 컬러 배열 초기화
    colors = np.zeros((N, 3), dtype=np.uint8)
    # boundary 포인트는 흰색
    colors[boundary == 1] = [255, 255, 255]
    
    # 나머지는 semantic color
    # mask = (boundary == 0)
    # if labels_pointcept is not None:
    #     colors[mask] = np.array(SCANNET_20_COLORS, dtype=np.uint8)[labels_pointcept[mask]]

    # PLY element 생성
    vertex = np.empty(N, dtype=[('x','f4'),('y','f4'),('z','f4'),
                                 ('red','u1'),('green','u1'),('blue','u1')])
    vertex['x']   = points_pointcept[:,0]
    vertex['y']   = points_pointcept[:,1]
    vertex['z']   = points_pointcept[:,2]
    vertex['red']   = colors[:,0]
    vertex['green'] = colors[:,1]
    vertex['blue']  = colors[:,2]

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(output_ply_path)
    print(f"Saved boundary-colored PLY to {output_ply_path}")
