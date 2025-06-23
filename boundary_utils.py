# boundary_utils.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm

def boundary_labeling_with_3dgs(points: np.ndarray,
                      features: np.ndarray,
                      prune_methods: dict,
                      prune_params: dict) -> np.ndarray:
    N = points.shape[0]
    mask = np.ones(N, dtype=bool)

    # 1) Scale-based pruning
    if prune_methods.get('scale', False):
        ratio = prune_methods.get('scale_ratio', 0.0)
        if ratio > 0:
            scales = features[:, 0:3]  # 각 포인트의 (sx, sy, sz)
            mags   = np.linalg.norm(scales, axis=1)
            thr    = np.percentile(mags, 100 * (1.0 - ratio))
            mask_scale = (mags <= thr)
            print(f"[Scale] threshold={thr:.4f}, pruned={np.sum(~mask_scale)}")
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
