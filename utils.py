import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
import os
from sklearn.neighbors import NearestNeighbors, KDTree

def load_pointcept_data(pointcept_dir):
    """
    Pointcept 데이터를 .npy 파일에서 로드.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/train/scene0000_00).
    
    Returns:
        dict: 로드된 데이터 (coord, color, normal, segment20, segment200, instance).
              segment200과 instance는 선택적으로 로드됨.
    """
    data = {}
    # 필수 데이터 로드
    required_keys = ['coord', 'color', 'normal', 'segment20']
    for key in required_keys:
        file_path = os.path.join(pointcept_dir, f"{key}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{key}.npy not found in {pointcept_dir}")
        data[key] = np.load(file_path)

    # 선택적 데이터 로드 (segment200, instance)
    optional_keys = ['segment200', 'instance']
    for key in optional_keys:
        file_path = os.path.join(pointcept_dir, f"{key}.npy")
        if os.path.exists(file_path):
            data[key] = np.load(file_path)
        else:
            data[key] = np.full_like(data['segment20'], -1, dtype=np.int64)

    print(f"Loaded Pointcept data from {pointcept_dir}: {data['coord'].shape[0]} points")
    #print(f"Segment20 label distribution: min={data['segment20'].min()}, max={data['segment20'].max()}")
    return data

def load_3dgs_data(path_3dgs):
    
    with open(path_3dgs, 'rb') as f:
        ply_data_3dgs = PlyData.read(f)
    vertex_data_3dgs = ply_data_3dgs['vertex']
    
    points_3dgs = np.stack([vertex_data_3dgs['x'], vertex_data_3dgs['y'], vertex_data_3dgs['z']], axis=-1)
    #normals_3dgs = np.stack([vertex_data_3dgs['nx'], vertex_data_3dgs['ny'], vertex_data_3dgs['nz']], axis=-1)
    
    raw_features_3dgs = np.hstack([
        np.vstack([vertex_data_3dgs[f'scale_{i}'] for i in range(3)]).T,  # [N, 3]
        vertex_data_3dgs['opacity'][:, None],  # [N, 1]
        np.vstack([vertex_data_3dgs[f'rot_{i}'] for i in range(4)]).T  # [N, 4]
    ])
    # 법선 벡터 크기 확인
   # norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
    #print(f"3DGS normals norm: min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
    
    # 법선이 모두 0인 경우 추정
    # if np.all(norm_3dgs == 0):
    #     pcd_3dgs = o3d.geometry.PointCloud()
    #     pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
    #     pcd_3dgs.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    #     normals_3dgs = np.asarray(pcd_3dgs.normals)
    #     norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
        #print(f"After estimation - 3DGS normals norm: min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
    
    print(f"Loaded 3DGS data from {path_3dgs}: {points_3dgs.shape[0]} points")
    # return points_3dgs, normals_3dgs, raw_features_3dgs
    return points_3dgs, raw_features_3dgs

def update_3dgs_attributes(points_3dgs, points_pointcept, colors_pointcept, normals_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept, k_neighbors=10, use_label_consistency=True, ignore_threshold=0.6):
    
    """
    3DGS 점의 속성을 Pointcept 점에서 복사.
    
    Args:
        points_3dgs (np.ndarray): 3DGS 점 좌표.
        points_pointcept (np.ndarray): Pointcept 점 좌표.
        colors_pointcept (np.ndarray): Pointcept 점 색상.
        normals_pointcept (np.ndarray): Pointcept 점 법선 벡터.
        labels_pointcept (np.ndarray): Pointcept 점 segment20 라벨.
        labels200_pointcept (np.ndarray): Pointcept 점 segment200 라벨.
        instances_pointcept (np.ndarray): Pointcept 점 instance ID.
        k_neighbors (int): 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
        ignore_threshold (float): ignore_index(-1) 라벨의 비율이 이 값 이상이면 결과 라벨을 -1로 설정.
    
    Returns:
        tuple: (colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask)
    """
    # KNN을 사용하여 가장 가까운 Pointcept 점 찾기
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(points_pointcept)
    distances, indices = nbrs.kneighbors(points_3dgs)

    # 색상, 법선, 라벨, instance 초기화
    colors_3dgs = np.zeros((len(points_3dgs), 3), dtype=np.uint8)
    normals_3dgs = np.zeros((len(points_3dgs), 3), dtype=np.float32)  # 법선 벡터 초기화
    labels_3dgs = np.full(len(points_3dgs), -1, dtype=np.int64)  # ignore_index로 초기화
    labels200_3dgs = np.full(len(points_3dgs), -1, dtype=np.int64)  # ignore_index로 초기화
    instances_3dgs = np.full(len(points_3dgs), -1, dtype=np.int64)  # ignore_index로 초기화
    mask = np.ones(len(points_3dgs), dtype=bool)

    for i in range(len(points_3dgs)):
        nearest_colors = colors_pointcept[indices[i]]
        nearest_normals = normals_pointcept[indices[i]]
        nearest_labels = labels_pointcept[indices[i]]
        nearest_labels200 = labels200_pointcept[indices[i]]
        nearest_instances = instances_pointcept[indices[i]]

        # 색상은 평균으로 계산
        colors_3dgs[i] = np.mean(nearest_colors, axis=0).astype(np.uint8)

        # 법선은 평균으로 계산 (단위 벡터로 정규화)
        avg_normal = np.mean(nearest_normals, axis=0)
        norm = np.linalg.norm(avg_normal)
        normals_3dgs[i] = avg_normal / norm if norm > 0 else avg_normal  # 0 벡터 방지

        # ignore_index(-1)의 비율 계산
        ignore_ratio_labels = np.mean(nearest_labels == -1)
        ignore_ratio_labels200 = np.mean(nearest_labels200 == -1)
        ignore_ratio_instances = np.mean(nearest_instances == -1)

        # ignore_index 비율이 임계값 이상이면 -1로 설정
        if ignore_ratio_labels >= ignore_threshold:
            labels_3dgs[i] = -1
        else:
            temp_labels = np.where(nearest_labels < 0, 0, nearest_labels)
            label_counts = np.bincount(temp_labels)
            labels_3dgs[i] = np.argmax(label_counts)

        if ignore_ratio_labels200 >= ignore_threshold:
            labels200_3dgs[i] = -1
        else:
            temp_labels200 = np.where(nearest_labels200 < 0, 0, nearest_labels200)
            label200_counts = np.bincount(temp_labels200)
            labels200_3dgs[i] = np.argmax(label200_counts)

        if ignore_ratio_instances >= ignore_threshold:
            instances_3dgs[i] = -1
        else:
            temp_instances = np.where(nearest_instances < 0, 0, nearest_instances)
            instance_counts = np.bincount(temp_instances)
            instances_3dgs[i] = np.argmax(instance_counts)

        # 라벨 일관성 체크
        if use_label_consistency:
            temp_labels = np.where(nearest_labels < 0, 0, nearest_labels)
            label_counts = np.bincount(temp_labels)
            if label_counts.max() < k_neighbors * ignore_threshold:  # 60% 이상 일관성 없으면 제외
                mask[i] = False

    
    return colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask

def remove_duplicates(points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs):
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
    return points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs

def quaternion_to_direction(quaternion):
    # 쿼터니언 정규화
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = np.divide(quaternion, norm, where=norm != 0, out=np.zeros_like(quaternion))
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    # 기준 벡터 [0, 0, 1]에 회전 적용
    directions = np.zeros((len(quaternion), 3))
    directions[:, 0] = 2 * (x * z + w * y)
    directions[:, 1] = 2 * (y * z - w * x)
    directions[:, 2] = 1 - 2 * (x * x + y * y)

    # 방향 벡터 정규화
    norm = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.divide(directions, norm, where=norm != 0, out=np.zeros_like(directions))
    
    return directions

def average_quaternions(quaternions):
    """
    Quaternion 배열의 평균 계산.
    Args:
        quaternions: (N, 4) 배열
    Returns:
        avg_quaternion: [w, x, y, z]
    """
    avg = np.mean(quaternions, axis=0)
    norm = np.linalg.norm(avg)
    return avg / (norm + 1e-8)

def process_rotation(rotation_values):
    """
    Rotation을 Quaternion 평균화 후 Direction 벡터로 변환하고 정규화.
    
    Args:
        rotation_values (np.ndarray): Quaternion 배열 [N, 4] (w, x, y, z).
    
    Returns:
        np.ndarray: 정규화된 Direction 벡터 [3].
    """
    avg_quaternion = average_quaternions(rotation_values)
    direction = quaternion_to_direction(avg_quaternion[np.newaxis, :])[0]
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else np.array([0, 0, 1], dtype=np.float32)

def voxelize_3dgs(points_3dgs, features_3dgs, voxel_size=0.02, k_neighbors=None, k_neighbors_max=15):
    print(f"Voxelizing 3DGS points with voxel_size={voxel_size}...")
    num_points_before = len(points_3dgs)

    # Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3dgs)

    # Open3D를 사용한 Voxelization
    pcd_voxelized = pcd.voxel_down_sample(voxel_size=voxel_size)
    points_voxelized = np.asarray(pcd_voxelized.points, dtype=np.float32)

    # k_neighbors 동적 설정
    avg_points_per_voxel = len(points_3dgs) / len(points_voxelized) if len(points_voxelized) > 0 else 1
    k_neighbors = min(k_neighbors_max, max(10, int(np.sqrt(avg_points_per_voxel))))
    print(f"Using k_neighbors={k_neighbors} for voxelization...")

    # 원본 점과 Voxelized 점 간의 매핑 계산
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points_3dgs)
    tree = o3d.geometry.KDTreeFlann(pcd_original)
    indices = []
    for point in points_voxelized:
        _, idx, _ = tree.search_knn_vector_3d(point, k_neighbors)
        indices.append(idx)
    indices = np.array(indices)  # (N_voxelized, k_neighbors)

    # 속성 집계
    features_voxelized = np.zeros((len(points_voxelized), 7), dtype=np.float32)  # [scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z]
    for i in range(len(points_voxelized)):
        neighbor_features = features_3dgs[indices[i]]  # (k_neighbors, 8)

        # Scale: 극단값 제거 후 평균값이 가장 큰 점 선택
        scale_values = neighbor_features[:, 0:3]  # (k_neighbors, 3)
        scale_means = np.mean(scale_values, axis=1)  # (k_neighbors,)
        lower_bound, upper_bound = np.percentile(scale_means, [1, 99])
        valid_mask = (scale_means >= lower_bound) & (scale_means <= upper_bound)

        if np.sum(valid_mask) > 0:
            valid_indices = np.where(valid_mask)[0]
            max_mean_idx = valid_indices[np.argmax(scale_means[valid_mask])]
        else:
            max_mean_idx = np.argmax(scale_means)  # 유효한 점이 없으면 전체에서 선택
        features_voxelized[i, 0:3] = scale_values[max_mean_idx]

        # Opacity: 평균값 사용
        features_voxelized[i, 3] = np.mean(neighbor_features[:, 3])
        features_voxelized[i, 3] = np.clip(features_voxelized[i, 3], 0, 1)

        # Rotation 처리
        rotation_values = neighbor_features[:, 4:8]  # (k_neighbors, 4)
        features_voxelized[i, 4:7] = process_rotation(rotation_values)

    print(f"Voxelization complete: Before {num_points_before} points, After {len(points_voxelized)} points")
    return points_voxelized, features_voxelized

def fps_knn_sampling(points, features, sample_ratio, k_neighbors=None, aggregation_method='mean'):
    # 1. FPS 샘플링
    N = len(points)
    sample_size = max(1, int(N * sample_ratio))  # 최소 1개 점 보장

    indices = np.zeros(sample_size, dtype=np.int64)
    distances = np.full(N, np.inf)
    indices[0] = np.random.randint(0, N)  # 첫 점 무작위 선택

    for i in range(1, sample_size):
        last_point = points[indices[i-1]]
        dist = np.sum((points - last_point) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        indices[i] = np.argmax(distances)

    sampled_points = points[indices]

    # 2. k_neighbors 동적 설정
    if k_neighbors is None:
        ratio = N / sample_size if sample_size > 0 else 1
        k_neighbors = max(5, min(20, int(np.sqrt(ratio) * 5)))
    print(f"Using k_neighbors={k_neighbors} for kNN...")

    # 3. kNN으로 이웃 점 찾기
    knn = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    _, neighbor_indices = knn.kneighbors(sampled_points)

    # 4. 속성 집계
    sampled_features = np.zeros((sample_size, 7))  # [scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z]
    for i in range(sample_size):
        neighbors = neighbor_indices[i]
        neighbor_features = features[neighbors]  # (k_neighbors, 8)

        # Scale: 극단값 제거 후 평균값이 가장 큰 점 선택
        scale_values = neighbor_features[:, 0:3]  # (k_neighbors, 3)
        scale_means = np.mean(scale_values, axis=1)  # (k_neighbors,)
        percentile_bounds = [1, 99] if k_neighbors >= 10 else [10, 90]
        lower_bound, upper_bound = np.percentile(scale_means, percentile_bounds)
        valid_mask = (scale_means >= lower_bound) & (scale_means <= upper_bound)

        if np.sum(valid_mask) > 0:
            valid_indices = np.where(valid_mask)[0]
            max_mean_idx = valid_indices[np.argmax(scale_means[valid_mask])]
        else:
            valid_indices = np.arange(len(scale_means))
            max_mean_idx = valid_indices[np.argmax(scale_means)]
        sampled_features[i, 0:3] = scale_values[max_mean_idx]

        # Opacity: max 또는 mean 적용
        if aggregation_method == 'max':
            sampled_features[i, 3] = np.max(neighbor_features[:, 3])
        else:  # 'mean'
            sampled_features[i, 3] = np.mean(neighbor_features[:, 3])
        sampled_features[i, 3] = np.clip(sampled_features[i, 3], 0, 1)

        # Rotation 처리
        rotation_values = neighbor_features[:, 4:8]  # (k_neighbors, 4)
        sampled_features[i, 4:7] = process_rotation(rotation_values)

    print(f"FPS + kNN Sampling: Before {N} points, After {sample_size} points, Method: {aggregation_method}")
    return sampled_points, sampled_features


def save_ply(points_merged, colors_merged, labels_merged, output_path, points_pointcept=None, colors_pointcept=None, points_3dgs=None, colors_3dgs=None, features_3dgs=None):
    """
    병합된 점과 개별 점을 PLY 파일로 저장.
    
    Args:
        points_merged (np.ndarray): 병합된 또는 3DGS 점 좌표 (N, 3).
        colors_merged (np.ndarray): 병합된 또는 3DGS 점 색상 (N, 3), 0-255 범위.
        labels_merged (np.ndarray): 병합된 또는 3DGS 점 라벨 (N,).
        output_path (str): 저장할 기본 PLY 파일 경로 (예: 'path/exp.ply').
        points_pointcept (np.ndarray, optional): Pointcept 점 좌표 (M, 3).
        colors_pointcept (np.ndarray, optional): Pointcept 점 색상 (M, 3).
        points_3dgs (np.ndarray, optional): 3DGS 점 좌표 (N, 3).
        colors_3dgs (np.ndarray, optional): 3DGS 점 색상 (N, 3).
        features_3dgs (np.ndarray, optional): 3DGS 점의 속성 (N, 7), [scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z].
    """
    # 필수 인자 차원 확인
    assert points_merged.shape[0] == colors_merged.shape[0] == labels_merged.shape[0], "Merged points, colors, and labels must have the same length"
    assert points_merged.shape[1] == 3 and colors_merged.shape[1] == 3, "Merged points and colors must have 3 dimensions"

    # 선택적 인자 차원 확인 (입력 시)
    if points_pointcept is not None and colors_pointcept is not None:
        assert points_pointcept.shape[0] == colors_pointcept.shape[0], "Pointcept points and colors must have the same length"
        assert points_pointcept.shape[1] == 3 and colors_pointcept.shape[1] == 3, "Pointcept points and colors must have 3 dimensions"
    if points_3dgs is not None and colors_3dgs is not None:
        assert points_3dgs.shape[0] == colors_3dgs.shape[0], "3DGS points and colors must have the same length"
        assert points_3dgs.shape[1] == 3 and colors_3dgs.shape[1] == 3, "3DGS points and colors must have 3 dimensions"
        # features_3dgs가 제공될 경우 차원 확인
        if features_3dgs is not None:
            print(f'dim: {features_3dgs.shape[1]}')
            assert points_3dgs.shape[0] == features_3dgs.shape[0], "3DGS points and features must have the same length"
            assert features_3dgs.shape[1] == 7, "features_3dgs must have 7 dimensions (scale: 3, opacity: 1, rotation: 3)"

    # 라벨 색상 생성 (merged_label.ply용)
    unique_labels = np.unique(labels_merged)


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
    label_colors = {}
    
    # -1 라벨은 흰색으로 처리
    if -1 in unique_labels:
        label_colors[-1] = np.array([0, 0, 0], dtype=np.uint8)  # 흰색

    # 나머지 라벨에 대해 색상 생성 (0 이상)
    for label in unique_labels:
        if label != -1:
            if label < 20:
                # ScanNet 20 클래스 색상 팔레트 (예시)
                label_colors[label] = np.array(SCANNET_20_COLORS[label], dtype=np.uint8)
            else:
                label_colors[label] = np.random.randint(0, 256, size=3, dtype=np.uint8)

    # 라벨 분포 출력
    print("Label distribution:")
    for label in unique_labels:
        count = np.sum(labels_merged == label)
        print(f"Label {label}: {count} points")

    # 1. Merged PLY (점 색상 유지)
    vertex_merged = np.zeros(points_merged.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    vertex_merged['x'] = points_merged[:, 0].astype('f4')
    vertex_merged['y'] = points_merged[:, 1].astype('f4')
    vertex_merged['z'] = points_merged[:, 2].astype('f4')
    vertex_merged['red'] = colors_merged[:, 0].astype('u1')
    vertex_merged['green'] = colors_merged[:, 1].astype('u1')
    vertex_merged['blue'] = colors_merged[:, 2].astype('u1')
    el_merged = PlyElement.describe(vertex_merged, 'vertex')
    PlyData([el_merged], text=True).write(output_path)
    print(f"Saved merged PLY file to {output_path}")

    # 2. Merged Label PLY (라벨 색상 사용)
    vertex_merged_label = np.zeros(points_merged.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', 'i4')
    ])
    vertex_merged_label['x'] = points_merged[:, 0].astype('f4')
    vertex_merged_label['y'] = points_merged[:, 1].astype('f4')
    vertex_merged_label['z'] = points_merged[:, 2].astype('f4')
    for i in range(len(points_merged)):
        vertex_merged_label['red'][i] = label_colors[labels_merged[i]][0]
        vertex_merged_label['green'][i] = label_colors[labels_merged[i]][1]
        vertex_merged_label['blue'][i] = label_colors[labels_merged[i]][2]
    vertex_merged_label['label'] = labels_merged.astype('i4')
    el_merged_label = PlyElement.describe(vertex_merged_label, 'vertex')
    merged_label_path = output_path.replace('.ply', '_label.ply')
    PlyData([el_merged_label], text=True).write(merged_label_path)
    print(f"Saved merged label PLY file to {merged_label_path}")

    # 3. 3DGS PLY (3DGS 점만, 점 색상 유지, 모든 features 추가)
    if points_3dgs is not None and colors_3dgs is not None:
        # vertex dtype 정의 (features 포함 여부에 따라 동적 설정)
        if features_3dgs is not None:
            # Features 분리
            scale_3dgs = features_3dgs[:, :3]  # (N, 3)
            opacity_3dgs = features_3dgs[:, 3:4]  # (N, 1)
            rotation_3dgs = features_3dgs[:, 4:]  # (N, 3)

            # Opacity 클리핑
            opacity_3dgs = np.clip(opacity_3dgs, 0, 1)

            # Scale 평균값 계산
            scale_mean = np.mean(scale_3dgs, axis=1)  # (N,)

            # Rotation을 z축과의 각도로 변환 (theta)
            # rotation_3dgs는 단위 벡터로 가정 (norm ≈ 1)
            # theta = arccos(z), z축과의 각도 (라디안, [0, π])
            rotation_theta = np.arccos(np.clip(rotation_3dgs[:, 2], -1, 1))  # (N,), clip으로 안정성 확보

            vertex_3dgs = np.zeros(points_3dgs.shape[0], dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('scalar_scale_mean', 'f4'),  # Scale 평균
                ('scalar_opacity', 'f4'),    # Opacity
                ('scalar_rotation_theta', 'f4')  # Rotation theta (z축과의 각도)
            ])
            vertex_3dgs['scalar_scale_mean'] = scale_mean.astype('f4')
            vertex_3dgs['scalar_opacity'] = opacity_3dgs[:, 0].astype('f4')
            vertex_3dgs['scalar_rotation_theta'] = rotation_theta.astype('f4')
        else:
            vertex_3dgs = np.zeros(points_3dgs.shape[0], dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
        
        vertex_3dgs['x'] = points_3dgs[:, 0].astype('f4')
        vertex_3dgs['y'] = points_3dgs[:, 1].astype('f4')
        vertex_3dgs['z'] = points_3dgs[:, 2].astype('f4')
        vertex_3dgs['red'] = colors_3dgs[:, 0].astype('u1')
        vertex_3dgs['green'] = colors_3dgs[:, 1].astype('u1')
        vertex_3dgs['blue'] = colors_3dgs[:, 2].astype('u1')
        el_3dgs = PlyElement.describe(vertex_3dgs, 'vertex')
        dgs_path = output_path.replace('.ply', '_3dgs.ply')
        PlyData([el_3dgs], text=True).write(dgs_path)
        print(f"Saved 3DGS PLY file to {dgs_path}")

    # 4. Pointcept PLY (Pointcept 점만, 점 색상 유지, 선택적 생성)
    if points_pointcept is not None and colors_pointcept is not None:
        vertex_pointcept = np.zeros(points_pointcept.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        vertex_pointcept['x'] = points_pointcept[:, 0].astype('f4')
        vertex_pointcept['y'] = points_pointcept[:, 1].astype('f4')
        vertex_pointcept['z'] = points_pointcept[:, 2].astype('f4')
        vertex_pointcept['red'] = colors_pointcept[:, 0].astype('u1')
        vertex_pointcept['green'] = colors_pointcept[:, 1].astype('u1')
        vertex_pointcept['blue'] = colors_pointcept[:, 2].astype('u1')
        el_pointcept = PlyElement.describe(vertex_pointcept, 'vertex')
        pointcept_path = output_path.replace('.ply', '_pointcept.ply')
        PlyData([el_pointcept], text=True).write(pointcept_path)
        print(f"Saved Pointcept PLY file to {pointcept_path}")