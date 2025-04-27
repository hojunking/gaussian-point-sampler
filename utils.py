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

    # 음수 값 경고 (필터링 또는 변환하지 않음)
    # if np.any(data['segment20'] < 0):
    #     print(f"Warning: Negative values found in segment20: {data['segment20'][data['segment20'] < 0]}")
    #     print(f"Keeping negative values as ignore_index (-1) following Pointcept's approach")

    # 선택적 데이터 로드 (segment200, instance)
    optional_keys = ['segment200', 'instance']
    for key in optional_keys:
        file_path = os.path.join(pointcept_dir, f"{key}.npy")
        if os.path.exists(file_path):
            data[key] = np.load(file_path)
            # 음수 값 경고 (필터링 또는 변환하지 않음)
            # if np.any(data[key] < 0):
            #     print(f"Warning: Negative values found in {key}: {data[key][data[key] < 0]}")
            #     print(f"Keeping negative values as ignore_index (-1) following Pointcept's approach")
        else:
            #print(f"Warning: {key}.npy not found in {pointcept_dir}, setting to ignore_index (-1)")
            data[key] = np.full_like(data['segment20'], -1, dtype=np.int64)

    print(f"Loaded Pointcept data from {pointcept_dir}: {data['coord'].shape[0]} points")
    print(f"Segment20 label distribution: min={data['segment20'].min()}, max={data['segment20'].max()}")
    return data

def load_3dgs_data(path_3dgs):
    """
    3DGS Point Cloud 데이터를 로드.
    
    Args:
        path_3dgs (str): 3DGS PLY 파일 경로.
    
    Returns:
        tuple: (points_3dgs, normals_3dgs, vertex_data_3dgs).
    """
    with open(path_3dgs, 'rb') as f:
        ply_data_3dgs = PlyData.read(f)
    vertex_data_3dgs = ply_data_3dgs['vertex']
    
    points_3dgs = np.stack([vertex_data_3dgs['x'], vertex_data_3dgs['y'], vertex_data_3dgs['z']], axis=-1)
    normals_3dgs = np.stack([vertex_data_3dgs['nx'], vertex_data_3dgs['ny'], vertex_data_3dgs['nz']], axis=-1)
    
    raw_features_3dgs = np.hstack([
        np.vstack([vertex_data_3dgs[f'scale_{i}'] for i in range(3)]).T,  # [N, 3]
        vertex_data_3dgs['opacity'][:, None],  # [N, 1]
        np.vstack([vertex_data_3dgs[f'rot_{i}'] for i in range(4)]).T  # [N, 4]
    ])
    
    # 법선 벡터 크기 확인
    norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
    #print(f"3DGS normals norm: min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
    
    # 법선이 모두 0인 경우 추정
    if np.all(norm_3dgs == 0):
        print("All 3DGS normals are zero. Estimating normals...")
        pcd_3dgs = o3d.geometry.PointCloud()
        pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
        pcd_3dgs.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normals_3dgs = np.asarray(pcd_3dgs.normals)
        norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
        #print(f"After estimation - 3DGS normals norm: min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
    
    print(f"Loaded 3DGS data from {path_3dgs}: {points_3dgs.shape[0]} points")
    return points_3dgs, normals_3dgs, raw_features_3dgs

def update_3dgs_attributes(points_3dgs, points_pointcept, colors_pointcept, normals_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept, k_neighbors=5, use_label_consistency=True, ignore_threshold=0.6):
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
            if label_counts.max() < k_neighbors * 0.6:  # 60% 이상 일관성 없으면 제외
                mask[i] = False

    return colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask

def remove_duplicates(points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs):
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
    
    Returns:
        tuple: (points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs)
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
    return points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs

def voxelize_3dgs(points_3dgs, features_3dgs, voxel_size=0.02, k_neighbors=5):
    print(f"Voxelizing 3DGS points with voxel_size={voxel_size}...")
    num_points_before = len(points_3dgs)

    # Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3dgs)

    # Open3D를 사용한 Voxelization
    pcd_voxelized = pcd.voxel_down_sample(voxel_size=voxel_size)
    points_voxelized = np.asarray(pcd_voxelized.points, dtype=np.float32)

    # 원본 점과 Voxelized 점 간의 매핑 계산
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(points_3dgs)
    tree = o3d.geometry.KDTreeFlann(pcd_original)
    indices = []
    for point in points_voxelized:
        _, idx, _ = tree.search_knn_vector_3d(point, k_neighbors)  # k_neighbors개의 이웃 점 인덱스
        indices.append(idx)
    indices = np.array(indices)  # (N_voxelized, k_neighbors)

    # 전처리된 속성 평균화
    features_voxelized = np.zeros((len(points_voxelized), features_3dgs.shape[1]), dtype=np.float32)
    for i in range(len(points_voxelized)):
        features_voxelized[i] = np.mean(features_3dgs[indices[i]], axis=0)

    # 방향 벡터 (rotation) 정규화
    if features_voxelized.shape[1] >= 7:  # features_3dgs: [scale_x, scale_y, scale_z, opacity, dir_x, dir_y, dir_z]
        rotation_voxelized = features_voxelized[:, 4:7]  # dir_x, dir_y, dir_z
        rotation_norms = np.linalg.norm(rotation_voxelized, axis=1, keepdims=True)
        features_voxelized[:, 4:7] = np.divide(rotation_voxelized, rotation_norms, where=rotation_norms > 0, out=rotation_voxelized)


    print(f"Voxelization complete: Before {num_points_before} points, After {len(points_voxelized)} points")
    return points_voxelized, features_voxelized

def save_ply(points, colors, labels, output_path, save_separate_labels=False, points_pointcept=None, colors_pointcept=None, points_3dgs=None):
    """
    점과 색상, 라벨을 PLY 파일로 저장.
    
    Args:
        points (np.ndarray): 점 좌표 (N, 3).
        colors (np.ndarray): 점 색상 (N, 3), 0-255 범위.
        labels (np.ndarray): 점 라벨 (N,).
        output_path (str): 저장할 PLY 파일 경로.
        save_separate_labels (bool): 라벨만 별도로 저장할지 여부.
        points_pointcept (np.ndarray): Pointcept 점 좌표 (옵션).
        colors_pointcept (np.ndarray): Pointcept 점 색상 (옵션).
        points_3dgs (np.ndarray): 3DGS 점 좌표 (옵션).
    """
    # 점과 색상, 라벨 확인
    assert points.shape[0] == colors.shape[0] == labels.shape[0], "Points, colors, and labels must have the same length"
    assert points.shape[1] == 3, "Points must have 3 dimensions"
    assert colors.shape[1] == 3, "Colors must have 3 dimensions"

    # 라벨 색상 생성
    unique_labels = np.unique(labels)
    label_colors = {}
    
    # -1 라벨에 눈에 띄는 색상 (밝은 노란색) 할당
    if -1 in unique_labels:
        label_colors[-1] = np.array([255, 255, 0], dtype=np.uint8)  # 밝은 노란색

    # 나머지 라벨에 대해 색상 생성 (0 이상)
    for label in unique_labels:
        if label != -1:  # -1은 이미 처리됨
            # 고정된 색상 팔레트 사용 (예: 20개 클래스에 대해)
            if label < 20:
                # ScanNet 20 클래스 색상 팔레트 (예시, 실제 팔레트로 대체 가능)
                palette = [
                    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
                    [0, 255, 255], [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0],
                    [128, 0, 128], [0, 128, 128], [255, 128, 0], [255, 0, 128], [128, 255, 0],
                    [0, 255, 128], [128, 128, 128], [255, 128, 128], [128, 255, 128], [128, 128, 255]
                ]
                label_colors[label] = np.array(palette[label % len(palette)], dtype=np.uint8)
            else:
                # 20 이상의 라벨은 랜덤 색상
                label_colors[label] = np.random.randint(0, 256, size=3, dtype=np.uint8)

    # 라벨 색상 분포 출력
    print("Label colors distribution:")
    for channel, channel_name in enumerate(['Red', 'Green', 'Blue']):
        channel_values = [color[channel] for color in label_colors.values()]
        print(f"{channel_name}: min={min(channel_values)}, max={max(channel_values)}")

    # 라벨 분포 출력
    print("Label distribution:")
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"Label {label}: {count} points")

    # PLY 데이터 생성
    vertex = np.zeros(points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('label', 'i4')
    ])
    vertex['x'] = points[:, 0].astype('f4')
    vertex['y'] = points[:, 1].astype('f4')
    vertex['z'] = points[:, 2].astype('f4')
    
    # 색상 설정
    for i in range(len(points)):
        vertex['red'][i] = label_colors[labels[i]][0]
        vertex['green'][i] = label_colors[labels[i]][1]
        vertex['blue'][i] = label_colors[labels[i]][2]
    vertex['label'] = labels.astype('i4')

    # PLY 파일 저장
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(output_path)
    print(f"Saved merged PLY file to {output_path}")

    # 라벨만 별도로 저장
    if save_separate_labels:
        vertex_labels = np.zeros(points.shape[0], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        vertex_labels['x'] = points[:, 0].astype('f4')
        vertex_labels['y'] = points[:, 1].astype('f4')
        vertex_labels['z'] = points[:, 2].astype('f4')
        for i in range(len(points)):
            vertex_labels['red'][i] = label_colors[labels[i]][0]
            vertex_labels['green'][i] = label_colors[labels[i]][1]
            vertex_labels['blue'][i] = label_colors[labels[i]][2]
        el_labels = PlyElement.describe(vertex_labels, 'vertex')
        PlyData([el_labels], text=True).write(output_path.replace('.ply', '_labels.ply'))
        print(f"Saved labels-only PLY file to {output_path.replace('.ply', '_labels.ply')}")

    # Pointcept 점 별도 저장
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
        PlyData([el_pointcept], text=True).write(output_path.replace('.ply', '_pointcept.ply'))
        print(f"Saved Pointcept PLY file to {output_path.replace('.ply', '_pointcept.ply')}")