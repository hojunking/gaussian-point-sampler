import numpy as np
import open3d as o3d
from plyfile import PlyData
import os
from plyfile import PlyData, PlyElement


def load_pointcept_data(pointcept_dir):
    """
    Pointcept 데이터를 .npy 파일에서 로드.
    
    Args:
        pointcept_dir (str): Pointcept 데이터 디렉토리 (예: scannet/train/scene0000_00).
    
    Returns:
        dict: 로드된 데이터 (coord, color, normal, segment20, segment200, instance).
    """
    data = {}
    for key in ['coord', 'color', 'normal', 'segment20', 'segment200', 'instance']:
        file_path = os.path.join(pointcept_dir, f"{key}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{key}.npy not found in {pointcept_dir}")
        data[key] = np.load(file_path)

    # 라벨 데이터 검증 및 수정
    for key in ['segment20', 'segment200', 'instance']:
        if np.any(data[key] < 0):
            #print(f"Warning: Negative values found in {key}: {data[key][data[key] < 0]}")
            #print(f"Converting negative values in {key} to 0")
            data[key] = np.where(data[key] < 0, 0, data[key])

    print(f"Loaded Pointcept data from {pointcept_dir}: {data['coord'].shape[0]} points")
    print(f"Segment20 label distribution: min={data['segment20'].min()}, max={data['segment20'].max()}")
    print(f"Segment200 label distribution: min={data['segment200'].min()}, max={data['segment200'].max()}")
    print(f"Instance label distribution: min={data['instance'].min()}, max={data['instance'].max()}")
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
    
    # 법선 벡터 크기 확인
    norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
    print(f"3DGS normals norm (before estimation): min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
    
    # 법선이 모두 0인 경우 추정
    if np.all(norm_3dgs == 0):
        print("All 3DGS normals are zero. Estimating normals...")
        pcd_3dgs = o3d.geometry.PointCloud()
        pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
        pcd_3dgs.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
        )
        pcd_3dgs.orient_normals_consistent_tangent_plane(k=20)
        normals_3dgs = np.asarray(pcd_3dgs.normals)
        norm_3dgs = np.linalg.norm(normals_3dgs, axis=-1)
        print(f"After estimation - 3DGS normals norm: min={norm_3dgs.min():.4f}, max={norm_3dgs.max():.4f}, zero_count={np.sum(norm_3dgs == 0)}")
    
    print(f"Loaded 3DGS data from {path_3dgs}: {points_3dgs.shape[0]} points")
    return points_3dgs, normals_3dgs, vertex_data_3dgs

def update_3dgs_attributes(points_3dgs, points_pointcept, colors_pointcept, labels_pointcept, labels200_pointcept, instances_pointcept, k_neighbors=5, use_label_consistency=True):
    """
    3DGS 점의 색상과 라벨을 근처 Pointcept 점에서 복사.
    
    Args:
        points_3dgs (np.ndarray): 3DGS 점 좌표 (N, 3).
        points_pointcept (np.ndarray): Pointcept 점 좌표 (M, 3).
        colors_pointcept (np.ndarray): Pointcept 점 색상 (M, 3).
        labels_pointcept (np.ndarray): Pointcept 점 라벨 (M,).
        labels200_pointcept (np.ndarray): Pointcept 점 200 클래스 라벨 (M,).
        instances_pointcept (np.ndarray): Pointcept 점 인스턴스 라벨 (M,).
        k_neighbors (int): 라벨 복사 및 일관성 체크에 사용할 이웃 점 개수.
        use_label_consistency (bool): 라벨 일관성 필터링 사용 여부.
    
    Returns:
        tuple: (colors_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask).
    """
    # KDTree 생성
    pcd_pointcept = o3d.geometry.PointCloud()
    pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
    tree = o3d.geometry.KDTreeFlann(pcd_pointcept)

    # 3DGS 점의 속성 초기화
    colors_3dgs = np.zeros_like(points_3dgs, dtype=np.uint8)
    labels_3dgs = np.zeros(len(points_3dgs), dtype=np.int32)
    labels200_3dgs = np.zeros(len(points_3dgs), dtype=np.int32)
    instances_3dgs = np.zeros(len(points_3dgs), dtype=np.int32)

    # 유효한 라벨이 있는지 확인하기 위한 초기 마스크
    initial_mask = np.ones(len(points_3dgs), dtype=bool)

    # 다수결 기반으로 속성 복사
    for i, point in enumerate(points_3dgs):
        [k, idx, _] = tree.search_knn_vector_3d(point, k_neighbors)
        if k > 0:
            colors_3dgs[i] = colors_pointcept[idx[0]]  # 색상은 가장 가까운 점에서 복사
            # segment20 라벨
            neighbor_labels = labels_pointcept[idx]
            valid_labels = neighbor_labels[neighbor_labels >= 0]  # -1 제외
            if len(valid_labels) > 0:
                labels_3dgs[i] = np.bincount(valid_labels).argmax()
            else:
                initial_mask[i] = False  # 유효한 라벨이 없으면 제거
                continue
            # segment200 라벨
            neighbor_labels200 = labels200_pointcept[idx]
            valid_labels200 = neighbor_labels200[neighbor_labels200 >= 0]  # -1 제외
            if len(valid_labels200) > 0:
                labels200_3dgs[i] = np.bincount(valid_labels200).argmax()
            else:
                initial_mask[i] = False  # 유효한 라벨이 없으면 제거
                continue
            # instance 라벨
            neighbor_instances = instances_pointcept[idx]
            valid_instances = neighbor_instances[neighbor_instances >= 0]  # -1 제외
            if len(valid_instances) > 0:
                instances_3dgs[i] = np.bincount(valid_instances).argmax()
            else:
                initial_mask[i] = False  # 유효한 라벨이 없으면 제거
                continue
    
    # 유효한 라벨이 없는 점 제거
    points_3dgs = points_3dgs[initial_mask]
    colors_3dgs = colors_3dgs[initial_mask]
    labels_3dgs = labels_3dgs[initial_mask]
    labels200_3dgs = labels200_3dgs[initial_mask]
    instances_3dgs = instances_3dgs[initial_mask]

    print(f"After removing points with no valid labels: {len(points_3dgs)} 3DGS points remaining")
    print(f"Updated attributes for {len(points_3dgs)} 3DGS points using majority voting")

    # 라벨 일관성 기반 필터링
    mask = np.ones(len(points_3dgs), dtype=bool)
    if use_label_consistency:
        for i, point in enumerate(points_3dgs):
            [k, idx, _] = tree.search_knn_vector_3d(point, k_neighbors)
            if k < k_neighbors:
                mask[i] = False
                continue
            
            # 주변 Pointcept 점의 라벨과 비교
            neighbor_labels = labels_pointcept[idx]
            point_label = labels_3dgs[i]
            valid_neighbor_labels = neighbor_labels[neighbor_labels >= 0]
            if len(valid_neighbor_labels) == 0 or not np.any(valid_neighbor_labels == point_label):
                mask[i] = False
        
        num_points_pruned = len(points_3dgs) - np.sum(mask)
        print(f"Label Consistency Filtering: Pruned {num_points_pruned} points, Remaining {np.sum(mask)} points")
    
    return colors_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, mask

def remove_duplicates(points_3dgs, points_pointcept, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs):
    """
    3DGS 점과 Pointcept 점 간의 중복 점 제거 (3DGS 점 제거).
    
    Args:
        points_3dgs (np.ndarray): 3DGS 점 좌표 (N, 3).
        points_pointcept (np.ndarray): Pointcept 점 좌표 (M, 3).
        normals_3dgs (np.ndarray): 3DGS 점 법선 (N, 3).
        labels_3dgs (np.ndarray): 3DGS 점 라벨 (N,).
        labels200_3dgs (np.ndarray): 3DGS 점 200 클래스 라벨 (N,).
        instances_3dgs (np.ndarray): 3DGS 점 인스턴스 라벨 (N,).
        colors_3dgs (np.ndarray): 3DGS 점 색상 (N, 3).
    
    Returns:
        tuple: 중복 제거된 (points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs).
    """
    pcd_pointcept = o3d.geometry.PointCloud()
    pcd_pointcept.points = o3d.utility.Vector3dVector(points_pointcept)
    tree = o3d.geometry.KDTreeFlann(pcd_pointcept)

    indices_to_remove_3dgs = []
    for i, point in enumerate(points_3dgs):
        [k, idx, _] = tree.search_knn_vector_3d(point, 1)
        if k > 0 and np.linalg.norm(points_pointcept[idx[0]] - point) < 1e-6:
            indices_to_remove_3dgs.append(i)
    
    indices_to_keep_3dgs = np.setdiff1d(np.arange(len(points_3dgs)), indices_to_remove_3dgs)
    points_3dgs = points_3dgs[indices_to_keep_3dgs]
    colors_3dgs = colors_3dgs[indices_to_keep_3dgs]
    normals_3dgs = normals_3dgs[indices_to_keep_3dgs]
    labels_3dgs = labels_3dgs[indices_to_keep_3dgs]
    labels200_3dgs = labels200_3dgs[indices_to_keep_3dgs]
    instances_3dgs = instances_3dgs[indices_to_keep_3dgs]
    print(f"After removing duplicates - 3DGS points: {len(points_3dgs)}")
    return points_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs, colors_3dgs

def remove_outliers_3dgs(points_3dgs, colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs):
    """
    3DGS 점에 대해 Statistical Outlier Removal 적용.
    
    Args:
        points_3dgs (np.ndarray): 3DGS 점 좌표 (N, 3).
        colors_3dgs (np.ndarray): 3DGS 점 색상 (N, 3).
        normals_3dgs (np.ndarray): 3DGS 점 법선 (N, 3).
        labels_3dgs (np.ndarray): 3DGS 점 라벨 (N,).
        labels200_3dgs (np.ndarray): 3DGS 점 200 클래스 라벨 (N,).
        instances_3dgs (np.ndarray): 3DGS 점 인스턴스 라벨 (N,).
    
    Returns:
        tuple: Outlier 제거된 (points_3dgs, colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs).
    """
    pcd_3dgs = o3d.geometry.PointCloud()
    pcd_3dgs.points = o3d.utility.Vector3dVector(points_3dgs)
    pcd_3dgs.colors = o3d.utility.Vector3dVector(colors_3dgs / 255.0)
    pcd_3dgs.normals = o3d.utility.Vector3dVector(normals_3dgs)
    pcd_3dgs, ind = pcd_3dgs.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    points_3dgs = np.asarray(pcd_3dgs.points)
    colors_3dgs = (np.asarray(pcd_3dgs.colors) * 255).astype(np.uint8)
    normals_3dgs = np.asarray(pcd_3dgs.normals)
    labels_3dgs = labels_3dgs[ind]
    labels200_3dgs = labels200_3dgs[ind]
    instances_3dgs = instances_3dgs[ind]
    print(f"After 3DGS Outlier Removal - 3DGS points: {len(points_3dgs)}")
    return points_3dgs, colors_3dgs, normals_3dgs, labels_3dgs, labels200_3dgs, instances_3dgs


def save_ply(points, colors, labels, output_path, save_separate_labels=True, points_pointcept=None, points_3dgs=None):
    """
    PLY 파일로 저장하는 함수.
    
    Args:
        points (np.ndarray): 점 좌표 (N, 3).
        colors (np.ndarray): 점 색상 (N, 3), 0~1 범위.
        labels (np.ndarray): 점 라벨 (N,).
        output_path (str): 저장할 PLY 파일 경로.
        save_separate_labels (bool): 라벨만 포함된 별도의 PLY 파일을 저장할지 여부.
        points_pointcept (np.ndarray): Pointcept 점 좌표 (M, 3), 선택적.
        points_3dgs (np.ndarray): 3DGS 점 좌표 (K, 3), 선택적.
    """
    # Merged Point Cloud 저장
    vertex = np.array(
        [(p[0], p[1], p[2], c[0] * 255, c[1] * 255, c[2] * 255, l)
         for p, c, l in zip(points, colors, labels)],
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('label', 'i4')
        ]
    )
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=True).write(output_path)
    print(f"Saved merged PLY file to {output_path}")

    # 라벨만 포함된 PLY 파일 저장
    if save_separate_labels:
        label_path = output_path.replace(".ply", "_labels.ply")
        vertex_labels = np.array(
            [(p[0], p[1], p[2], l) for p, l in zip(points, labels)],
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('label', 'i4')
            ]
        )
        el_labels = PlyElement.describe(vertex_labels, 'vertex')
        PlyData([el_labels], text=True).write(label_path)
        print(f"Saved labels-only PLY file to {label_path}")

    # Pointcept 점과 3DGS 점 별도로 저장 (선택적)
    if points_pointcept is not None:
        pointcept_path = output_path.replace(".ply", "_pointcept.ply")
        vertex_pointcept = np.array(
            [(p[0], p[1], p[2]) for p in points_pointcept],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
        el_pointcept = PlyElement.describe(vertex_pointcept, 'vertex')
        PlyData([el_pointcept], text=True).write(pointcept_path)
        print(f"Saved Pointcept PLY file to {pointcept_path}")

    if points_3dgs is not None:
        dgs_path = output_path.replace(".ply", "_3dgs.ply")
        vertex_3dgs = np.array(
            [(p[0], p[1], p[2]) for p in points_3dgs],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        )
        el_3dgs = PlyElement.describe(vertex_3dgs, 'vertex')
        PlyData([el_3dgs], text=True).write(dgs_path)
        print(f"Saved 3DGS PLY file to {dgs_path}")