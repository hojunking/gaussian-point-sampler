#!/bin/bash

# 실행 스크립트: ScanNet 데이터를 3DGS와 병합하여 Pointcept 포맷으로 저장

# Python 실행 환경 설정 (필요 시 수정)
PYTHON_EXEC=python3

# 기본 실행 명령어
$PYTHON_EXEC merge_3dgs_point_cloud.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance0001_scale05_vox004-l \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --scale_ratio 0.5 \
    --opacity_ratio 0 \
    --k_neighbors 5 \
    --use_label_consistency \
    --enable_pointcept_distance \
    --pdistance 0.001 \
    --voxelize \
    --voxel_size 0.04


$PYTHON_EXEC merge_3dgs_point_cloud.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance00005_scale05_vox004-l \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --scale_ratio 0.5 \
    --opacity_ratio 0 \
    --k_neighbors 5 \
    --use_label_consistency \
    --enable_pointcept_distance \
    --pdistance 0.0005 \
    --voxelize \
    --voxel_size 0.04

$PYTHON_EXEC merge_3dgs_point_cloud.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance0005_scale05_vox004-l \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --scale_ratio 0.5 \
    --opacity_ratio 0 \
    --k_neighbors 5 \
    --use_label_consistency \
    --enable_pointcept_distance \
    --pdistance 0.005 \
    --voxelize \
    --voxel_size 0.04

# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance00005_scale075_opacity05_keep-dup \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0.75 \
#     --opacity_ratio 0.5 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
#     --pdistance 0.0005 \


# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance00005_scale05_opacity05_keep-dup \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0.5 \
#     --opacity_ratio 0.5 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
#     --pdistance 0.0005 \



# 기본 실행 명령어
# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance0001_scale05 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0.5 \
#     --opacity_ratio 0 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
#     --pdistance 0.001 \


# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance0001_scale075_opacity05 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0.75 \
#     --opacity_ratio 0.5 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
#     --pdistance 0.001 \


# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance0001_scale05_opacity05 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0.5 \
#     --opacity_ratio 0.5 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
#     --pdistance 0.001 \
