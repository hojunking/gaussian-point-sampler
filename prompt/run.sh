#!/bin/bash

# 실행 스크립트: ScanNet 데이터를 3DGS와 병합하여 Pointcept 포맷으로 저장

# Python 실행 환경 설정 (필요 시 수정)
PYTHON_EXEC=python3

# # 기본 실행 명령어
# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance_max00002 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0 \
#     --opacity_ratio 0 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \

# # 기본 실행 명령어
# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance_max00002_label-knn10 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0 \
#     --opacity_ratio 0 \
#     --k_neighbors 10 \
#     --use_label_consistency \
#     --enable_pointcept_distance \

$PYTHON_EXEC merge_3dgs_point_cloud.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance_max000001 \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --scale_ratio 0 \
    --opacity_ratio 0 \
    --k_neighbors 5 \
    --use_label_consistency \
    --enable_pointcept_distance \
# 기본 실행 명령어
# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/opacity05 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0 \
#     --opacity_ratio 0.5 \
#     --k_neighbors 5 \
#     --use_label_consistency \
 #   --enable_pointcept_distance \

# 기본 실행 명령어
# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/opacity75 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0 \
#     --opacity_ratio 0.75 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#    --enable_pointcept_distance \
# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance_max0001 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0 \
#     --opacity_ratio 0 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
    #  --enable_sor \
    #--enable_normal \
# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0.5 \
#     --opacity_ratio 0 \
#     --k_neighbors 5 \
#     --use_label_consistency \

# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/scale075 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0.75 \
#     --opacity_ratio 0 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#    --enable_pointcept_distance \
# $PYTHON_EXEC merge_3dgs_point_cloud.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/pdistance_max00005 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --scale_ratio 0 \
#     --opacity_ratio 0 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
#     --num_workers 3