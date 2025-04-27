#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_scale09_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.9 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features scale
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_scale09_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.9 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features scale

