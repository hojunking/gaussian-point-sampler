#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept


python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_rotation02_rotation \
  --scale_ratio 0 \
  --data_type samples100 \
  --scale_ratio 0 \
  --rotation_ratio 0.2 \
  --opacity_ratio 0 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --use_features rotation

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_rotation04_rotation \
  --scale_ratio 0 \
  --data_type samples100 \
  --scale_ratio 0 \
  --rotation_ratio 0.4 \
  --opacity_ratio 0 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --use_features rotation

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_rotation06_rotation \
  --scale_ratio 0 \
  --data_type samples100 \
  --scale_ratio 0 \
  --rotation_ratio 0.6 \
  --opacity_ratio 0 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --use_features rotation


python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_rotation02_rotation \
  --scale_ratio 0 \
  --data_type samples100 \
  --scale_ratio 0 \
  --rotation_ratio 0.2 \
  --opacity_ratio 0 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_rotation04_rotation \
  --scale_ratio 0 \
  --data_type samples100 \
  --scale_ratio 0 \
  --rotation_ratio 0.4 \
  --opacity_ratio 0 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_rotation06_rotation \
  --scale_ratio 0 \
  --data_type samples100 \
  --scale_ratio 0 \
  --rotation_ratio 0.6 \
  --opacity_ratio 0 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation
