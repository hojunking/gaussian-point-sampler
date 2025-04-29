#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept

# attr_weight scale rotation opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-rotation05_ratio02 \
#   --data_type samples100 \
#   --pruning_ratio 0.2 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_weight 0.5 0.5 0 \
#   --use_features scale rotation
  
python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-rotation05_ratio04 \
  --data_type samples100 \
  --pruning_ratio 0.4 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_weight 0.5 0.5 0 \
  --use_features scale rotation

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-rotation05_ratio06 \
  --data_type samples100 \
  --pruning_ratio 0.6 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_weight 0.5 0.5 0 \
  --use_features scale rotation

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-opacity05_ratio02 \
  --data_type samples100 \
  --pruning_ratio 0.2 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_weight 0.5 0.5 0 \
  --use_features scale opacity
  
python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-opacity05_ratio04 \
  --data_type samples100 \
  --pruning_ratio 0.4 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_weight 0.5 0.5 0 \
  --use_features scale opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-opacity05_ratio06 \
  --data_type samples100 \
  --pruning_ratio 0.6 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_weight 0.5 0.5 0 \
  --use_features scale opacity


python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/rotation05-opacity05_ratio02 \
  --data_type samples100 \
  --pruning_ratio 0.2 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_weight 0.5 0.5 0 \
  --use_features rotation opacity
  
python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/rotation05-opacity05_ratio04 \
  --data_type samples100 \
  --pruning_ratio 0.4 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_weight 0.5 0.5 0 \
  --use_features rotation opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/rotation05-opacity05_ratio06 \
  --data_type samples100 \
  --pruning_ratio 0.6 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_weight 0.5 0.5 0 \
  --use_features rotation opacity
# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_rotation04_rotation \
#   --scale_ratio 0 \
#   --data_type samples100 \
#   --scale_ratio 0 \
#   --rotation_ratio 0.4 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features rotation


