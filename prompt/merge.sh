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
  --output_root /home/knuvi/Desktop/song/Pointcept/data/vox004_opacity \
  --data_type samples100 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0 0 0 \
  --use_features opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/vox002_opacity \
  --data_type samples100 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0 0 0 \
  --use_features opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/vox006_opacity \
  --data_type samples100 \
  --voxel_size 0.06 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0 0 0 \
  --use_features opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/vox004_scale02_opacity \
  --data_type samples100 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0.2 0 0 \
  --use_features opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/FPS005_opacity \
  --data_type samples100 \
  --voxel_size 0 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0 0 0 \
  --use_features opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/vox004_scale04_opacity \
  --data_type samples100 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0.4 0 0 \
  --use_features opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/vox004_scale06_opacity \
  --data_type samples100 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0.6 0 0 \
  --use_features opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/vox004_scale08_opacity \
  --data_type samples100 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0.8 0 0 \
  --use_features opacity

python PC-3DGS_fusion.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/vox004_scale099_opacity \
  --data_type samples100 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --attr_pruning_ratio 0.99 0 0 \
  --use_features opacity  
# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/FPS01_opacity \
#   --data_type samples100 \
#   --voxel_size 0 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0 \
#   --use_features opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/FPS01_rotation \
#   --data_type samples100 \
#   --voxel_size 0 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0 \
#   --use_features rotation

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/vox002-modified_scale \
#   --data_type samples100 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0 \
#   --use_features scale

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/vox002-modified_opacity \
#   --data_type samples100 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0 \
#   --use_features opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/vox002-modified_rotation \
#   --data_type samples100 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0 \
#   --use_features rotation

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/no-pruning_scale-rotation \
#   --data_type samples100 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0 \
#   --use_features scale rotation

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/no-pruning_rotation-opacity \
#   --data_type samples100 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0 \
#   --use_features rotation opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/no-pruning_scale-opacity \
#   --data_type samples100 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0 \
#   --use_features scale opacity
# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale04-rotation04-opacity04 \
#   --data_type samples100 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0.4 0.4 0.4 \
#   --use_features scale rotation opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale06-rotation06-opacity06 \
#   --data_type samples100 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0.6 0.6 0.6 \
#   --use_features scale rotation opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale03-rotation03-opacity03 \
#   --data_type samples100 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0.3 0.3 0.3 \
#   --use_features scale rotation opacity

# ##

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale02-rotation02-opacity02_vox002 \
#   --data_type samples100 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0.2 0.2 0.2 \
#   --use_features scale rotation opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale04-rotation04-opacity04_vox002 \
#   --data_type samples100 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0.4 0.4 0.4 \
#   --use_features scale rotation opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale06-rotation06-opacity06_vox002 \
#   --data_type samples100 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0.6 0.6 0.6 \
#   --use_features scale rotation opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale03-rotation03-opacity03_vox002 \
#   --data_type samples100 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0.3 0.3 0.3 \
#   --use_features scale rotation opacity
# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-rotation05_ratio06 \-vox002
#   --data_type samples100 \
#   --pruning_ratio 0.6 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_weight 0.5 0.5 0 \
#   --use_features scale rotation

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-opacity05_ratio02 \
#   --data_type samples100 \
#   --pruning_ratio 0.2 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_weight 0.5 0.5 0 \
#   --use_features scale opacity
  
# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-opacity05_ratio04 \
#   --data_type samples100 \
#   --pruning_ratio 0.4 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_weight 0.5 0.5 0 \
#   --use_features scale opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/scale05-opacity05_ratio06 \
#   --data_type samples100 \
#   --pruning_ratio 0.6 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_weight 0.5 0.5 0 \
#   --use_features scale opacity


# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/rotation05-opacity05_ratio02 \
#   --data_type samples100 \
#   --pruning_ratio 0.2 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_weight 0.5 0.5 0 \
#   --use_features rotation opacity
  
# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/rotation05-opacity05_ratio04 \
#   --data_type samples100 \
#   --pruning_ratio 0.4 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_weight 0.5 0.5 0 \
#   --use_features rotation opacity

# python PC-3DGS_fusion.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/rotation05-opacity05_ratio06 \
#   --data_type samples100 \
#   --pruning_ratio 0.6 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_weight 0.5 0.5 0 \
#   --use_features rotation opacity
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


