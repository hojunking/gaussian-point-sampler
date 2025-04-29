#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept

# attr_weight scale rotation opacity

python 3DGS-only.py \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/3DGS-only \
  --data_type samples100 \
  --voxel_size 0 \

# python 3DGS-only.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/3DGS-only-vox002 \
#   --data_type samples100 \
#   --voxel_size 0.02 \
# python 3DGS-only.py \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/3DGS-only-vox004 \
#   --data_type samples100 \
#   --voxel_size 0.04 \
