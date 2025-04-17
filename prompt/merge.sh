#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept

# Python 스크립트 실행
python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale02-opa05_vox004_all \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split train \
  --k_neighbors 5 \
  --scale_ratio 0.2 \
  --opacity_ratio 0.5 \
  --voxel_size 0.04 \
  --pdistance 0.005 \
  --use_features scale opacity rotation

python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale02-opa05_vox004_all \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split val \
  --k_neighbors 5 \
  --scale_ratio 0.2 \
  --opacity_ratio 0.5 \
  --voxel_size 0.04 \
  --pdistance 0.005 \
  --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0001_scale02-opa05_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.5 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features scale opacity rotation
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0001_scale02-opa05_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.5 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale02-opa05_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.5 \
#   --voxel_size 0.04 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale02-opa05_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.5 \
#   --voxel_size 0.04 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale02-opa07_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.7 \
#   --voxel_size 0.04 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale02-opa07_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.7 \
#   --voxel_size 0.04 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd001_scale05-opa07_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.5 \
#   --opacity_ratio 0.7 \
#   --voxel_size 0.04 \
#   --pdistance 0.01 \
#   --use_features scale opacity rotation
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd001_scale05-opa07_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.5 \
#   --opacity_ratio 0.7 \
#   --voxel_size 0.04 \
#   --pdistance 0.01 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd00005_scale02-opa05_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.5 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features scale opacity rotation
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd00005_scale02-opa05_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.5 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale02-opa04_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.04 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale02-opa04_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.04 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0001_scale02-opa04_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features scale opacity rotation
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0001_scale02-opa04_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features scale opacity rotation