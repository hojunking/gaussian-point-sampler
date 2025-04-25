#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features opacity

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_scale-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features scale opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_scale-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features scale opacity

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_scale-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_scale-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale opacity

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_vox004_scale-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features scale opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_vox004_scale-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features scale opacity

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_vox002_scale-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.001 \
#   --use_features scale opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_vox002_scale-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.001 \
#   --use_features scale opacity

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_rotation-norm-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features rotation opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_rotation-norm-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features rotation opacity

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_rotation-norm-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features rotation opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_rotation-norm-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features rotation opacity

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_vox004_rotation-norm-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features rotation opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_vox004_rotation-norm-opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features rotation opacity


# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_vox004_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_vox004_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features opacity


# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale02_vox004_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale02_vox004_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features opacity

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac04_vox002_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac04_vox002_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac04_vox002_rotation-norm \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features rotation
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac04_vox002_rotation-norm \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac04_vox002_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac04_vox002_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.4 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features opacity

########################################################################################################

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale

python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_rotation-norm \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split train \
  --k_neighbors 5 \
  --scale_ratio 0 \
  --opacity_ratio 0 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation
python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_rotation-norm \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split val \
  --k_neighbors 5 \
  --scale_ratio 0 \
  --opacity_ratio 0 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox002_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features opacity
###################################################################################################
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features scale
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0 \
#   --opacity_ratio 0 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --use_features scale

python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_rotation-norm \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split train \
  --k_neighbors 5 \
  --scale_ratio 0 \
  --opacity_ratio 0 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --use_features rotation
python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_vox004_rotation-norm \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split val \
  --k_neighbors 5 \
  --scale_ratio 0 \
  --opacity_ratio 0 \
  --voxel_size 0.04 \
  --pdistance 0.0005 \
  --use_features rotation



####################################################################################################
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale02_opac02_vox002_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale02_opac02_vox002_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale

python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale02_opac02_vox002_rotation-norm \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split train \
  --k_neighbors 5 \
  --scale_ratio 0.2 \
  --opacity_ratio 0.2 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation
python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale02_opac02_vox002_rotation-norm \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split val \
  --k_neighbors 5 \
  --scale_ratio 0.2 \
  --opacity_ratio 0.2 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale02_opac02_vox002_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale02_opac02_vox002_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features opacity


####################################################################################################
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac02_vox002_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac02_vox002_scale \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features scale

python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac02_vox002_rotation-norm \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split train \
  --k_neighbors 5 \
  --scale_ratio 0.4 \
  --opacity_ratio 0.2 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation
python PC-3DGS_fusion.py \
  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
  --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac02_vox002_rotation-norm \
  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
  --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
  --split val \
  --k_neighbors 5 \
  --scale_ratio 0.4 \
  --opacity_ratio 0.2 \
  --voxel_size 0.02 \
  --pdistance 0.0005 \
  --use_features rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac02_vox002_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features opacity
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd00005_scale04_opac02_vox002_opacity \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.4 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --use_features opacity


########################################################################################################
##### 3DGS-attr-all-samples (0420)
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_scale05-opa02_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.5 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0001_scale05-opa02_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.5 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.04 \
#   --pdistance 0.001 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0005_scale02-opa02_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.04 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0005_scale02-opa02_vox004_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.04 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0005_scale05-opa02_vox002_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.5 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0005_scale05-opa02_vox002_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.5 \
#   --opacity_ratio 0.2 \
#   --voxel_size 0.02 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0005_scale02-opa05_vox002_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.5 \
#   --voxel_size 0.02 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/pd0005_scale02-opa05_vox002_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.5 \
#   --voxel_size 0.02 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation



# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd00005_scale02_opa07_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.7 \
#   --voxel_size 0.02 \
#   --pdistance 0.0005 \
#   --num_workers 4 \
#   --use_features scale opacity rotation

  # Python 스크립트 실행
# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0001_scale02_opa03_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.3 \
#   --voxel_size 0.02 \
#   --pdistance 0.001 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0001_scale02_opa03_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.2 \
#   --opacity_ratio 0.3 \
#   --voxel_size 0.02 \
#   --pdistance 0.001 \
#   --use_features scale opacity rotation

#   python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale03_opa03_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split train \
#   --k_neighbors 5 \
#   --scale_ratio 0.3 \
#   --opacity_ratio 0.3 \
#   --voxel_size 0.02 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion.py \
#   --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#   --output_root /home/knuvi/Desktop/song/Pointcept/data/fusion_pd0005_scale03_opa03_all \
#   --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#   --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#   --split val \
#   --k_neighbors 5 \
#   --scale_ratio 0.3 \
#   --opacity_ratio 0.3 \
#   --voxel_size 0.02 \
#   --pdistance 0.005 \
#   --use_features scale opacity rotation
