#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/path/to/your/project

# Python 스크립트 실행

# python merge_pointcept_with_3dgs_attributes.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_opacity-rotation-agg-mean \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --split train \
#     --enable_pointcept_distance \
#     --k_neighbors 5 \
#     --use_features rotation opacity \
#     --aggregation mean
# python merge_pointcept_with_3dgs_attributes.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_opacity-rotation-agg-mean \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --split val \
#     --enable_pointcept_distance \
#     --k_neighbors 5 \
#     --use_features rotation opacity \
#     --aggregation mean

# python merge_pointcept_with_3dgs_attributes.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_opacity-scale-rotation-agg-mean \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
#     --split train \
#     --enable_pointcept_distance \
#     --k_neighbors 5 \
#     --use_features scale rotation opacity \
#     --aggregation mean
python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_opacity-scale-rotation-agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split val \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features scale rotation opacity \
    --aggregation mean

python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_scale-rotation-agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split train \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features scale rotation\
    --aggregation mean
python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_scale-rotation-agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split val \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features scale rotation\
    --aggregation mean

python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_scale_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split train \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features scale \
    --aggregation mean
python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_scale_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split val \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features scale \
    --aggregation mean

python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_rotation_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split train \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features rotation \
    --aggregation mean
python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_rotation_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split val \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features rotation \
    --aggregation mean

python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_opacity_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split train \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features opacity \
    --aggregation mean
python merge_pointcept_with_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_3dgs-attr_pdis0001_opacity_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split val \
    --enable_pointcept_distance \
    --k_neighbors 5 \
    --use_features opacity \
    --aggregation mean
