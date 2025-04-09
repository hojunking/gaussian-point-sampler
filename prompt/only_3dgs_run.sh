#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/path/to/your/project

# Python 스크립트 실행
# python process_3dgs_pruning.py \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --split val \
#     --visualize \
#     --scene scene0011_00 \
#     --exp 3dgs_pdistance0001_pruned \
#     --k_neighbors 5 \
#     --ignore_threshold 0.6

python process_3dgs_pruning.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/3dgs_prune-pditance0001_scale_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split train \
    --k_neighbors 5 \
    --use_label_consistency True \
    --ignore_threshold 0.6 \
    --pdistance_max 0.001 \
    --use_features scale \
    --aggregation mean
python process_3dgs_pruning.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/3dgs_prune-pditance0001_scale_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split val \
    --k_neighbors 5 \
    --use_label_consistency True \
    --ignore_threshold 0.6 \
    --pdistance_max 0.001 \
    --use_features scale \
    --aggregation mean


python process_3dgs_pruning.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/3dgs_prune-pditance0001_rotation_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split train \
    --k_neighbors 5 \
    --use_label_consistency True \
    --ignore_threshold 0.6 \
    --pdistance_max 0.001 \
    --use_features rotation \
    --aggregation mean
python process_3dgs_pruning.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/3dgs_prune-pditance0001_rotation_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split val \
    --k_neighbors 5 \
    --use_label_consistency True \
    --ignore_threshold 0.6 \
    --pdistance_max 0.001 \
    --use_features rotation \
    --aggregation mean


python process_3dgs_pruning.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/3dgs_prune-pditance0001_opacity_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split train \
    --k_neighbors 5 \
    --use_label_consistency True \
    --ignore_threshold 0.6 \
    --pdistance_max 0.001 \
    --use_features opacity \
    --aggregation mean
python process_3dgs_pruning.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/3dgs_prune-pditance0001_opacity_agg-mean \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split val \
    --k_neighbors 5 \
    --use_label_consistency True \
    --ignore_threshold 0.6 \
    --pdistance_max 0.001 \
    --use_features opacity \
    --aggregation mean