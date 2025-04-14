#!/bin/bash

# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept

# Python 스크립트 실행
python merge_PC_3dgs_attributes.py \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/Pointcept/data/merge_PC-3dgs_pdis00005_scale-rotation \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --meta_root /home/knuvi/Desktop/song/gaussian-point-sampler/meta \
    --split train \
    --scale_ratio 0.5 \
    --opacity_ratio 0 \
    --enable_pointcept_distance \
    --pdistance 0.0005 \
    --k_neighbors 5 \
    --use_features scale rotation \
    --aggregation mean