
# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept


# python 3DGS-only_ply-save.py \
#   --scene scene0011_00 \
#   --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10 \
#   --exp 3dgs_only \
#   --voxel_size 0

python 3DGS-only_ply-save.py \
  --scene scene0011_00 \
  --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10 \
  --exp only_3dgs \
  --voxel_size 0 \