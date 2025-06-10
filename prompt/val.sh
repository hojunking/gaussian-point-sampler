
# 실행 환경 설정
export PYTHONPATH=$PYTHONPATH:/home/knuvi/Desktop/song/Pointcept
## use_features scale opacity rotation 필수
# exp : 저장 이름 폴더
# pdistance : 0.0005 권장
# pruning ratio : 0 0 0 (scale opacity rotation 순서)

# python PC-3DGS_fusion_single.py \
#   --scene scene0011_00 \
#   --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10 \
#   --exp pd_vox004_opacity02-v2 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0.2 0 \
#   --use_features scale opacity rotation

# python PC-3DGS_fusion_single.py \
#   --scene scene0011_00 \
#   --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10 \
#   --exp pd_vox004_rotation02-v2 \
#   --voxel_size 0.04 \
#   --pdistance 0.0005 \
#   --attr_pruning_ratio 0 0 0.2 \
#   --use_features scale opacity rotation


python PC-3DGS_fusion_single.py \
  --scene scene0011_00 \
  --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10 \
  --exp xx \ 
  --voxel_size 0 \
  --pdistance 0 \
  --attr_pruning_ratio 0 0 0 \
  --use_features scale opacity rotation
