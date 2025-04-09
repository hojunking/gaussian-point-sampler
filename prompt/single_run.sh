# python single_sample_3dgs_test.py \
#     --scene scene0011_00 \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --scale_ratio 0 \
#     --opacity_ratio 0.5 \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --exp opacity50

python single_sample_3dgs_test.py \
    --scene scene0011_00 \
    --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
    --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test/test_samples10 \
    --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
    --scale_ratio 0 \
    --opacity_ratio 0 \
    --k_neighbors 5 \
    --use_label_consistency \
    --enable_pointcept_distance \
    --exp pdistance0001 \

# python single_sample_3dgs_test.py \
#     --scene scene0011_00 \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test_samples10 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --scale_ratio 0 \
#     --opacity_ratio 0 \
#     --enable_density \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
#     --exp density_eps005_mpoint50


    #--enable_normal \

# python single_sample_3dgs_test.py \
#     --scene scene0011_00 \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test_samples10 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --scale_ratio 0 \
#     --opacity_ratio 0 \
#     --enable_normal \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --enable_pointcept_distance \
#     --exp normal_kneighbors5_cos09_pdistance
    
    #--enable_density \

# python single_sample_3dgs_test.py \
#     --scene scene0011_00 \
#     --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
#     --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test_samples10 \
#     --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
#     --scale_ratio 0 \
#     --opacity_ratio 0 \
#     --enable_sor \
#     --k_neighbors 5 \
#     --use_label_consistency \
#     --exp sor_kneighbors20_std2
#     #--enable_normal \
#     #--enable_pointcept_distance \
#     #--enable_density \


   #  python single_sample_3dgs_test.py \
   #  --scene scene0011_00 \
   #  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
   #  --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test_samples10 \
   #  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
   #  --scale_ratio 0 \
   #  --opacity_ratio 0 \
   #  --enable_sor \
   #  --k_neighbors 5 \
   #  --use_label_consistency \
   #  --exp sor_kneighbors30_std1
    #--enable_normal \
    #--enable_pointcept_distance \
    #--enable_density \

   #    python single_sample_3dgs_test.py \
   #  --scene scene0011_00 \
   #  --input_root /home/knuvi/Desktop/song/Pointcept/data/scannet \
   #  --output_root /home/knuvi/Desktop/song/gaussian-point-sampler/test_samples10 \
   #  --path_3dgs_root /home/knuvi/Desktop/song/data/3dgs_scans/3dgs_output \
   #  --scale_ratio 0.1 \
   #  --opacity_ratio 0 \
   #  --k_neighbors 5 \
   #    --enable_sor \
   #  --use_label_consistency \
   #  --exp scale01_sor_kneighbors40_std01