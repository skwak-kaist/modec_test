

workdir=$1
datatype=$2 # blender, hypernerf, llff, nvidia, dycheck
export CUDA_VISIBLE_DEVICES=0

#$workdir/rgb/1x 에 있는 png 이미지 전체를 가로 1/2, 세로 1/2로 줄여서 $workdir/rgb/2x에 저장
python scripts/downscale.py --input_dir $workdir/rgb




rm -rf $workdir/sparse_
rm -rf $workdir/image_colmap
python scripts/"$datatype"2colmap.py $workdir
rm -rf $workdir/colmap
rm -rf $workdir/colmap/sparse/0

mkdir $workdir/colmap
cp -r $workdir/image_colmap $workdir/colmap/images
cp -r $workdir/sparse_ $workdir/colmap/sparse_custom
colmap feature_extractor --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
python database.py --database_path $workdir/colmap/database.db --txt_path $workdir/colmap/sparse_custom/cameras.txt
colmap exhaustive_matcher --database_path $workdir/colmap/database.db
mkdir -p $workdir/colmap/sparse/0

colmap point_triangulator --database_path $workdir/colmap/database.db --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse_custom --output_path $workdir/colmap/sparse/0 --clear_points 1

mkdir -p $workdir/colmap/dense/workspace
colmap image_undistorter --image_path $workdir/colmap/images --input_path $workdir/colmap/sparse/0 --output_path $workdir/colmap/dense/workspace
colmap patch_match_stereo --workspace_path $workdir/colmap/dense/workspace
colmap stereo_fusion --workspace_path $workdir/colmap/dense/workspace --output_path $workdir/colmap/dense/workspace/fused.ply
