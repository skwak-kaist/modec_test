#!/bin/bash 
# s.kwak@etri.re.kr 

eval "$(conda shell.bash hook)"
conda activate modecgs

read -p "Enter the GPU ID (default: 0): " GPU_id
if [ -z "$GPU_id" ]; then
	GPU_id=0
fi

read -p "Enter the port (default: 5000): " port
if [ -z "$port" ]; then
	port=5000
fi

read -p "Enter the config number (default: 1.0): " config_number
if [ -z "$config_number" ]; then
	config_number=1.0
fi

########## set parameters ##########
dataset=hypernerf
dataset_config=hypernerf_half0

colmap=0
down_sample=0
train=1
render=1
eval=1
#####################################

source dataset_config/${dataset_config}.config
cd ..

# config=config_${config_number}
base_config=config_${config_number}
output_path=${dataset}_${config_number}

# rename the config as base.py
rm arguments/${dataset}/base.py # remove the existing base.py
cp arguments/${dataset}/${base_config}.py arguments/${dataset}/base.py # copy the new base.py


if [ ! -d "output/${output_path}" ]; then
	mkdir output/${output_path}
fi

echo "training time" > "output/${output_path}/training_time.txt"
echo "training log" > "output/${output_path}/training_log.txt"

for scene in $scenes; do

	scene_path=$(echo $scnen_paths | cut -d' ' -f$((idx+1)))

	echo "########################################"
	echo "dataset" $dataset
	echo "scene: "$scene
	echo "scene path: "$scene_path
	echo "GPU" $GPU_id
	
	if [ -f arguments/${dataset}/${scene}.py ]; then
		echo "Using scene config: $scene.py with base config $base_config"
		config=${scene}
	else
		echo "Using base config: $base_config.py"
		config=${base_config}
	fi
	echo "########################################"


	if [ $colmap == 1 ]
	then
		echo "Running COLMAP"
		bash colmap.sh data/${dataset}/${scene_path} ${dataset}
	fi
	
	if [ $down_sample == 1 ]
	then
		echo "Downsampling the point cloud"
		PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python scripts/downsample_point.py data/${dataset}/${scene_path}/colmap/dense/workspace/fused.ply data/${dataset}/${scene_path}/points3D_downsample2.ply
	else
		echo "Skip downsampling"
	fi

	
	if [ $train == 1 ]
	then
		start_time=$(date '+%s')
		echo "Training the model"
		PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python train.py -s data/${dataset}/${scene_path} --port ${port} --expname "${output_path}/${scene}" --configs arguments/${dataset}/${config}.py >> "output/${output_path}/training_log.txt"
		end_time=$(date '+%s')
		diff=$((end_time - start_time))
		hour=$((diff / 3600 % 24))
		echo "Training time: $(($diff / 3600)) hours, $(($diff % 3600 / 60)) minutes, $(($diff % 60)) seconds" >> "output/${output_path}/training_time.txt"
	else
		echo "Skip training"
	fi

	if [ $render == 1 ]
	then
	export OMP_NUM_THREADS=1
		echo "Rendering the model"
		PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python render.py --model_path "output/${output_path}/${scene}" --skip_train --configs arguments/${dataset}/${config}.py >> "output/${output_path}/training_log.txt"
	else
		echo "Skip rendering"
	fi

	if [ $eval == 1 ]
	then
		echo "Evaluating the model"
		PYTHONPATH='.' CUDA_VISIBLE_DEVICES=$GPU_id python metrics.py --model_path "output/${output_path}/${scene}" >> "output/${output_path}/training_log.txt"
	else
		echo "Skip evaluation"
	fi

	# idx +1
	idx=$(($idx + 1))

done

python collect_metric.py --output_path "output/${output_path}" --dataset ${dataset}


