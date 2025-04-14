#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode24
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=training_output_ptsq_t_sch.out
#SBATCH --error=training_output2_ptsq_t_sch.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin
MODEL=rcvit_t # model to evaluate: rcvit_{xs, s, m, t}

srun python3 post_training_static_quantization.py --model ${MODEL} --data_path /w/331/abdulbasit/image-net-small/new_train --resume /w/331/stutiwadhwa/Visual_mobile_project/classification/class_output_200classes_t/checkpoint-best_224.pth --input_size 224 --eval_data_path /w/331/abdulbasit/image-net-small/new_val --output_dir /w/331/stutiwadhwa/Visual_mobile_project/classification/ptsq --eval True