#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode20
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=training_output_m_sch.out
#SBATCH --error=training_output2_m_sch.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin
MODEL=rcvit_m # model to evaluate: rcvit_{xs, s, m, t}
# command below for evaluation and resuming from checkpoint
# srun python main.py --model ${MODEL} --eval True --resume /w/331/stutiwadhwa/Visual_mobile_project/classification/class_output_200classes/checkpoint-best_224.pth --input_size 224 --data_path /w/331/abdulbasit/image-net-small/new_train --eval_data_path /w/331/abdulbasit/image-net-small/new_val

# commands below for pretraining different sized models
# srun python3 main.py --data_path /w/331/abdulbasit/image-net-small/new_train --eval_data_path /w/331/abdulbasit/image-net-small/new_val --output_dir /w/331/stutiwadhwa/Visual_mobile_project/classification/class_output_200classes_xs
srun python main.py --model ${MODEL} --resume /w/331/stutiwadhwa/Visual_mobile_project/classification/class_output_200classes_m/checkpoint-276.pth --data_path /w/331/abdulbasit/image-net-small/new_train --eval_data_path /w/331/abdulbasit/image-net-small/new_val --output_dir /w/331/stutiwadhwa/Visual_mobile_project/classification/class_output_200classes_m

