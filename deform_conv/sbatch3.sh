#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode21
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2025-Apr-19/abdulbasit/training_output_m.out
#SBATCH --error=/scratch/expires-2025-Apr-19/abdulbasit/training_output_m.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 main.py --data_path /w/331/abdulbasit/image-net-small/new_train --eval_data_path /w/331/abdulbasit/image-net-small/new_val --model rcvit_m --output_dir /scratch/expires-2025-Apr-19/abdulbasit/deform_m --update_freq 4 --batch_size 8 --resume /scratch/expires-2025-Apr-10/abdulbasit/deform_conv_amp/checkpoint-105.pth