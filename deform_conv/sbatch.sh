#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode29
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2025-Apr-19/abdulbasit/xs.out
#SBATCH --error=/scratch/expires-2025-Apr-19/abdulbasit/xs.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 main.py --data_path /w/331/abdulbasit/image-net-small/new_train --eval_data_path /w/331/abdulbasit/image-net-small/new_val --output_dir /scratch/expires-2025-Apr-19/abdulbasit/deform_conv --resume /scratch/expires-2025-Apr-10/abdulbasit/deform_conv/checkpoint-299.pth --epochs 500