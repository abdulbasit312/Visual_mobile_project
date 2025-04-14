#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode26
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2025-Apr-19/abdulbasit/training_output_small.out
#SBATCH --error=/scratch/expires-2025-Apr-19/abdulbasit/training_output_small.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 main.py --data_path /w/331/abdulbasit/image-net-small/new_train --eval_data_path /w/331/abdulbasit/image-net-small/new_val --output_dir /scratch/expires-2025-Apr-19/abdulbasit/deform_s --model rcvit_s --resume /scratch/expires-2025-Apr-19/abdulbasit/deform_s/checkpoint-220.pth --lr 0.0001703345968059372