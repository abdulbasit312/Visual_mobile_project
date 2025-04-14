#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode24
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=training_output_sch3.out
#SBATCH --error=training_output2_sch3.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 main.py --data-path /w/331/abdulbasit/image-net-small/ --batch-size 256 --model graph_propagation_deit_small_patch16_224 --sparsity 1.0 --alpha 0.1 --num_prop 4 --selection MixedAttnMax --propagation GraphProp --output_dir /w/331/navtegh/GTP-ViT/output3
    