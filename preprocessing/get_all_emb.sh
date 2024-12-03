#!/bin/bash
#SBATCH --time=1-00
#SBATCH --array=1-6
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --requeue
NAMES=("ProtBert" "ChemBERTa" "MoLFormer" "esm3-small" "esm3-medium" "esm3-large")


NAME=${NAMES[$SLURM_ARRAY_TASK_ID - 1]}
python preprocessing/seq_to_vec.py --model $NAME