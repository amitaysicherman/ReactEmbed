#!/bin/bash

#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A4000:1
#SBATCH --array=1-12


configs="--flip_prob 0.1  |\
--flip_prob 0.5  |\
--samples_ratio 0.5  |\
--samples_ratio 0.1  |\
--data_name pathbank  |\
--no_pp_mm 1 "

IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

eval "$(conda shell.bash hook)"

conda activate ReactEmbedESM
export PYTHONPATH=$PYTHONPATH:$(pwd)
python contrastive_learning/trainer.py $config --override