#!/bin/bash

#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=1-12


configs="--m_model MolCLR --p_model GearNet  |\
--m_model ChemBERTa --p_model GearNet  |\
--m_model MoLFormer --p_model GearNet  |\
--m_model MolCLR --p_model ProtBert  |\
--m_model ChemBERTa --p_model ProtBert  |\
--m_model MoLFormer --p_model ProtBert  |\
--m_model MolCLR --p_model esm3-small  |\
--m_model ChemBERTa --p_model esm3-small  |\
--m_model MoLFormer --p_model esm3-small  |\
--m_model MolCLR --p_model esm3-medium  |\
--m_model ChemBERTa --p_model esm3-medium  |\
--m_model MoLFormer --p_model esm3-medium"
IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

eval "$(conda shell.bash hook)"

conda activate ReactEmbedESM
export PYTHONPATH=$PYTHONPATH:$(pwd)
python contrastive_learning/trainer.py $config --override
python contrastive_learning/trainer.py $config --flip_prob 0.1 --override
python contrastive_learning/trainer.py $config --flip_prob 0.5 --override
python contrastive_learning/trainer.py $config --data_name "pathbank" --override
python contrastive_learning/trainer.py $config --n_layers 2 --epochs 10 --override
python contrastive_learning/trainer.py $config --min_value 1 --override
python contrastive_learning/trainer.py $config --min_value 10 --override
