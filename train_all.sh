#!/bin/bash

#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=1-23


configs="--m_model MolCLR --p_model GearNet --n_layers 1 --epochs 1 |\
--m_model MolCLR --p_model GearNet --n_layers 2 --epochs 10 |\
--m_model ChemBERTa --p_model GearNet --n_layers 1 --epochs 1 |\
--m_model ChemBERTa --p_model GearNet --n_layers 2 --epochs 10 |\
--m_model MoLFormer --p_model GearNet --n_layers 1 --epochs 1 |\
--m_model MoLFormer --p_model GearNet --n_layers 2 --epochs 10 |\
--m_model MolCLR --p_model ProtBert --n_layers 1 --epochs 1 |\
--m_model MolCLR --p_model ProtBert --n_layers 2 --epochs 10 |\
--m_model ChemBERTa --p_model ProtBert --n_layers 1 --epochs 1 |\
--m_model ChemBERTa --p_model ProtBert --n_layers 2 --epochs 10 |\
--m_model MoLFormer --p_model ProtBert --n_layers 1 --epochs 1 |\
--m_model MoLFormer --p_model ProtBert --n_layers 2 --epochs 10 |\
--m_model MolCLR --p_model esm3-small --n_layers 1 --epochs 1 |\
--m_model MolCLR --p_model esm3-small --n_layers 2 --epochs 10 |\
--m_model ChemBERTa --p_model esm3-small --n_layers 1 --epochs 1 |\
--m_model ChemBERTa --p_model esm3-small --n_layers 2 --epochs 10 |\
--m_model MoLFormer --p_model esm3-small --n_layers 1 --epochs 1 |\
--m_model MoLFormer --p_model esm3-small --n_layers 2 --epochs 10 |\
--m_model MolCLR --p_model esm3-medium --n_layers 1 --epochs 1 |\
--m_model MolCLR --p_model esm3-medium --n_layers 2 --epochs 10 |\
--m_model ChemBERTa --p_model esm3-medium --n_layers 1 --epochs 1 |\
--m_model ChemBERTa --p_model esm3-medium --n_layers 2 --epochs 10 |\
--m_model MoLFormer --p_model esm3-medium --n_layers 1 --epochs 1 |\
--m_model MoLFormer --p_model esm3-medium --n_layers 2 --epochs 10"

IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

eval "$(conda shell.bash hook)"

conda activate ReactEmbedESM
export PYTHONPATH=$PYTHONPATH:$(pwd)
#python contrastive_learning/trainer.py $config --data_name "pathbank"
#python contrastive_learning/trainer.py $config --data_name "pathbank" --flip_prob 0.1
#python contrastive_learning/trainer.py $config --data_name "pathbank" --flip_prob 0.5
#python contrastive_learning/trainer.py $config
#python contrastive_learning/trainer.py $config --flip_prob 0.1
#python contrastive_learning/trainer.py $config --flip_prob 0.5
python contrastive_learning/trainer.py $config --data_name "pathbank" --min_value 5
python contrastive_learning/trainer.py $config --data_name "pathbank" --flip_prob 0.1 --min_value 5
python contrastive_learning/trainer.py $config --data_name "pathbank" --flip_prob 0.5 --min_value 5
python contrastive_learning/trainer.py $config --min_value 5
python contrastive_learning/trainer.py $config --flip_prob 0.1 --min_value 5
python contrastive_learning/trainer.py $config --flip_prob 0.5 --min_value 5