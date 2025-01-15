#!/bin/bash

#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A4000:1
#SBATCH --array=1-71


configs=" --p_model ProtBert --m_model ChemBERTa --no_pp_mm 1|\
 --p_model ProtBert --m_model ChemBERTa --data_name pathbank |\
 --p_model ProtBert --m_model ChemBERTa  --samples_ratio 0.1 |\
 --p_model ProtBert --m_model ChemBERTa --samples_ratio 0.5 |\
 --p_model ProtBert --m_model ChemBERTa --flip_prob 0.5 |\
 --p_model ProtBert --m_model ChemBERTa  --flip_prob 0.1 |\
 --p_model ProtBert --m_model MoLFormer --no_pp_mm 1|\
 --p_model ProtBert --m_model MoLFormer --data_name pathbank |\
 --p_model ProtBert --m_model MoLFormer  --samples_ratio 0.1 |\
 --p_model ProtBert --m_model MoLFormer --samples_ratio 0.5 |\
 --p_model ProtBert --m_model MoLFormer --flip_prob 0.5 |\
 --p_model ProtBert --m_model MoLFormer  --flip_prob 0.1 |\
 --p_model ProtBert --m_model MolCLR --no_pp_mm 1|\
 --p_model ProtBert --m_model MolCLR --data_name pathbank |\
 --p_model ProtBert --m_model MolCLR  --samples_ratio 0.1 |\
 --p_model ProtBert --m_model MolCLR --samples_ratio 0.5 |\
 --p_model ProtBert --m_model MolCLR --flip_prob 0.5 |\
 --p_model ProtBert --m_model MolCLR  --flip_prob 0.1 |\
 --p_model esm3-small --m_model ChemBERTa --no_pp_mm 1|\
 --p_model esm3-small --m_model ChemBERTa --data_name pathbank |\
 --p_model esm3-small --m_model ChemBERTa  --samples_ratio 0.1 |\
 --p_model esm3-small --m_model ChemBERTa --samples_ratio 0.5 |\
 --p_model esm3-small --m_model ChemBERTa --flip_prob 0.5 |\
 --p_model esm3-small --m_model ChemBERTa  --flip_prob 0.1 |\
 --p_model esm3-small --m_model MoLFormer --no_pp_mm 1|\
 --p_model esm3-small --m_model MoLFormer --data_name pathbank |\
 --p_model esm3-small --m_model MoLFormer  --samples_ratio 0.1 |\
 --p_model esm3-small --m_model MoLFormer --samples_ratio 0.5 |\
 --p_model esm3-small --m_model MoLFormer --flip_prob 0.5 |\
 --p_model esm3-small --m_model MoLFormer  --flip_prob 0.1 |\
 --p_model esm3-small --m_model MolCLR --no_pp_mm 1|\
 --p_model esm3-small --m_model MolCLR --data_name pathbank |\
 --p_model esm3-small --m_model MolCLR  --samples_ratio 0.1 |\
 --p_model esm3-small --m_model MolCLR --samples_ratio 0.5 |\
 --p_model esm3-small --m_model MolCLR --flip_prob 0.5 |\
 --p_model esm3-small --m_model MolCLR  --flip_prob 0.1 |\
 --p_model esm3-medium --m_model ChemBERTa --no_pp_mm 1|\
 --p_model esm3-medium --m_model ChemBERTa --data_name pathbank |\
 --p_model esm3-medium --m_model ChemBERTa  --samples_ratio 0.1 |\
 --p_model esm3-medium --m_model ChemBERTa --samples_ratio 0.5 |\
 --p_model esm3-medium --m_model ChemBERTa --flip_prob 0.5 |\
 --p_model esm3-medium --m_model ChemBERTa  --flip_prob 0.1 |\
 --p_model esm3-medium --m_model MoLFormer --no_pp_mm 1|\
 --p_model esm3-medium --m_model MoLFormer --data_name pathbank |\
 --p_model esm3-medium --m_model MoLFormer  --samples_ratio 0.1 |\
 --p_model esm3-medium --m_model MoLFormer --samples_ratio 0.5 |\
 --p_model esm3-medium --m_model MoLFormer --flip_prob 0.5 |\
 --p_model esm3-medium --m_model MoLFormer  --flip_prob 0.1 |\
 --p_model esm3-medium --m_model MolCLR --no_pp_mm 1|\
 --p_model esm3-medium --m_model MolCLR --data_name pathbank |\
 --p_model esm3-medium --m_model MolCLR  --samples_ratio 0.1 |\
 --p_model esm3-medium --m_model MolCLR --samples_ratio 0.5 |\
 --p_model esm3-medium --m_model MolCLR --flip_prob 0.5 |\
 --p_model esm3-medium --m_model MolCLR  --flip_prob 0.1 |\
 --p_model GearNet --m_model ChemBERTa --no_pp_mm 1|\
 --p_model GearNet --m_model ChemBERTa --data_name pathbank |\
 --p_model GearNet --m_model ChemBERTa  --samples_ratio 0.1 |\
 --p_model GearNet --m_model ChemBERTa --samples_ratio 0.5 |\
 --p_model GearNet --m_model ChemBERTa --flip_prob 0.5 |\
 --p_model GearNet --m_model ChemBERTa  --flip_prob 0.1 |\
 --p_model GearNet --m_model MoLFormer --no_pp_mm 1|\
 --p_model GearNet --m_model MoLFormer --data_name pathbank |\
 --p_model GearNet --m_model MoLFormer  --samples_ratio 0.1 |\
 --p_model GearNet --m_model MoLFormer --samples_ratio 0.5 |\
 --p_model GearNet --m_model MoLFormer --flip_prob 0.5 |\
 --p_model GearNet --m_model MoLFormer  --flip_prob 0.1 |\
 --p_model GearNet --m_model MolCLR --no_pp_mm 1|\
 --p_model GearNet --m_model MolCLR --data_name pathbank |\
 --p_model GearNet --m_model MolCLR  --samples_ratio 0.1 |\
 --p_model GearNet --m_model MolCLR --samples_ratio 0.5 |\
 --p_model GearNet --m_model MolCLR --flip_prob 0.5 |\
 --p_model GearNet --m_model MolCLR  --flip_prob 0.1"

IFS='|' read -ra config_array <<< "$configs"
cmd=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

eval "$(conda shell.bash hook)"

conda activate ReactEmbedESM
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo $cmd
python contrastive_learning/trainer.py $cmd


