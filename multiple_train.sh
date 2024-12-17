#!/bin/sh
#SBATCH --time=1-00
#SBATCH --mem=16G
#SBATCH --requeue
#SBATCH -c 1
#SBATCH --array=1-27

configs="--m_model ChemBERTa --p_model ProtBert --batch_size 8192 |\
--m_model ChemBERTa --p_model ProtBert --batch_size 2048 |\
--m_model ChemBERTa --p_model ProtBert --batch_size 128 |\
--m_model ChemBERTa --p_model esm3-small --batch_size 8192 |\
--m_model ChemBERTa --p_model esm3-small --batch_size 2048 |\
--m_model ChemBERTa --p_model esm3-small --batch_size 128 |\
--m_model ChemBERTa --p_model esm3-medium --batch_size 8192 |\
--m_model ChemBERTa --p_model esm3-medium --batch_size 2048 |\
--m_model ChemBERTa --p_model esm3-medium --batch_size 128 |\
--m_model MoLFormer --p_model ProtBert --batch_size 8192 |\
--m_model MoLFormer --p_model ProtBert --batch_size 2048 |\
--m_model MoLFormer --p_model ProtBert --batch_size 128 |\
--m_model MoLFormer --p_model esm3-small --batch_size 8192 |\
--m_model MoLFormer --p_model esm3-small --batch_size 2048 |\
--m_model MoLFormer --p_model esm3-small --batch_size 128 |\
--m_model MoLFormer --p_model esm3-medium --batch_size 8192 |\
--m_model MoLFormer --p_model esm3-medium --batch_size 2048 |\
--m_model MoLFormer --p_model esm3-medium --batch_size 128"
IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python multiple_train.py $config
