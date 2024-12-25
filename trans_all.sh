#!/bin/bash
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=1-4

configs="--metric f1_max |\
  --metric auprc |\
  --metric acc |\
  --metric auc"


models="--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-2-256-0.3-10-5e-05-256-0.0/  |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-2-256-0.3-10-5e-05-256-0.0/  |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-2-256-0.3-10-5e-05-256-0.0/  |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-2-256-0.3-10-5e-05-256-0.0/  |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-2-256-0.3-10-5e-05-256-0.0/  |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-2-256-0.3-10-5e-05-256-0.0/  |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-1-256-0.3-1-5e-05-256-0.0/  |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-2-256-0.3-10-5e-05-256-0.0/  |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-2-256-0.3-10-5e-05-256-0.0/  |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-2-256-0.3-10-5e-05-256-0.0/"
IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

IFS='|' read -ra models_array <<< "$models"

eval "$(conda shell.bash hook)"
conda activate ReactEmbedTorchDrug
export PYTHONPATH=$PYTHONPATH:$(pwd)
for model in "${models_array[@]}"
do
    python transferrin/train_mol_pred_prot.py $config $model
done


