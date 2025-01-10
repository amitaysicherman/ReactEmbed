#!/bin/bash
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-12


configs="--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/GearNet-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/GearNet-MolCLR-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-512-0.0-10-0.0001-256-0.0-256 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-1-512-0.0-10-0.0001-256-0.0-256"

IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

tasks_configs="--task_name BACE --bs 2048 --metric auc |\
--task_name BBBP --bs 2048 --metric auc |\
--task_name CEP --bs 2048 --metric rmse |\
--task_name ClinTox --bs 2048 --metric auc |\
--task_name Delaney --bs 2048 --metric rmse |\
--task_name FreeSolv --bs 2048 --metric rmse |\
--task_name HIV --bs 2048 --metric auc |\
--task_name Lipophilicity --bs 2048 --metric rmse |\
--task_name Malaria --bs 2048 --metric rmse |\
--task_name SIDER --bs 2048 --metric auc |\
--task_name Tox21 --bs 2048 --metric auc |\
--task_name BetaLactamase --bs 2048 --metric rmse |\
--task_name Fluorescence --bs 2048 --metric rmse |\
--task_name Stability --bs 2048 --metric rmse |\
--task_name Solubility --bs 2048 --metric rmse |\
--task_name BinaryLocalization --bs 2048 --metric auc |\
--task_name SubcellularLocalization --bs 2048 --metric auc |\
--task_name EnzymeCommission --bs 2048 --metric auc |\
--task_name GeneOntologyMF --bs 2048 --metric auc |\
--task_name GeneOntologyBP --bs 2048 --metric auc |\
--task_name GeneOntologyCC --bs 2048 --metric auc |\
--task_name HumanPPI  --bs 2048 --metric auc |\
--task_name YeastPPI --bs 2048 --metric auc |\
--task_name PPIAffinity --bs 2048 --metric rmse |\
--task_name BindingDB --bs 2048 --metric rmse |\
--task_name PDBBind --bs 2048 --metric rmse |\
--task_name DrugBank --bs 2048 --metric auc |\
--task_name Davis --bs 2048 --metric auc"


IFS='|' read -ra task_array <<< "$tasks_configs"



eval "$(conda shell.bash hook)"
conda activate retd
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo eval_tasks/trainer.py $config $task


for task in "${task_array[@]}"
do
    echo $task
    python eval_tasks/trainer.py $config $task --use_fuse 1 --use_model 1
    python eval_tasks/trainer.py $config $task --use_fuse 0 --use_model 1
    python eval_tasks/trainer.py $config $task --use_fuse 1 --use_model 0
done
