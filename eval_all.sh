#!/bin/bash
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A4000:1
#SBATCH --array=1-6


configs="--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 |\
         --p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 |\
          --p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 |\
         --p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 |\
         --p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 |\
         --p_model esm3-medium --m_model MoLFormer --fusion_name data/pathbank/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0"

IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

tasks_configs="--task_name BBBP --bs 2048 --metric auc |\
--task_name CEP --bs 2048 --metric rmse |\
--task_name ClinTox --bs 2048 --metric auc |\
--task_name FreeSolv --bs 2048 --metric rmse |\
--task_name HIV --bs 2048 --metric auc |\
--task_name Lipophilicity --bs 2048 --metric rmse |\
--task_name BetaLactamase --bs 2048 --metric rmse |\
--task_name Fluorescence --bs 2048 --metric rmse |\
--task_name Stability --bs 2048 --metric rmse |\
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
