#!/bin/bash
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --array=1-120

configs="--p_model ProtBert --m_model MoLFormer --fusion_name data/pathbank/model/MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/pathbank/model/MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/pathbank/model/MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/pathbank/model/MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/pathbank/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/pathbank/model/MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/pathbank/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/pathbank/model/MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/pathbank/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/pathbank/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/pathbank/model/MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/pathbank/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/pathbank/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/pathbank/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/pathbank/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/pathbank/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/pathbank/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/pathbank/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/pathbank/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/pathbank/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/pathbank/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/pathbank/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/pathbank/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/MolCLR-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-2-256-0.3-10-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/MolCLR-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/ChemBERTa-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/MoLFormer-1-256-0.3-1-5e-05-256-0.0-10"

IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

tasks_configs="--task_name BACE --bs 64 --metric f1_max |\
--task_name BBBP --bs 64 --metric f1_max |\
--task_name CEP --bs 512 --metric pearsonr |\
--task_name ClinTox --bs 64 --metric f1_max |\
--task_name Delaney --bs 64 --metric pearsonr |\
--task_name FreeSolv --bs 64 --metric pearsonr |\
--task_name HIV --bs 512 --metric f1_max |\
--task_name Lipophilicity --bs 64 --metric pearsonr |\
--task_name Malaria --bs 256 --metric pearsonr |\
--task_name SIDER --bs 64 --metric f1_max |\
--task_name Tox21 --bs 256 --metric f1_max |\
--task_name Fluorescence --bs 512 --metric pearsonr |\
--task_name Stability --bs 512 --metric pearsonr |\
--task_name BinaryLocalization --bs 256 --metric f1_max |\
--task_name HumanPPI --bs 512 --bs 256 --metric f1_max |\
--task_name YeastPPI --bs 256 --metric f1_max |\
--task_name PPIAffinity --bs 64 --metric pearsonr |\
--task_name BindingDB --bs 256 --metric pearsonr |\
--task_name PDBBind --bs 512 --metric pearsonr |\
--task_name DrugBank --bs 512 --metric f1_max |\
--task_name Davis --bs 512 --metric f1_max"
IFS='|' read -ra task_array <<< "$tasks_configs"


eval "$(conda shell.bash hook)"
conda activate retd
export PYTHONPATH=$PYTHONPATH:$(pwd)


for task in "${task_array[@]}"; do
    python eval_tasks/trainer.py $config $task --use_fuse 1 --use_model 1
    python eval_tasks/trainer.py $config $task --use_fuse 0 --use_model 1
done

