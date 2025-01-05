#!/bin/bash
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --array=1-90

configs="--p_model GearNet --m_model MoLFormer --fusion_name data/pathbank/model/GearNet-MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/pathbank/model/GearNet-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/pathbank/model/ProtBert-MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/pathbank/model/ProtBert-MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/pathbank/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/pathbank/model/GearNet-MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/pathbank/model/esm3-small-MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/pathbank/model/esm3-small-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/pathbank/model/esm3-medium-MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/pathbank/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/pathbank/model/esm3-small-MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/pathbank/model/esm3-medium-MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/GearNet-MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/GearNet-ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-2-256-0.3-10-5e-05-256-0.0-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/GearNet-MolCLR-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/GearNet-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/GearNet-MolCLR-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/GearNet-ChemBERTa-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-256-0.3-1-5e-05-256-0.1-5 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/GearNet-ChemBERTa-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/GearNet-MolCLR-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-256-0.3-1-5e-05-256-0.5-5 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/GearNet-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model GearNet --m_model ChemBERTa --fusion_name data/reactome/model/GearNet-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/GearNet-MolCLR-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-256-0.3-1-5e-05-256-0.0-1 |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model GearNet --m_model MolCLR --fusion_name data/reactome/model/GearNet-MolCLR-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-small --m_model MolCLR --fusion_name data/reactome/model/esm3-small-MolCLR-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.0-10 |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-256-0.3-1-5e-05-256-0.0-10"

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

