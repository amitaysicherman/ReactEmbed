#!/bin/bash
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --array=1-135

configs="--task_name BACE --metric f1_max --use_fuse 0 --use_model 1|\
--task_name BACE --metric f1_max --use_fuse 1 --use_model 0|\
--task_name BACE --metric f1_max --use_fuse 1 --use_model 1|\
--task_name BACE --metric auprc --use_fuse 0 --use_model 1|\
--task_name BACE --metric auprc --use_fuse 1 --use_model 0|\
--task_name BACE --metric auprc --use_fuse 1 --use_model 1|\
--task_name BBBP --metric f1_max --use_fuse 0 --use_model 1|\
--task_name BBBP --metric f1_max --use_fuse 1 --use_model 0|\
--task_name BBBP --metric f1_max --use_fuse 1 --use_model 1|\
--task_name BBBP --metric auprc --use_fuse 0 --use_model 1|\
--task_name BBBP --metric auprc --use_fuse 1 --use_model 0|\
--task_name BBBP --metric auprc --use_fuse 1 --use_model 1|\
--task_name CEP --metric r2 --use_fuse 0 --use_model 1|\
--task_name CEP --metric r2 --use_fuse 1 --use_model 0|\
--task_name CEP --metric r2 --use_fuse 1 --use_model 1|\
--task_name CEP --metric mse --use_fuse 0 --use_model 1|\
--task_name CEP --metric mse --use_fuse 1 --use_model 0|\
--task_name CEP --metric mse --use_fuse 1 --use_model 1|\
--task_name ClinTox --metric f1_max --use_fuse 0 --use_model 1|\
--task_name ClinTox --metric f1_max --use_fuse 1 --use_model 0|\
--task_name ClinTox --metric f1_max --use_fuse 1 --use_model 1|\
--task_name ClinTox --metric auprc --use_fuse 0 --use_model 1|\
--task_name ClinTox --metric auprc --use_fuse 1 --use_model 0|\
--task_name ClinTox --metric auprc --use_fuse 1 --use_model 1|\
--task_name Delaney --metric r2 --use_fuse 0 --use_model 1|\
--task_name Delaney --metric r2 --use_fuse 1 --use_model 0|\
--task_name Delaney --metric r2 --use_fuse 1 --use_model 1|\
--task_name Delaney --metric mse --use_fuse 0 --use_model 1|\
--task_name Delaney --metric mse --use_fuse 1 --use_model 0|\
--task_name Delaney --metric mse --use_fuse 1 --use_model 1|\
--task_name FreeSolv --metric r2 --use_fuse 0 --use_model 1|\
--task_name FreeSolv --metric r2 --use_fuse 1 --use_model 0|\
--task_name FreeSolv --metric r2 --use_fuse 1 --use_model 1|\
--task_name FreeSolv --metric mse --use_fuse 0 --use_model 1|\
--task_name FreeSolv --metric mse --use_fuse 1 --use_model 0|\
--task_name FreeSolv --metric mse --use_fuse 1 --use_model 1|\
--task_name HIV --metric f1_max --use_fuse 0 --use_model 1|\
--task_name HIV --metric f1_max --use_fuse 1 --use_model 0|\
--task_name HIV --metric f1_max --use_fuse 1 --use_model 1|\
--task_name HIV --metric auprc --use_fuse 0 --use_model 1|\
--task_name HIV --metric auprc --use_fuse 1 --use_model 0|\
--task_name HIV --metric auprc --use_fuse 1 --use_model 1|\
--task_name Lipophilicity --metric r2 --use_fuse 0 --use_model 1|\
--task_name Lipophilicity --metric r2 --use_fuse 1 --use_model 0|\
--task_name Lipophilicity --metric r2 --use_fuse 1 --use_model 1|\
--task_name Lipophilicity --metric mse --use_fuse 0 --use_model 1|\
--task_name Lipophilicity --metric mse --use_fuse 1 --use_model 0|\
--task_name Lipophilicity --metric mse --use_fuse 1 --use_model 1|\
--task_name Malaria --metric r2 --use_fuse 0 --use_model 1|\
--task_name Malaria --metric r2 --use_fuse 1 --use_model 0|\
--task_name Malaria --metric r2 --use_fuse 1 --use_model 1|\
--task_name Malaria --metric mse --use_fuse 0 --use_model 1|\
--task_name Malaria --metric mse --use_fuse 1 --use_model 0|\
--task_name Malaria --metric mse --use_fuse 1 --use_model 1|\
--task_name SIDER --metric f1_max --use_fuse 0 --use_model 1|\
--task_name SIDER --metric f1_max --use_fuse 1 --use_model 0|\
--task_name SIDER --metric f1_max --use_fuse 1 --use_model 1|\
--task_name SIDER --metric auprc --use_fuse 0 --use_model 1|\
--task_name SIDER --metric auprc --use_fuse 1 --use_model 0|\
--task_name SIDER --metric auprc --use_fuse 1 --use_model 1|\
--task_name Tox21 --metric f1_max --use_fuse 0 --use_model 1|\
--task_name Tox21 --metric f1_max --use_fuse 1 --use_model 0|\
--task_name Tox21 --metric f1_max --use_fuse 1 --use_model 1|\
--task_name Tox21 --metric auprc --use_fuse 0 --use_model 1|\
--task_name Tox21 --metric auprc --use_fuse 1 --use_model 0|\
--task_name Tox21 --metric auprc --use_fuse 1 --use_model 1|\
--task_name BetaLactamase --metric r2 --use_fuse 0 --use_model 1|\
--task_name BetaLactamase --metric r2 --use_fuse 1 --use_model 0|\
--task_name BetaLactamase --metric r2 --use_fuse 1 --use_model 1|\
--task_name BetaLactamase --metric mse --use_fuse 0 --use_model 1|\
--task_name BetaLactamase --metric mse --use_fuse 1 --use_model 0|\
--task_name BetaLactamase --metric mse --use_fuse 1 --use_model 1|\
--task_name Fluorescence --metric r2 --use_fuse 0 --use_model 1|\
--task_name Fluorescence --metric r2 --use_fuse 1 --use_model 0|\
--task_name Fluorescence --metric r2 --use_fuse 1 --use_model 1|\
--task_name Fluorescence --metric mse --use_fuse 0 --use_model 1|\
--task_name Fluorescence --metric mse --use_fuse 1 --use_model 0|\
--task_name Fluorescence --metric mse --use_fuse 1 --use_model 1|\
--task_name Stability --metric r2 --use_fuse 0 --use_model 1|\
--task_name Stability --metric r2 --use_fuse 1 --use_model 0|\
--task_name Stability --metric r2 --use_fuse 1 --use_model 1|\
--task_name Stability --metric mse --use_fuse 0 --use_model 1|\
--task_name Stability --metric mse --use_fuse 1 --use_model 0|\
--task_name Stability --metric mse --use_fuse 1 --use_model 1|\
--task_name BinaryLocalization --metric f1_max --use_fuse 0 --use_model 1|\
--task_name BinaryLocalization --metric f1_max --use_fuse 1 --use_model 0|\
--task_name BinaryLocalization --metric f1_max --use_fuse 1 --use_model 1|\
--task_name BinaryLocalization --metric auprc --use_fuse 0 --use_model 1|\
--task_name BinaryLocalization --metric auprc --use_fuse 1 --use_model 0|\
--task_name BinaryLocalization --metric auprc --use_fuse 1 --use_model 1|\
--task_name HumanPPI --metric f1_max --use_fuse 0 --use_model 1|\
--task_name HumanPPI --metric f1_max --use_fuse 1 --use_model 0|\
--task_name HumanPPI --metric f1_max --use_fuse 1 --use_model 1|\
--task_name HumanPPI --metric auprc --use_fuse 0 --use_model 1|\
--task_name HumanPPI --metric auprc --use_fuse 1 --use_model 0|\
--task_name HumanPPI --metric auprc --use_fuse 1 --use_model 1|\
--task_name YeastPPI --metric f1_max --use_fuse 0 --use_model 1|\
--task_name YeastPPI --metric f1_max --use_fuse 1 --use_model 0|\
--task_name YeastPPI --metric f1_max --use_fuse 1 --use_model 1|\
--task_name YeastPPI --metric auprc --use_fuse 0 --use_model 1|\
--task_name YeastPPI --metric auprc --use_fuse 1 --use_model 0|\
--task_name YeastPPI --metric auprc --use_fuse 1 --use_model 1|\
--task_name PPIAffinity --metric r2 --use_fuse 0 --use_model 1|\
--task_name PPIAffinity --metric r2 --use_fuse 1 --use_model 0|\
--task_name PPIAffinity --metric r2 --use_fuse 1 --use_model 1|\
--task_name PPIAffinity --metric mse --use_fuse 0 --use_model 1|\
--task_name PPIAffinity --metric mse --use_fuse 1 --use_model 0|\
--task_name PPIAffinity --metric mse --use_fuse 1 --use_model 1|\
--task_name BindingDB --metric r2 --use_fuse 0 --use_model 1|\
--task_name BindingDB --metric r2 --use_fuse 1 --use_model 0|\
--task_name BindingDB --metric r2 --use_fuse 1 --use_model 1|\
--task_name BindingDB --metric mse --use_fuse 0 --use_model 1|\
--task_name BindingDB --metric mse --use_fuse 1 --use_model 0|\
--task_name BindingDB --metric mse --use_fuse 1 --use_model 1|\
--task_name PDBBind --metric r2 --use_fuse 0 --use_model 1|\
--task_name PDBBind --metric r2 --use_fuse 1 --use_model 0|\
--task_name PDBBind --metric r2 --use_fuse 1 --use_model 1|\
--task_name PDBBind --metric mse --use_fuse 0 --use_model 1|\
--task_name PDBBind --metric mse --use_fuse 1 --use_model 0|\
--task_name PDBBind --metric mse --use_fuse 1 --use_model 1|\
--task_name DrugBank --metric f1_max --use_fuse 0 --use_model 1|\
--task_name DrugBank --metric f1_max --use_fuse 1 --use_model 0|\
--task_name DrugBank --metric f1_max --use_fuse 1 --use_model 1|\
--task_name DrugBank --metric auprc --use_fuse 0 --use_model 1|\
--task_name DrugBank --metric auprc --use_fuse 1 --use_model 0|\
--task_name DrugBank --metric auprc --use_fuse 1 --use_model 1|\
--task_name Davis --metric f1_max --use_fuse 0 --use_model 1|\
--task_name Davis --metric f1_max --use_fuse 1 --use_model 0|\
--task_name Davis --metric f1_max --use_fuse 1 --use_model 1|\
--task_name Davis --metric auprc --use_fuse 0 --use_model 1|\
--task_name Davis --metric auprc --use_fuse 1 --use_model 0|\
--task_name Davis --metric auprc --use_fuse 1 --use_model 1"

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
    model01=$(echo $model | sed 's/0.0/0.1/g')
    python eval_tasks/trainer.py $config $model01
    model05=$(echo $model | sed 's/0.0/0.1/g')
    python eval_tasks/trainer.py $config $model05
    modelpb=$(echo $model | sed 's/reactome/pathbank/g')
    python eval_tasks/trainer.py $config $modelpb

done