#!/bin/bash
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:A4000:1
#SBATCH --array=1-100

configs="--p_model esm3-small --m_model ChemBERTa --fusion_name data/pathbank/model/esm3-small-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name BBBP --bs 2048 --metric auc |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name BBBP --bs 2048 --metric auc |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name BBBP --bs 2048 --metric auc |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name BBBP --bs 2048 --metric auc |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name BBBP --bs 2048 --metric auc |\
--p_model esm3-small --m_model ChemBERTa --fusion_name data/reactome/model/esm3-small-ChemBERTa-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name BBBP --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/pathbank/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name CEP --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name CEP --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name CEP --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name CEP --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name CEP --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name CEP --bs 2048 --metric rmse |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/pathbank/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name ClinTox --bs 2048 --metric auc |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name ClinTox --bs 2048 --metric auc |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name ClinTox --bs 2048 --metric auc |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name ClinTox --bs 2048 --metric auc |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name ClinTox --bs 2048 --metric auc |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name ClinTox --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/pathbank/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name FreeSolv --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name FreeSolv --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name FreeSolv --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name FreeSolv --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name FreeSolv --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name FreeSolv --bs 2048 --metric rmse |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/pathbank/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name Lipophilicity --bs 2048 --metric rmse |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name Lipophilicity --bs 2048 --metric rmse |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name Lipophilicity --bs 2048 --metric rmse |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name Lipophilicity --bs 2048 --metric rmse |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name Lipophilicity --bs 2048 --metric rmse |\
--p_model esm3-small --m_model MoLFormer --fusion_name data/reactome/model/esm3-small-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name Lipophilicity --bs 2048 --metric rmse |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/pathbank/model/ProtBert-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name BindingDB --bs 2048 --metric rmse |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name BindingDB --bs 2048 --metric rmse |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name BindingDB --bs 2048 --metric rmse |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name BindingDB --bs 2048 --metric rmse |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name BindingDB --bs 2048 --metric rmse |\
--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name BindingDB --bs 2048 --metric rmse |\
--p_model GearNet --m_model MoLFormer --fusion_name data/pathbank/model/GearNet-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name Davis --bs 2048 --metric auc |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name Davis --bs 2048 --metric auc |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name Davis --bs 2048 --metric auc |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name Davis --bs 2048 --metric auc |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name Davis --bs 2048 --metric auc |\
--p_model GearNet --m_model MoLFormer --fusion_name data/reactome/model/GearNet-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name Davis --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/pathbank/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name DrugBank --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name DrugBank --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name DrugBank --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name DrugBank --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name DrugBank --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name DrugBank --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/pathbank/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name PDBBind --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name PDBBind --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name PDBBind --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name PDBBind --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name PDBBind --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name PDBBind --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/pathbank/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name HumanPPI --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name HumanPPI --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name HumanPPI --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name HumanPPI --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name HumanPPI --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name HumanPPI --bs 2048 --metric auc |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/pathbank/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name PPIAffinity --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name PPIAffinity --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name PPIAffinity --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name PPIAffinity --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name PPIAffinity --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name PPIAffinity --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/pathbank/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name YeastPPI --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name YeastPPI --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name YeastPPI --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name YeastPPI --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name YeastPPI --bs 2048 --metric auc |\
--p_model ProtBert --m_model MoLFormer --fusion_name data/reactome/model/ProtBert-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name YeastPPI --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/pathbank/model/esm3-medium-MolCLR-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name BetaLactamase --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name BetaLactamase --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name BetaLactamase --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name BetaLactamase --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name BetaLactamase --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MolCLR --fusion_name data/reactome/model/esm3-medium-MolCLR-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name BetaLactamase --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MolCLR --fusion_name data/pathbank/model/ProtBert-MolCLR-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name Fluorescence --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name Fluorescence --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name Fluorescence --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name Fluorescence --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name Fluorescence --bs 2048 --metric rmse |\
--p_model ProtBert --m_model MolCLR --fusion_name data/reactome/model/ProtBert-MolCLR-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name Fluorescence --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/pathbank/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name GeneOntologyCC --bs 2048 --metric auc |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name GeneOntologyCC --bs 2048 --metric auc |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name GeneOntologyCC --bs 2048 --metric auc |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name GeneOntologyCC --bs 2048 --metric auc |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name GeneOntologyCC --bs 2048 --metric auc |\
--p_model esm3-medium --m_model ChemBERTa --fusion_name data/reactome/model/esm3-medium-ChemBERTa-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name GeneOntologyCC --bs 2048 --metric auc |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/pathbank/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-0 --task_name Stability --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.1-0 --task_name Stability --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-0.5-0 --task_name Stability --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256-1-1 --task_name Stability --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.1-256-1-0 --task_name Stability --bs 2048 --metric rmse |\
--p_model esm3-medium --m_model MoLFormer --fusion_name data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.5-256-1-0 --task_name Stability --bs 2048 --metric rmse"

IFS="|" read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}

eval "$(conda shell.bash hook)"
conda activate retd
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo eval_tasks/trainer.py $config

python eval_tasks/trainer.py $config --use_fuse 1 --use_model 1
