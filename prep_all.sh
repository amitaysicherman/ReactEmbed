#!/bin/bash

# Define molecule and protein models
molecule_models=("ChemBERTa" "MoLFormer")
protein_models=("ProtBert" "esm3-small" "esm3-medium")

# Step 1: Generate vectors for each molecule and protein model
for m_model in "${molecule_models[@]}"; do
    python preprocessing/seq_to_vec.py --model "$m_model"
done
for p_model in "${protein_models[@]}"; do
    python preprocessing/seq_to_vec.py --model "$p_model"
done

mol_task=("BACE" "BBBP" "CEP" "ClinTox" "Delaney" "FreeSolv" "HIV" "Lipophilicity" "Malaria" "SIDER" "Tox21")
for m_task in "${mol_task[@]}"; do
    python eval_tasks/prep_tasks_seqs.py --task "$m_task"
done

prot_task=("BetaLactamase" "Fluorescence" "Stability" "BinaryLocalization" "HumanPPI" "YeastPPI" "PPIAffinity")
for p_task in "${prot_task[@]}"; do
    python eval_tasks/prep_tasks_seqs.py --task "$p_task"
done

lig_task=("BindingDB" "PDBBind" "DrugBank" "Davis")
for l_task in "${lig_task[@]}"; do
    python eval_tasks/prep_tasks_seqs.py --task "$l_task"
done