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
  for m_model in "${molecule_models[@]}"; do
    python preprocessing/prep_tasks.py --task "$m_task" --m_model "$m_model"
  done
done

prot_task=("BetaLactamase" "Fluorescence" "Stability" "BinaryLocalization" "HumanPPI" "YeastPPI" "PPIAffinity")
for p_task in "${prot_task[@]}"; do
  for p_model in "${protein_models[@]}"; do
    python preprocessing/prep_tasks.py --task "$p_task" --p_model "$p_model"
  done
done

lig_task=("BindingDB" "PDBBind" "DrugBank" "Davis")
for l_task in "${lig_task[@]}"; do
  for p_model in "${protein_models[@]}"; do
    for m_model in "${molecule_models[@]}"; do
      python preprocessing/prep_tasks.py --task "$l_task" --p_model "$p_model" --m_model "$m_model"
    done
  done
done