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


# Define tasks
tasks=("BACE" "BBBP" "CEP" "ClinTox" "Delaney" "FreeSolv" "HIV" "Lipophilicity" "Malaria" "SIDER" "Tox21" \
"BetaLactamase" "Fluorescence" "Stability" "BinaryLocalization" "HumanPPI" "YeastPPI" "PPIAffinity" \
"BindingDB" "PDBBind" "DrugBank" "Davis")

# Step 2: Evaluate tasks for each molecule-protein pair
for task in "${tasks[@]}"; do
  for m_model in "${molecule_models[@]}"; do
    for p_model in "${protein_models[@]}"; do
      echo "Running task: $task with molecule model: $m_model and protein model: $p_model"
      python eval_tasks/prep_tasks.py --task "$task" --p_model "$p_model" --m_model "$m_model"
    done
  done
done
