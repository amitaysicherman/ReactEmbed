#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-11

#pathbank reactome reactome_all
DATA_NAMES_ARRAY=("pathbank" "reactome" "reactome_all")
PROTEINS_MODELS_ARRAY=("GearNet" "ProtBert" "esm3-small", "esm3-medium")
MOLECULES_MODELS_ARRAY=("MolCLR" "ChemBERTa" "MoLFormer")

for data_name in "${DATA_NAMES_ARRAY[@]}" ; do
  echo python preprocessing/biopax_parser.py --data_name $data_name
done

for data_name in "${DATA_NAMES_ARRAY[@]}" ; do
  for p_model in "${PROTEINS_MODELS_ARRAY[@]}" ;do
    echo python preprocessing/seq_to_vec.py --data_name $data_name --model $p_model
  done  
  for m_model in "${MOLECULES_MODELS_ARRAY[@]}" ;do
      echo python preprocessing/seq_to_vec.py --data_name $data_name --model $m_model
  done
  for p_model in "${PROTEINS_MODELS_ARRAY[@]}" ; do
    for m_model in "${MOLECULES_MODELS_ARRAY[@]}" ; do
      echo python contrastive_learning/trainer.py --data_name $data_name --p_model $p_model --m_model $m_model
    done
  done
done


for m_model in "${molecule_models[@]}"; do
    echo python preprocessing/seq_to_vec.py --model "$m_model"
done
for p_model in "${protein_models[@]}"; do
    echo python preprocessing/seq_to_vec.py --model "$p_model"
done

mol_task=("BACE" "BBBP" "CEP" "ClinTox" "Delaney" "FreeSolv" "HIV" "Lipophilicity" "Malaria" "SIDER" "Tox21")
for m_task in "${mol_task[@]}"; do
  for m_model in "${molecule_models[@]}"; do
    echo python eval_tasks/prep_tasks_vecs.py --task "$m_task" --m_model "$m_model"
  done
done

prot_task=("BetaLactamase" "Fluorescence" "Stability" "BinaryLocalization" "HumanPPI" "YeastPPI" "PPIAffinity")
for p_task in "${prot_task[@]}"; do
  for p_model in "${protein_models[@]}"; do
    echo python eval_tasks/prep_tasks_vecs.py --task "$p_task" --p_model "$p_model"
  done
done

lig_task=("BindingDB" "PDBBind" "DrugBank" "Davis")
for l_task in "${lig_task[@]}"; do
  for p_model in "${protein_models[@]}"; do
    for m_model in "${molecule_models[@]}"; do
      echo python eval_tasks/prep_tasks_vecs.py --task "$l_task" --p_model "$p_model" --m_model "$m_model"
    done
  done
done
