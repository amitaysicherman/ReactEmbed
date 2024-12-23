#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-414

DATA_NAMES_ARRAY=("pathbank" "reactome" "reactome_all")
PROTEINS_MODELS_ARRAY=("GearNet" "ProtBert" "esm3-small", "esm3-medium")
MOLECULES_MODELS_ARRAY=("MolCLR" "ChemBERTa" "MoLFormer")
mol_task=("BACE" "BBBP" "CEP" "ClinTox" "Delaney" "FreeSolv" "HIV" "Lipophilicity" "Malaria" "SIDER" "Tox21")
prot_task=("BetaLactamase" "Fluorescence" "Stability" "BinaryLocalization" "HumanPPI" "YeastPPI" "PPIAffinity")
lig_task=("BindingDB" "PDBBind" "DrugBank" "Davis")
all_tasks=("${mol_task[@]}" "${prot_task[@]}" "${lig_task[@]}")

commands_array=()

#for data_name in "${DATA_NAMES_ARRAY[@]}" ; do
#  echo python preprocessing/biopax_parser.py --data_name $data_name
#done

#for task in "${all_tasks[@]}"; do
#    python eval_tasks/prep_tasks_seqs.py --task "$task"
#done


for data_name in "${DATA_NAMES_ARRAY[@]}" ; do
  for p_model in "${PROTEINS_MODELS_ARRAY[@]}" ;do
    cmd="python preprocessing/seq_to_vec.py --data_name $data_name --model $p_model"
    commands_array+=($cmd)
  done
  for m_model in "${MOLECULES_MODELS_ARRAY[@]}" ;do
    cmd="python preprocessing/seq_to_vec.py --data_name $data_name --model $m_model"
    commands_array+=($cmd)
  done
  for p_model in "${PROTEINS_MODELS_ARRAY[@]}" ; do
    for m_model in "${MOLECULES_MODELS_ARRAY[@]}" ; do
      cmd="python contrastive_learning/trainer.py --data_name $data_name --p_model $p_model --m_model $m_model"
      commands_array+=($cmd)
    done
  done
done


for m_model in "${molecule_models[@]}"; do
    cmd="python preprocessing/seq_to_vec.py --model $m_model"
    commands_array+=($cmd)
done
for p_model in "${protein_models[@]}"; do
    cmd="python preprocessing/seq_to_vec.py --model $p_model"
    commands_array+=($cmd)
done

for m_task in "${mol_task[@]}"; do
  for m_model in "${molecule_models[@]}"; do
    cmd="python eval_tasks/prep_tasks_vecs.py --task $m_task --m_model $m_model"
    commands_array+=($cmd)
  done
done

for p_task in "${prot_task[@]}"; do
  for p_model in "${protein_models[@]}"; do
    cmd="python eval_tasks/prep_tasks_vecs.py --task $p_task --p_model $p_model"
    commands_array+=($cmd)
  done
done

for l_task in "${lig_task[@]}"; do
  for p_model in "${protein_models[@]}"; do
    for m_model in "${molecule_models[@]}"; do
      cmd="python eval_tasks/prep_tasks_vecs.py --task $l_task --p_model $p_model --m_model $m_model"
      commands_array+=($cmd)
    done
  done
done

# print number of commands
echo "Number of commands: ${#commands_array[@]}"
# run the command based on the $SLURM_ARRAY_TASK_ID-1
cmd_to_run=${commands_array[$SLURM_ARRAY_TASK_ID-1]}
echo "Running command: $cmd_to_run"

# if the cmd contain "esm3" change conda environment to ReactEmbedESM
if [[ $cmd_to_run == *"esm3"* ]]; then
  conda activate ReactEmbedESM
fi
eval $cmd_to_run
