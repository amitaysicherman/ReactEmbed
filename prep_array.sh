#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-11



configs="--task BetaLactamase --p_model GearNet|\
      --task Fluorescence --p_model GearNet|\
      --task Stability --p_model GearNet|\
      --task BinaryLocalization --p_model GearNet|\
      --task HumanPPI --p_model GearNet|\
      --task YeastPPI --p_model GearNet|\
      --task PPIAffinity --p_model GearNet|\
      --task BindingDB --p_model GearNet --m_model ChemBERTa|\
      --task PDBBind --p_model GearNet --m_model ChemBERTa|\
      --task DrugBank --p_model GearNet --m_model ChemBERTa|\
      --task Davis --p_model GearNet --m_model ChemBERTa"

# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python eval_tasks/prep_tasks_vecs.py $config