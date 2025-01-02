#!/bin/bash
#SBATCH --mem=64G
#SBATCH --requeue
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
--task_name CEP --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name CEP --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name CEP --metric pearsonr --use_fuse 1 --use_model 1|\
--task_name CEP --metric mse --use_fuse 0 --use_model 1|\
--task_name CEP --metric mse --use_fuse 1 --use_model 0|\
--task_name CEP --metric mse --use_fuse 1 --use_model 1|\
--task_name ClinTox --metric f1_max --use_fuse 0 --use_model 1|\
--task_name ClinTox --metric f1_max --use_fuse 1 --use_model 0|\
--task_name ClinTox --metric f1_max --use_fuse 1 --use_model 1|\
--task_name ClinTox --metric auprc --use_fuse 0 --use_model 1|\
--task_name ClinTox --metric auprc --use_fuse 1 --use_model 0|\
--task_name ClinTox --metric auprc --use_fuse 1 --use_model 1|\
--task_name Delaney --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name Delaney --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name Delaney --metric pearsonr --use_fuse 1 --use_model 1|\
--task_name Delaney --metric mse --use_fuse 0 --use_model 1|\
--task_name Delaney --metric mse --use_fuse 1 --use_model 0|\
--task_name Delaney --metric mse --use_fuse 1 --use_model 1|\
--task_name FreeSolv --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name FreeSolv --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name FreeSolv --metric pearsonr --use_fuse 1 --use_model 1|\
--task_name FreeSolv --metric mse --use_fuse 0 --use_model 1|\
--task_name FreeSolv --metric mse --use_fuse 1 --use_model 0|\
--task_name FreeSolv --metric mse --use_fuse 1 --use_model 1|\
--task_name HIV --metric f1_max --use_fuse 0 --use_model 1|\
--task_name HIV --metric f1_max --use_fuse 1 --use_model 0|\
--task_name HIV --metric f1_max --use_fuse 1 --use_model 1|\
--task_name HIV --metric auprc --use_fuse 0 --use_model 1|\
--task_name HIV --metric auprc --use_fuse 1 --use_model 0|\
--task_name HIV --metric auprc --use_fuse 1 --use_model 1|\
--task_name Lipophilicity --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name Lipophilicity --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name Lipophilicity --metric pearsonr --use_fuse 1 --use_model 1|\
--task_name Lipophilicity --metric mse --use_fuse 0 --use_model 1|\
--task_name Lipophilicity --metric mse --use_fuse 1 --use_model 0|\
--task_name Lipophilicity --metric mse --use_fuse 1 --use_model 1|\
--task_name Malaria --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name Malaria --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name Malaria --metric pearsonr --use_fuse 1 --use_model 1|\
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
--task_name BetaLactamase --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name BetaLactamase --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name BetaLactamase --metric pearsonr --use_fuse 1 --use_model 1|\
--task_name BetaLactamase --metric mse --use_fuse 0 --use_model 1|\
--task_name BetaLactamase --metric mse --use_fuse 1 --use_model 0|\
--task_name BetaLactamase --metric mse --use_fuse 1 --use_model 1|\
--task_name Fluorescence --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name Fluorescence --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name Fluorescence --metric pearsonr --use_fuse 1 --use_model 1|\
--task_name Fluorescence --metric mse --use_fuse 0 --use_model 1|\
--task_name Fluorescence --metric mse --use_fuse 1 --use_model 0|\
--task_name Fluorescence --metric mse --use_fuse 1 --use_model 1|\
--task_name Stability --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name Stability --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name Stability --metric pearsonr --use_fuse 1 --use_model 1|\
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
--task_name PPIAffinity --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name PPIAffinity --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name PPIAffinity --metric pearsonr --use_fuse 1 --use_model 1|\
--task_name PPIAffinity --metric mse --use_fuse 0 --use_model 1|\
--task_name PPIAffinity --metric mse --use_fuse 1 --use_model 0|\
--task_name PPIAffinity --metric mse --use_fuse 1 --use_model 1|\
--task_name BindingDB --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name BindingDB --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name BindingDB --metric pearsonr --use_fuse 1 --use_model 1|\
--task_name BindingDB --metric mse --use_fuse 0 --use_model 1|\
--task_name BindingDB --metric mse --use_fuse 1 --use_model 0|\
--task_name BindingDB --metric mse --use_fuse 1 --use_model 1|\
--task_name PDBBind --metric pearsonr --use_fuse 0 --use_model 1|\
--task_name PDBBind --metric pearsonr --use_fuse 1 --use_model 0|\
--task_name PDBBind --metric pearsonr --use_fuse 1 --use_model 1|\
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

IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}



# Initialize an empty array to store configurations
declare -a models

# Function to process directories and build configurations
process_directories() {
    local base_dir=$1

    # Loop through all directories in the base directory
    for d in "$base_dir"/*; do
        # Get just the directory name without the path
        dir_name=$(basename "$d")

        # Initialize variables
        local p_model=""
        local m_model=""
        local processed_name="$dir_name"

        # Process esm3-medium
        if [[ $dir_name == esm3-medium* ]]; then
            p_model="esm3-medium"
            processed_name=${dir_name#esm3-medium-}
        # Process esm3-small
        elif [[ $dir_name == esm3-small* ]]; then
            p_model="esm3-small"
            processed_name=${dir_name#esm3-small-}
        # Process other cases
        else
            p_model=$(echo "$dir_name" | cut -d'-' -f1)
            processed_name=${dir_name#$p_model-}
        fi

        # Get m_model (first part after removing p_model)
        m_model=$(echo "$processed_name" | cut -d'-' -f1)

        # Add configuration to array
        models+=("--p_model $p_model --m_model $m_model --fusion_name $base_dir/$dir_name/")
    done
}
if [[ $config == *"--use_fuse 0 --use_model 1"* ]]; then
  models+=(
    "--p_model ProtBert --m_model ChemBERTa --fusion_name data/reactome/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.0/"
    "--p_model ProtBert --m_model ChemBERTa --fusion_name data/pathbank/model/ProtBert-ChemBERTa-1-256-0.3-1-5e-05-256-0.0/"
  )
else
  process_directories "data/pathbank/model"
  process_directories "data/reactome/model"
fi
IFS='|' read -ra models_array <<< "$models"


eval "$(conda shell.bash hook)"
conda activate retd
export PYTHONPATH=$PYTHONPATH:$(pwd)
for model in "${models_array[@]}"
do
  python eval_tasks/trainer.py $config $model
done