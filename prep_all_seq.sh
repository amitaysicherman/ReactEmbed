#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --array=1-30
#task_names=("BACE BBBP CEP ClinTox Delaney FreeSolv HIV Lipophilicity Malaria SIDER Tox21 BetaLactamase Fluorescence Stability Solubility BinaryLocalization SubcellularLocalization EnzymeCommission GOMF GOBP GOCC SecondaryStructure HumanPPI YeastPPI PPIAffinity BindingDB PDBBind DrugBank Davis")
configs="--task_name BACE|\
--task_name BBBP|\
--task_name CEP|\
--task_name ClinTox|\
--task_name Delaney|\
--task_name FreeSolv|\
--task_name HIV|\
--task_name Lipophilicity|\
--task_name Malaria|\
--task_name SIDER|\
--task_name Tox21|\
--task_name BetaLactamase|\
--task_name Fluorescence|\
--task_name Stability|\
--task_name Solubility|\
--task_name BinaryLocalization|\
--task_name SubcellularLocalization|\
--task_name EnzymeCommission|\
--task_name GOMF|\
--task_name GOBP|\
--task_name GOCC|\
--task_name SecondaryStructure|\
--task_name HumanPPI|\
--task_name YeastPPI|\
--task_name PPIAffinity|\
--task_name BindingDB|\
--task_name PDBBind|\
--task_name DrugBank|\
--task_name Davis"
IFS='|' read -ra config_array <<< "$configs"
eval "$(conda shell.bash hook)"
conda activate retd
export PYTHONPATH=$PYTHONPATH:$(pwd)
task_name=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python eval_tasks/prep_tasks_seqs.py "$task_name"
