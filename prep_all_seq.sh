#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --array=1-30
#tasks=("BACE BBBP CEP ClinTox Delaney FreeSolv HIV Lipophilicity Malaria SIDER Tox21 BetaLactamase Fluorescence Stability Solubility BinaryLocalization SubcellularLocalization EnzymeCommission GOMF GOBP GOCC SecondaryStructure HumanPPI YeastPPI PPIAffinity BindingDB PDBBind DrugBank Davis")
configs="--task BACE|\
--task BBBP|\
--task CEP|\
--task ClinTox|\
--task Delaney|\
--task FreeSolv|\
--task HIV|\
--task Lipophilicity|\
--task Malaria|\
--task SIDER|\
--task Tox21|\
--task BetaLactamase|\
--task Fluorescence|\
--task Stability|\
--task Solubility|\
--task BinaryLocalization|\
--task SubcellularLocalization|\
--task EnzymeCommission|\
--task GOMF|\
--task GOBP|\
--task GOCC|\
--task SecondaryStructure|\
--task HumanPPI|\
--task YeastPPI|\
--task PPIAffinity|\
--task BindingDB|\
--task PDBBind|\
--task DrugBank|\
--task Davis"
IFS='|' read -ra config_array <<< "$configs"
eval "$(conda shell.bash hook)"
conda activate retd
export PYTHONPATH=$PYTHONPATH:$(pwd)
task=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python eval_tasks/prep_tasks_seqs.py "$task"
