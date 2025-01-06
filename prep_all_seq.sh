#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --array=1-30
#task_names=("BACE BBBP CEP ClinTox Delaney FreeSolv HIV Lipophilicity Malaria SIDER Tox21 BetaLactamase Fluorescence Stability Solubility BinaryLocalization SubcellularLocalization EnzymeCommission GOMF GOBP GOCC SecondaryStructure HumanPPI YeastPPI PPIAffinity BindingDB PDBBind DrugBank Davis")
configs="BACE|\
BBBP|\
CEP|\
ClinTox|\
Delaney|\
FreeSolv|\
HIV|\
Lipophilicity|\
Malaria|\
SIDER|\
Tox21|\
BetaLactamase|\
Fluorescence|\
Stability|\
Solubility|\
BinaryLocalization|\
SubcellularLocalization|\
EnzymeCommission|\
GeneOntologyMF|\
GeneOntologyBP|\
GeneOntologyCC|\
HumanPPI|\
YeastPPI|\
PPIAffinity|\
BindingDB|\
PDBBind|\
DrugBank|\
Davis"
IFS='|' read -ra config_array <<< "$configs"
eval "$(conda shell.bash hook)"
conda activate retd
export PYTHONPATH=$PYTHONPATH:$(pwd)
task_name=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python eval_tasks/prep_tasks_seqs.py --task_name "$task_name"
