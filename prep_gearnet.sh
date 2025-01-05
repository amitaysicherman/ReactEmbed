#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-4

commands="python eval_tasks/prep_gearnet.py --task_name EnzymeCommission|\
python eval_tasks/prep_gearnet.py --task_name GeneOntologyMF|\
python eval_tasks/prep_gearnet.py --task_name GeneOntologyBP|\
python eval_tasks/prep_gearnet.py --task_name GeneOntologyCC"

IFS='|' read -r -a array <<< "$commands"
cmd=${array[$((SLURM_ARRAY_TASK_ID - 1))]}


# if esm in cmd active cond env ReactEmbedESM else ReactEmbedTorchDrug
eval "$(conda shell.bash hook)"

conda activate retd
echo $cmd
export PYTHONPATH=$PYTHONPATH:$(pwd)
eval $cmd