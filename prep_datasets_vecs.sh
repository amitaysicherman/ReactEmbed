#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-14

commands="python preprocessing/seq_to_vec.py --model ChemBERTa --data_name pathbank|\
python preprocessing/seq_to_vec.py --model ChemBERTa --data_name reactome|\
python preprocessing/seq_to_vec.py --model MoLFormer --data_name pathbank|\
python preprocessing/seq_to_vec.py --model MoLFormer --data_name reactome|\
python preprocessing/seq_to_vec.py --model MolCLR --data_name pathbank|\
python preprocessing/seq_to_vec.py --model MolCLR --data_name reactome|\
python preprocessing/seq_to_vec.py --model ProtBert --data_name pathbank|\
python preprocessing/seq_to_vec.py --model ProtBert --data_name reactome|\
python preprocessing/seq_to_vec.py --model esm3-small --data_name pathbank|\
python preprocessing/seq_to_vec.py --model esm3-small --data_name reactome|\
python preprocessing/seq_to_vec.py --model esm3-medium --data_name pathbank|\
python preprocessing/seq_to_vec.py --model esm3-medium --data_name reactome|\
python preprocessing/seq_to_vec.py --model GearNet --data_name pathbank|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome"


IFS='|' read -r -a array <<< "$commands"
cmd=${array[$((SLURM_ARRAY_TASK_ID - 1))]}
# if esm in cmd active cond env ReactEmbedESM else ReactEmbedTorchDrug
if [[ $cmd == *"esm"* ]]; then
    conda activate ReactEmbedESM
else
    conda activate ReactEmbedTorchDrug
fi
echo $cmd
export PYTHONPATH=$PYTHONPATH:$(pwd)
eval $cmd