#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=14-34

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
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 0 --end_index 500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 500 --end_index 1000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 1000 --end_index 1500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 1500 --end_index 2000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 2000 --end_index 2500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 2500 --end_index 3000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 3000 --end_index 3500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 3500 --end_index 4000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 4000 --end_index 4500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 4500 --end_index 5000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 5000 --end_index 5500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 5500 --end_index 6000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 6000 --end_index 6500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 6500 --end_index 7000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 7000 --end_index 7500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 7500 --end_index 8000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 8000 --end_index 8500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 8500 --end_index 9000|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 9000 --end_index 9500|\
python preprocessing/seq_to_vec.py --model GearNet --data_name reactome --start_index 9500 --end_index 10000"




IFS='|' read -r -a array <<< "$commands"
cmd=${array[$((SLURM_ARRAY_TASK_ID - 1))]}
# if esm in cmd active cond env ReactEmbedESM else ReactEmbedTorchDrug

eval "$(conda shell.bash hook)"


if [[ $cmd == *"esm"* ]]; then
    conda activate ReactEmbedESM
else
    conda activate ReactEmbedTorchDrug
fi
echo $cmd
export PYTHONPATH=$PYTHONPATH:$(pwd)
eval $cmd