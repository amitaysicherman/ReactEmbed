#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-56

#DATA_NAMES_ARRAY=("pathbank" "reactome" "reactome_all")
#PROTEINS_MODELS_ARRAY=("GearNet" "ProtBert" "esm3-small" "esm3-medium")
#MOLECULES_MODELS_ARRAY=("MolCLR" "ChemBERTa" "MoLFormer")
#mol_task=("BACE" "BBBP" "CEP" "ClinTox" "Delaney" "FreeSolv" "HIV" "Lipophilicity" "Malaria" "SIDER" "Tox21")
#prot_task=("BetaLactamase" "Fluorescence" "Stability" "BinaryLocalization" "HumanPPI" "YeastPPI" "PPIAffinity")
#lig_task=("BindingDB" "PDBBind" "DrugBank" "Davis")
#all_tasks=("${mol_task[@]}" "${prot_task[@]}" "${lig_task[@]}")
#
#
##for data_name in "${DATA_NAMES_ARRAY[@]}" ; do
##  echo python preprocessing/biopax_parser.py --data_name $data_name
##done
#
##for task in "${all_tasks[@]}"; do
##    python eval_tasks/prep_tasks_seqs.py --task "$task"
##done
#
#
#for data_name in "${DATA_NAMES_ARRAY[@]}" ; do
#  for p_model in "${PROTEINS_MODELS_ARRAY[@]}" ;do
#    echo python preprocessing/seq_to_vec.py --data_name $data_name --model $p_model
#  done
#  for m_model in "${MOLECULES_MODELS_ARRAY[@]}" ;do
#    echo python preprocessing/seq_to_vec.py --data_name $data_name --model $m_model
#  done
#  for p_model in "${PROTEINS_MODELS_ARRAY[@]}" ; do
#    for m_model in "${MOLECULES_MODELS_ARRAY[@]}" ; do
#      echo python contrastive_learning/trainer.py --data_name $data_name --p_model $p_model --m_model $m_model
#    done
#  done
#done
#
#
#for m_model in "${molecule_models[@]}"; do
#    echo python preprocessing/seq_to_vec.py --model "$m_model"
#done
#for p_model in "${protein_models[@]}"; do
#    echo python preprocessing/seq_to_vec.py --model "$p_model"
#done
#
#for m_task in "${mol_task[@]}"; do
#  for m_model in "${molecule_models[@]}"; do
#    echo python eval_tasks/prep_tasks_vecs.py --task "$m_task" --m_model "$m_model"
#  done
#done
#
#for p_task in "${prot_task[@]}"; do
#  for p_model in "${protein_models[@]}"; do
#    echo python eval_tasks/prep_tasks_vecs.py --task "$p_task" --p_model "$p_model"
#  done
#done
#
#for l_task in "${lig_task[@]}"; do
#  for p_model in "${protein_models[@]}"; do
#    for m_model in "${molecule_models[@]}"; do
#      echo python eval_tasks/prep_tasks_vecs.py --task "$l_task" --p_model "$p_model" --m_model "$m_model"
#    done
#  done
#done

commands="python preprocessing/seq_to_vec.py --data_name pathbank --model GearNet|\
python preprocessing/seq_to_vec.py --data_name pathbank --model ProtBert|\
python preprocessing/seq_to_vec.py --data_name pathbank --model esm3-small|\
python preprocessing/seq_to_vec.py --data_name pathbank --model esm3-medium|\
python preprocessing/seq_to_vec.py --data_name pathbank --model MolCLR|\
python preprocessing/seq_to_vec.py --data_name pathbank --model ChemBERTa|\
python preprocessing/seq_to_vec.py --data_name pathbank --model MoLFormer|\
python contrastive_learning/trainer.py --data_name pathbank --p_model GearNet --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name pathbank --p_model GearNet --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name pathbank --p_model GearNet --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name pathbank --p_model ProtBert --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name pathbank --p_model ProtBert --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name pathbank --p_model ProtBert --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name pathbank --p_model esm3-small --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name pathbank --p_model esm3-small --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name pathbank --p_model esm3-small --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name pathbank --p_model esm3-medium --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name pathbank --p_model esm3-medium --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name pathbank --p_model esm3-medium --m_model MoLFormer|\
python preprocessing/seq_to_vec.py --data_name reactome --model GearNet|\
python preprocessing/seq_to_vec.py --data_name reactome --model ProtBert|\
python preprocessing/seq_to_vec.py --data_name reactome --model esm3-small|\
python preprocessing/seq_to_vec.py --data_name reactome --model esm3-medium|\
python preprocessing/seq_to_vec.py --data_name reactome --model MolCLR|\
python preprocessing/seq_to_vec.py --data_name reactome --model ChemBERTa|\
python preprocessing/seq_to_vec.py --data_name reactome --model MoLFormer|\
python contrastive_learning/trainer.py --data_name reactome --p_model GearNet --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name reactome --p_model GearNet --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name reactome --p_model GearNet --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name reactome --p_model ProtBert --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name reactome --p_model ProtBert --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name reactome --p_model ProtBert --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name reactome --p_model esm3-small --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name reactome --p_model esm3-small --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name reactome --p_model esm3-small --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name reactome --p_model esm3-medium --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name reactome --p_model esm3-medium --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name reactome --p_model esm3-medium --m_model MoLFormer|\
python preprocessing/seq_to_vec.py --data_name reactome_all --model GearNet|\
python preprocessing/seq_to_vec.py --data_name reactome_all --model ProtBert|\
python preprocessing/seq_to_vec.py --data_name reactome_all --model esm3-small|\
python preprocessing/seq_to_vec.py --data_name reactome_all --model esm3-medium|\
python preprocessing/seq_to_vec.py --data_name reactome_all --model MolCLR|\
python preprocessing/seq_to_vec.py --data_name reactome_all --model ChemBERTa|\
python preprocessing/seq_to_vec.py --data_name reactome_all --model MoLFormer|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model GearNet --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model GearNet --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model GearNet --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model ProtBert --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model ProtBert --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model ProtBert --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model esm3-small --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model esm3-small --m_model ChemBERTa|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model esm3-small --m_model MoLFormer|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model esm3-medium --m_model MolCLR|\
python contrastive_learning/trainer.py --data_name reactome_all --p_model esm3-medium --m_model ChemBERTa"


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