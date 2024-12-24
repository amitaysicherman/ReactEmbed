#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH --array=1-116

# Define molecule and protein models
#molecule_models=("ChemBERTa" "MoLFormer" "MolCLR")
#protein_models=("ProtBert" "esm3-small" "esm3-medium" "GearNet")
#
## Step 1: Generate vectors for each molecule and protein model
#for m_model in "${molecule_models[@]}"; do
#    echo python preprocessing/seq_to_vec.py --model "$m_model"
#done
#for p_model in "${protein_models[@]}"; do
#    echo python preprocessing/seq_to_vec.py --model "$p_model"
#done
#
#mol_task=("BACE" "BBBP" "CEP" "ClinTox" "Delaney" "FreeSolv" "HIV" "Lipophilicity" "Malaria" "SIDER" "Tox21")
#for m_task in "${mol_task[@]}"; do
#  for m_model in "${molecule_models[@]}"; do
#    echo python eval_tasks/prep_tasks_vecs.py --task "$m_task" --m_model "$m_model"
#  done
#done
#
#prot_task=("BetaLactamase" "Fluorescence" "Stability" "BinaryLocalization" "HumanPPI" "YeastPPI" "PPIAffinity")
#for p_task in "${prot_task[@]}"; do
#  for p_model in "${protein_models[@]}"; do
#    echo python eval_tasks/prep_tasks_vecs.py --task "$p_task" --p_model "$p_model"
#  done
#done
#
#lig_task=("BindingDB" "PDBBind" "DrugBank" "Davis")
#for l_task in "${lig_task[@]}"; do
#  for p_model in "${protein_models[@]}"; do
#    for m_model in "${molecule_models[@]}"; do
#      echo python eval_tasks/prep_tasks_vecs.py --task "$l_task" --p_model "$p_model" --m_model "$m_model"
#    done
#  done
#done
commands="python preprocessing/seq_to_vec.py --model ChemBERTa|\
python preprocessing/seq_to_vec.py --model MoLFormer|\
python preprocessing/seq_to_vec.py --model MolCLR|\
python preprocessing/seq_to_vec.py --model ProtBert|\
python preprocessing/seq_to_vec.py --model esm3-small|\
python preprocessing/seq_to_vec.py --model esm3-medium|\
python preprocessing/seq_to_vec.py --model GearNet|\
python eval_tasks/prep_tasks_vecs.py --task BACE --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task BACE --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task BACE --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task BBBP --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task BBBP --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task BBBP --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task CEP --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task CEP --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task CEP --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task ClinTox --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task ClinTox --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task ClinTox --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task Delaney --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task Delaney --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task Delaney --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task FreeSolv --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task FreeSolv --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task FreeSolv --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task HIV --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task HIV --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task HIV --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task Lipophilicity --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task Lipophilicity --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task Lipophilicity --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task Malaria --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task Malaria --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task Malaria --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task SIDER --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task SIDER --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task SIDER --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task Tox21 --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task Tox21 --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task Tox21 --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task BetaLactamase --p_model ProtBert|\
python eval_tasks/prep_tasks_vecs.py --task BetaLactamase --p_model esm3-small|\
python eval_tasks/prep_tasks_vecs.py --task BetaLactamase --p_model esm3-medium|\
python eval_tasks/prep_tasks_vecs.py --task BetaLactamase --p_model GearNet|\
python eval_tasks/prep_tasks_vecs.py --task Fluorescence --p_model ProtBert|\
python eval_tasks/prep_tasks_vecs.py --task Fluorescence --p_model esm3-small|\
python eval_tasks/prep_tasks_vecs.py --task Fluorescence --p_model esm3-medium|\
python eval_tasks/prep_tasks_vecs.py --task Fluorescence --p_model GearNet|\
python eval_tasks/prep_tasks_vecs.py --task Stability --p_model ProtBert|\
python eval_tasks/prep_tasks_vecs.py --task Stability --p_model esm3-small|\
python eval_tasks/prep_tasks_vecs.py --task Stability --p_model esm3-medium|\
python eval_tasks/prep_tasks_vecs.py --task Stability --p_model GearNet|\
python eval_tasks/prep_tasks_vecs.py --task BinaryLocalization --p_model ProtBert|\
python eval_tasks/prep_tasks_vecs.py --task BinaryLocalization --p_model esm3-small|\
python eval_tasks/prep_tasks_vecs.py --task BinaryLocalization --p_model esm3-medium|\
python eval_tasks/prep_tasks_vecs.py --task BinaryLocalization --p_model GearNet|\
python eval_tasks/prep_tasks_vecs.py --task HumanPPI --p_model ProtBert|\
python eval_tasks/prep_tasks_vecs.py --task HumanPPI --p_model esm3-small|\
python eval_tasks/prep_tasks_vecs.py --task HumanPPI --p_model esm3-medium|\
python eval_tasks/prep_tasks_vecs.py --task HumanPPI --p_model GearNet|\
python eval_tasks/prep_tasks_vecs.py --task YeastPPI --p_model ProtBert|\
python eval_tasks/prep_tasks_vecs.py --task YeastPPI --p_model esm3-small|\
python eval_tasks/prep_tasks_vecs.py --task YeastPPI --p_model esm3-medium|\
python eval_tasks/prep_tasks_vecs.py --task YeastPPI --p_model GearNet|\
python eval_tasks/prep_tasks_vecs.py --task PPIAffinity --p_model ProtBert|\
python eval_tasks/prep_tasks_vecs.py --task PPIAffinity --p_model esm3-small|\
python eval_tasks/prep_tasks_vecs.py --task PPIAffinity --p_model esm3-medium|\
python eval_tasks/prep_tasks_vecs.py --task PPIAffinity --p_model GearNet|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model ProtBert --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model ProtBert --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model ProtBert --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model esm3-small --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model esm3-small --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model esm3-small --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model esm3-medium --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model esm3-medium --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model esm3-medium --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model GearNet --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model GearNet --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task BindingDB --p_model GearNet --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model ProtBert --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model ProtBert --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model ProtBert --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model esm3-small --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model esm3-small --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model esm3-small --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model esm3-medium --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model esm3-medium --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model esm3-medium --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model GearNet --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model GearNet --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task PDBBind --p_model GearNet --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model ProtBert --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model ProtBert --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model ProtBert --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model esm3-small --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model esm3-small --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model esm3-small --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model esm3-medium --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model esm3-medium --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model esm3-medium --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model GearNet --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model GearNet --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task DrugBank --p_model GearNet --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model ProtBert --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model ProtBert --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model ProtBert --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model esm3-small --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model esm3-small --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model esm3-small --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model esm3-medium --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model esm3-medium --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model esm3-medium --m_model MolCLR|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model GearNet --m_model ChemBERTa|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model GearNet --m_model MoLFormer|\
python eval_tasks/prep_tasks_vecs.py --task Davis --p_model GearNet --m_model MolCLR"



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