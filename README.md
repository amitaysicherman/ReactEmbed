# ReactEmbed: Enhancing Protein and Molecule Representations with Biochemical Reactions Data

Official implementation of "ReactEmbed: A Cross-Domain Framework for Protein-Molecule Representation Learning via
Biochemical Reaction Networks".

![reaction](github_images/react.png)

## Overview

ReactEmbed is a novel method that enhances protein and molecule representations by integrating biochemical reaction data
with pre-trained embeddings. Our approach leverages biochemical reactions to capture functional context and dynamic
interactions between proteins and molecules, leading to more comprehensive and accurate representations.

## Installation

```bash
# Clone the repository
git clone https://github.com/amitaysicherman/ReactEmbed.git
cd ReactEmbed

# Install dependencies 
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

1. First download and preprocess the reaction data:

```bash
# Process Reactome data
python preprocessing/biopax_parser.py --data_name reactome

# Generate embeddings
python preprocessing/seq_to_vec.py --model ProtBert --data_name reactome
python preprocessing/seq_to_vec.py --model MolFormer --data_name reactome
```

### Training ReactEmbed Model

You can train the ReactEmbed model using the contrastive learning trainer:

```bash
python contrastive_learning/trainer.py \
    --batch_size 8192 \
    --p_model ProtBert \
    --m_model MolFormer \
    --shared_dim 256 \
    --n_layers 1 \
    --hidden_dim 512 \
    --dropout 0.0 \
    --epochs 10 \
    --lr 1e-4
```

Key training parameters:

- `batch_size`: Number of triplets per batch
- `p_model`: Protein model (ProtBert, ESM3-small, ESM3-medium, GearNet)
- `m_model`: Molecule model (MolFormer, MolCLR, ChemBERTa)
- `shared_dim`: Dimension of shared embedding space
- `n_layers`: Number of MLP layers
- `hidden_dim`: Hidden layer dimension
- `dropout`: Dropout rate
- `lr`: Learning rate

### Evaluating on Downstream Tasks

To evaluate the model on downstream tasks:

1. First prepare task-specific data:

```bash
# Prepare sequences
python eval_tasks/prep_tasks_seqs.py --task_name BBBP

# Prepare vectors
python eval_tasks/prep_tasks_vecs.py \
    --task_name BBBP \
    --p_model ProtBert \
    --m_model MolFormer
```

2. Run evaluation:

```bash
python eval_tasks/trainer.py \
    --use_fuse 1 \
    --use_model 1 \
    --bs 16 \
    --lr 0.001 \
    --drop_out 0.0 \
    --hidden_dim 512 \
    --task_name BBBP \
    --fusion_name "path/to/trained/model" \
    --m_model MolFormer \
    --p_model ProtBert \
    --n_layers 1 \
    --metric auc
```

Key evaluation parameters:

- `use_fuse`: Whether to use fused embeddings (1/0)
- `use_model`: Whether to use pre-trained embeddings (1/0)
- `bs`: Batch size
- `lr`: Learning rate
- `drop_out`: Dropout rate
- `hidden_dim`: Hidden dimension
- `task_name`: Name of evaluation task
- `fusion_name`: Path to trained ReactEmbed model
- `metric`: Evaluation metric (auc, rmse)

### End-to-End Pipeline

You can also run the complete pipeline using:

```bash 
python experiment_main.py \
    --p_model ProtBert \
    --m_model MolFormer \
    --data_name reactome \
    --cl_batch_size 8192 \
    --cl_n_layers 1 \
    --cl_hidden_dim 512 \
    --cl_dropout 0.0 \
    --cl_epochs 10 \
    --cl_lr 0.001 \
    --task_name BBBP \
    --task_metric auc
```

This will:

1. Preprocess the data if needed
2. Train ReactEmbed model
3. Evaluate on the specified task

## Supported Tasks

### Molecule Property Prediction

- BBBP (Blood-Brain Barrier Penetration)
- FreeSolv (Hydration Free Energy)
- Lipophilicity
- CEP (Clean Energy Project)

### Protein Property Prediction

- BetaLactamase
- Fluorescence
- Stability
- GeneOntology (MF/BP/CC)

### Interaction Prediction

- Drug-Target Interaction (DrugBank, Davis)
- Protein-Protein Interaction (HumanPPI, YeastPPI)
- Binding Affinity (BindingDB, PDBBind)


## License

This project is licensed under the MIT License.