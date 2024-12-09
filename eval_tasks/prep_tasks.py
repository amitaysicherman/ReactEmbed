# sbatch --time=1-0 --array=1-21 --gres=gpu:A40:1 --mem=64G -c 4 --requeue --wrap="python3 GO/preprocessing.py --task_index $SLURM_ARRAY_TASK_ID-1"
import os
from os.path import join as pjoin

import numpy as np
from torchdrug.data import ordered_scaffold_split
from torchdrug.transforms import ProteinView
from tqdm import tqdm

from common.path_manager import data_path
from eval_tasks.models import DataType
from eval_tasks.tasks import Task, PrepType
from preprocessing.seq_to_vec import SeqToVec

base_dir = f"{data_path}/torchdrug/"
os.makedirs(base_dir, exist_ok=True)
SIDER_LABELS = ['Hepatobiliary disorders',
                'Metabolism and nutrition disorders',
                'Product issues',
                'Eye disorders',
                'Investigations',
                'Musculoskeletal and connective tissue disorders',
                'Gastrointestinal disorders',
                'Social circumstances',
                'Immune system disorders',
                'Reproductive system and breast disorders',
                'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                'General disorders and administration site conditions',
                'Endocrine disorders',
                'Surgical and medical procedures',
                'Vascular disorders',
                'Blood and lymphatic system disorders',
                'Skin and subcutaneous tissue disorders',
                'Congenital, familial and genetic disorders',
                'Infections and infestations',
                'Respiratory, thoracic and mediastinal disorders',
                'Psychiatric disorders',
                'Renal and urinary disorders',
                'Pregnancy, puerperium and perinatal conditions',
                'Ear and labyrinth disorders',
                'Cardiac disorders',
                'Nervous system disorders',
                'Injury, poisoning and procedural complications']


def get_vec(seq2vec, x):
    try:
        # check if the sequence is a protein sequence have to_sequence method:
        if hasattr(x, "to_sequence"):
            return seq2vec.to_vec(x.to_sequence().replace(".G", ""))
        # check if the sequence is a molecule sequence have to_sequence method:
        else:
            seq2vec.to_vec(x.to_smiles())
    except Exception as e:
        print(e)
        return None


def prep_dataset(task: Task, p_seq2vec, m_seq2vec, protein_emd, mol_emd):
    output_file = pjoin(base_dir, f"{task.name}_{protein_emd}_{mol_emd}.npz")

    if os.path.exists(output_file):
        return

    if task.prep_type == PrepType.drugtarget:
        input_file = os.path.join(data_path, f"{task.name}.txt")
        if not os.path.exists(input_file):
            os.system(
                f"wget https://github.com/zhaoqichang/HpyerAttentionDTI/raw/main/data/{task.name}.txt -O {input_file}")

        with open(input_file) as f:
            lines = f.read().splitlines()
        proteins = []
        molecules = []
        labels = []
        for line in tqdm(lines):
            _, _, smiles, fasta, label = line.split(" ")
            proteins.append(p_seq2vec.to_vec(fasta))
            molecules.append(m_seq2vec.to_vec(smiles))
            labels.append(label)
        np.savez(output_file, x1=np.array(proteins)[:, 0, :], x2=np.array(molecules)[:, 0, :], label=np.array(labels))
        return

    if task.dtype1 == DataType.PROTEIN:
        if task.dtype2 is None:
            keys = ["graph"]
        elif task.dtype2 == DataType.MOLECULE:
            keys = ["graph1"]
        else:
            keys = ["graph1", "graph2"]

        args = dict(transform=ProteinView(view="residue", keys=keys),
                    atom_feature=None, bond_feature=None)
    else:
        args = dict()
    dataset = task.dataset(pjoin(base_dir, task.name), **args)
    labels_keys = getattr(task.dataset, 'target_fields')
    if task.name == "SIDER":
        labels_keys = SIDER_LABELS
    if hasattr(task.dataset, "splits"):
        splits = dataset.split()
        if len(splits) == 3:
            train, valid, test = splits
        elif len(splits) > 3:
            train, valid, test, *unused_test = splits
        else:
            raise Exception("splits", getattr(task.dataset, "splits"))

    else:
        train, valid, test = ordered_scaffold_split(dataset, None)
    x1_all = dict()
    x2_all = dict()
    labels_all = dict()
    for split, name in zip([train, valid, test], ["train", "valid", "test"]):
        x1_vecs = []
        x2_vecs = []
        labels = []
        for i in tqdm(range(len(split))):
            key1 = "graph" if task.dtype2 is None else "graph1"

            new_vec = get_vec(p_seq2vec if task.dtype1 == DataType.PROTEIN else m_seq2vec, split[i][key1])
            if new_vec is None:
                continue
            if task.dtype2 is not None:
                new_vec_2 = get_vec(p_seq2vec if task.dtype2 == DataType.PROTEIN else m_seq2vec, split[i]["graph2"])
                if new_vec_2 is None:
                    continue
                x2_vecs.append(new_vec_2)
            x1_vecs.append(new_vec)

            label = [split[i][key] for key in labels_keys]
            labels.append(label)
        x2_vecs = np.array(x2_vecs)

        x1_all[f'x1_{name}'] = np.array(x1_vecs)
        if len(x2_vecs):
            x2_all[f'x2_{name}'] = np.array(x2_vecs)[:, 0, :]
        labels_all[f'label_{name}'] = np.array(labels)
    if len(x2_all):
        np.savez(output_file, **x1_all, **x2_all, **labels_all)
    else:
        np.savez(output_file, **x1_all, **labels_all)


if __name__ == "__main__":
    import argparse
    from eval_tasks.tasks import name_to_task

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="SIDER")
    parser.add_argument('--p_model', type=str, help='Protein model', default="ProtBert")
    parser.add_argument('--m_model', type=str, help='Molecule model', default="ChemBERTa")
    parser.add_argument("--auth_token", type=str, default="")
    args = parser.parse_args()
    task = name_to_task[args.task_name]
    p_seq2vec = SeqToVec(args.p_model)
    m_seq2vec = SeqToVec(args.m_model)
    prep_dataset(task, p_seq2vec, m_seq2vec, args.p_model, args.m_model)
