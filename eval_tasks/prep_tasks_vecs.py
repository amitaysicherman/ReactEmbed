# sbatch --time=1-0 --array=1-21 --gres=gpu:A40:1 --mem=64G -c 4 --requeue --wrap="python3 GO/preprocessing.py --task_index $SLURM_ARRAY_TASK_ID-1"
import os

import numpy as np
from tqdm import tqdm

from common.path_manager import data_path
from eval_tasks.models import DataType
from eval_tasks.tasks import name_to_task
from preprocessing.seq_to_vec import SeqToVec

base_dir = f"{data_path}/torchdrug/"
os.makedirs(base_dir, exist_ok=True)


def save_vectors(input_file, output_file, converter):
    if os.path.exists(output_file):
        return
    with open(input_file, 'r') as f:
        lines = f.read().splitlines()
    vectors = converter.lines_to_vecs(lines)
    np.save(output_file, vectors)


def save_labels(input_file, output_file):
    if os.path.exists(output_file):
        return
    with open(input_file, 'r') as f:
        lines = f.read().splitlines()
    lines = [line.split() for line in lines]
    labels = np.stack([np.array([float(label) for label in line]) for line in tqdm(lines)])
    # replace nan with 0
    labels = np.nan_to_num(labels)
    if np.all(labels == labels.astype(int)):  # if classificaiton , convert to one hot
        if labels.max() > 1:
            # for example - 10 classes and the label is just the index of the class. convert to one hot
            labels = np.eye(labels.max() + 1)[labels.astype(int)]
    np.save(output_file, labels)


def prep_dataset(task, p_seq2vec, m_seq2vec, protein_emd, mol_emd):
    task_dir = os.path.join(base_dir, task.name)
    converter = p_seq2vec if task.dtype1 == DataType.PROTEIN else m_seq2vec
    emb_name = protein_emd if task.dtype1 == DataType.PROTEIN else mol_emd

    for split in ['train', 'valid', 'test']:
        save_vectors(f"{task_dir}/{split}_1.txt", f"{task_dir}/{split}_{emb_name}_1.npy", converter)

        if task.dtype2:
            converter2 = p_seq2vec if task.dtype2 == DataType.PROTEIN else m_seq2vec
            emb_name2 = protein_emd if task.dtype2 == DataType.PROTEIN else mol_emd
            save_vectors(f"{task_dir}/{split}_2.txt", f"{task_dir}/{split}_{emb_name2}_2.npy", converter2)

        save_labels(f"{task_dir}/{split}_labels.txt", f"{task_dir}/{split}_labels.npy")


def main(task_name, p_model, m_model):
    task = name_to_task[task_name]
    p_seq2vec = SeqToVec(p_model)
    m_seq2vec = SeqToVec(m_model)
    prep_dataset(task, p_seq2vec, m_seq2vec, p_model, m_model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="BACE")
    parser.add_argument('--p_model', type=str, help='Protein model', default="ProtBert")
    parser.add_argument('--m_model', type=str, help='Molecule model', default="ChemBERTa")
    args = parser.parse_args()
    main(args.task_name, args.p_model, args.m_model)
