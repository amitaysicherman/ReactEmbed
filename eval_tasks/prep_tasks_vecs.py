# sbatch --time=1-0 --array=1-21 --gres=gpu:A40:1 --mem=64G -c 4 --requeue --wrap="python3 GO/preprocessing.py --task_index $SLURM_ARRAY_TASK_ID-1"
import os
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

from common.path_manager import data_path
from eval_tasks.models import DataType
from eval_tasks.tasks import Task, PrepType
from eval_tasks.tasks import name_to_task
from preprocessing.seq_to_vec import SeqToVec

base_dir = f"{data_path}/torchdrug/"
os.makedirs(base_dir, exist_ok=True)


def prep_dataset(task: Task, p_seq2vec, m_seq2vec, protein_emd, mol_emd):
    task_dir = pjoin(base_dir, task.name)
    output_file = pjoin(task_dir, f"{protein_emd}_{mol_emd}.npz")
    if os.path.exists(output_file):
        return

    if task.prep_type == PrepType.drugtarget:
        with open(pjoin(task_dir, "1.txt"), "r") as f:
            lines = f.read().splitlines()
        proteins = [p_seq2vec.to_vec(line) for line in tqdm(lines)]
        proteins = np.stack(proteins)

        with open(pjoin(task_dir, "2.txt"), "r") as f:
            lines = f.read().splitlines()
        molecules = [m_seq2vec.to_vec(line) for line in tqdm(lines)]
        molecules = np.stack(molecules)
        with open(pjoin(task_dir, "labels.txt"), "r") as f:
            lines = f.read().splitlines()
        labels = np.array([float(line) for line in tqdm(lines)])
        np.savez(output_file, x1=proteins, x2=molecules, label=labels)
        return

    x1_all = dict()
    x2_all = dict()
    labels_all = dict()
    for name in ["train", "valid", "test"]:
        with open(pjoin(task_dir, f"{name}_1.txt"), "r") as f:
            x1_lines = f.read().splitlines()
        x1_vecs = [p_seq2vec.to_vec(line) if task.dtype1 == DataType.PROTEIN else m_seq2vec.to_vec(line) for line in
                   tqdm(x1_lines)]
        x1_vecs = np.stack(x1_vecs)
        if task.dtype2 is not None:
            with open(pjoin(task_dir, f"{name}_2.txt"), "r") as f:
                x2_lines = f.read().splitlines()
            x2_vecs = [p_seq2vec.to_vec(line) if task.dtype2 == DataType.PROTEIN else m_seq2vec.to_vec(line) for line in
                       tqdm(x2_lines)]
            x2_vecs = np.stack(x2_vecs)
        else:
            x2_vecs = []
        with open(pjoin(task_dir, f"{name}_labels.txt"), "r") as f:
            labels_lines = f.read().splitlines()
        labels = [np.array([float(x) for x in line.split()]) for line in labels_lines]
        labels = np.stack(labels)

        x1_all[f'x1_{name}'] = np.array(x1_vecs)
        if len(x2_vecs):
            x2_all[f'x2_{name}'] = np.array(x2_vecs)
        labels_all[f'label_{name}'] = np.array(labels)
    if len(x2_all):
        np.savez(output_file, **x1_all, **x2_all, **labels_all)
    else:
        np.savez(output_file, **x1_all, **labels_all)


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
