import os
from os.path import join as pjoin

import numpy as np
from torchdrug.data import ordered_scaffold_split
from torchdrug.transforms import ProteinView
from tqdm import tqdm

from common.path_manager import data_path
from eval_tasks.models import DataType
from eval_tasks.tasks import Task, PrepType
from eval_tasks.tasks import name_to_task

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


def get_seq(x):
    try:
        # check if the sequence is a protein sequence have to_sequence method:
        if hasattr(x, "to_sequence"):
            return x.to_sequence().replace(".G", "")
        # check if the sequence is a molecule sequence have to_sequence method:
        else:
            return x.to_smiles()
    except Exception as e:
        print(e)
        return None


def split_train_val_test(data, val_size=0.16, test_size=0.20):
    train_val_index = int((1 - val_size - test_size) * len(data))
    val_test_index = int((1 - test_size) * len(data))
    train_data = data[:train_val_index]
    val_data = data[train_val_index:val_test_index]
    test_data = data[val_test_index:]
    return train_data, val_data, test_data

def prep_dataset(task: Task):
    output_base = pjoin(base_dir, task.name)
    os.makedirs(output_base, exist_ok=True)
    labels_file = pjoin(output_base, "train_labels.txt")
    if os.path.exists(labels_file):
        return

    if task.prep_type == PrepType.drugtarget:
        input_file = os.path.join(data_path, f"{task.name}.txt")
        if not os.path.exists(input_file):
            os.system(
                f"wget https://github.com/zhaoqichang/HpyerAttentionDTI/raw/main/data/{task.name}.txt -O {input_file}")

        with open(input_file) as f:
            lines = f.read().splitlines()
        x1 = []
        x2 = []
        labels = []
        for line in tqdm(lines):
            _, _, smiles, fasta, label = line.split(" ")
            x1.append(fasta)
            x2.append(smiles)
            labels.append(label)
        x1 = np.array(x1)
        x2 = np.array(x2)
        labels = np.array(labels)

        shuffle_index = np.random.permutation(len(labels))
        x1 = x1[shuffle_index]
        x2 = x2[shuffle_index]
        labels = labels[shuffle_index]
        x1_train, x1_valid, x1_test = split_train_val_test(x1)
        x2_train, x2_valid, x2_test = split_train_val_test(x2)
        labels_train, labels_valid, labels_test = split_train_val_test(labels)

        for split, name in zip([x1_train, x1_valid, x1_test], ["train", "valid", "test"]):
            with open(pjoin(output_base, f"{name}_1.txt"), "w") as f:
                f.write("\n".join(split))
        for split, name in zip([x2_train, x2_valid, x2_test], ["train", "valid", "test"]):
            with open(pjoin(output_base, f"{name}_2.txt"), "w") as f:
                f.write("\n".join(split))
        for split, name in zip([labels_train, labels_valid, labels_test], ["train", "valid", "test"]):
            with open(pjoin(output_base, f"{name}_labels.txt"), "w") as f:
                f.write("\n".join(split))



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
    for split, name in zip([train, valid, test], ["train", "valid", "test"]):
        all_seq_1 = []
        all_seq_2 = []
        all_labels = []
        for i in tqdm(range(len(split))):
            key1 = "graph" if task.dtype2 is None else "graph1"
            seq_1 = get_seq(split[i][key1])
            if seq_1 is None:
                continue
            all_seq_1.append(seq_1)
            if task.dtype2 is not None:
                seq_2 = get_seq(split[i]["graph2"])
                if seq_2 is None:
                    continue
                all_seq_2.append(seq_2)
            all_labels.append(" ".join([str(split[i][key]) for key in labels_keys]))
        with open(pjoin(output_base, f"{name}_1.txt"), "w") as f:
            f.write("\n".join(all_seq_1))
        if task.dtype2 is not None:
            with open(pjoin(output_base, f"{name}_2.txt"), "w") as f:
                f.write("\n".join(all_seq_2))
        with open(pjoin(output_base, f"{name}_labels.txt"), "w") as f:
            f.write("\n".join(all_labels))


def main(task_name):
    task = name_to_task[task_name]
    prep_dataset(task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="BACE")
    args = parser.parse_args()
    main(args.task_name)
