import os

import numpy as np
from torch.utils.data import Dataset, DataLoader

from common.path_manager import data_path
from eval_tasks.models import DataType
from eval_tasks.tasks import name_to_task


def load_data(task_name, mol_emd, protein_emd):
    base_dir = f"{data_path}/torchdrug/"
    task_dir = os.path.join(base_dir, task_name)

    def load_split(split, emb_name1, emb_name2=None):
        x1 = np.load(f"{task_dir}/{split}_{emb_name1}_1.npy")
        x2 = np.load(f"{task_dir}/{split}_{emb_name2}_2.npy") if emb_name2 else None
        labels = np.load(f"{task_dir}/{split}_labels.npy")
        if len(labels.shape) == 1:
            labels = labels[:, None]
        return x1, x2, labels

    task = name_to_task[task_name]
    emb1 = protein_emd if task.dtype1 == DataType.PROTEIN else mol_emd
    emb2 = None
    if task.dtype2:
        emb2 = protein_emd if task.dtype2 == DataType.PROTEIN else mol_emd

    x1_train, x2_train, labels_train = load_split('train', emb1, emb2)
    x1_valid, x2_valid, labels_valid = load_split('valid', emb1, emb2)
    x1_test, x2_test, labels_test = load_split('test', emb1, emb2)
    return x1_train, x2_train, labels_train, x1_valid, x2_valid, labels_valid, x1_test, x2_test, labels_test


class TaskPrepDataset(Dataset):
    def __init__(self, x1, x2, labels):
        self.x1 = np.nan_to_num(x1)
        self.x2 = np.nan_to_num(x2) if x2 is not None else None
        self.labels = np.nan_to_num(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.x2 is not None:
            return self.x1[idx], self.x2[idx], self.labels[idx]
        else:
            return self.x1[idx], self.labels[idx]


def get_dataloaders(task_name, mol_emd, protein_emd, batch_size, train_all_data=False):
    x1_train, x2_train, labels_train, x1_valid, x2_valid, labels_valid, x1_test, x2_test, labels_test = load_data(
        task_name, mol_emd, protein_emd)
    if train_all_data:
        print("Using all data for training")
        print(f"Train: {len(x1_train)}")
        x1_train = np.concatenate([x1_train, x1_valid, x1_test])
        print(f"Train: {len(x1_train)}")
        if x2_train is not None:
            x2_train = np.concatenate([x2_train, x2_valid, x2_test])
        labels_train = np.concatenate([labels_train, labels_valid, labels_test])
        print(f"Train: {len(labels_train)}")
        train_loader = DataLoader(TaskPrepDataset(x1_train, x2_train, labels_train), batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)
        return train_loader, None, None
    train_loader = DataLoader(TaskPrepDataset(x1_train, x2_train, labels_train), batch_size=batch_size, shuffle=True,
                              drop_last=False)
    valid_loader = DataLoader(TaskPrepDataset(x1_valid, x2_valid, labels_valid), batch_size=batch_size, shuffle=False,
                              drop_last=False)
    test_loader = DataLoader(TaskPrepDataset(x1_test, x2_test, labels_test), batch_size=batch_size, shuffle=False,
                             drop_last=False)
    return train_loader, valid_loader, test_loader
