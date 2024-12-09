from os.path import join as pjoin

import numpy as np
from torch.utils.data import Dataset, DataLoader

from common.path_manager import data_path


def split_train_val_test(data, val_size=0.16, test_size=0.20):
    train_val_index = int((1 - val_size - test_size) * len(data))
    val_test_index = int((1 - test_size) * len(data))
    train_data = data[:train_val_index]
    val_data = data[train_val_index:val_test_index]
    test_data = data[val_test_index:]
    return train_data, val_data, test_data


def load_data(task_name, mol_emd, protein_emd):
    base_dir = f"{data_path}/torchdrug/"
    data_file = pjoin(base_dir, f"{task_name}_{protein_emd}_{mol_emd}.npz")
    data = np.load(data_file)
    if task_name in ["DrugBank", "Davis", "KIBA"]:
        x1, x2, labels = [data[f"{x}"] for x in ["x1", "x2", "label"]]
        labels = labels.astype(np.float32).reshape(-1, 1)
        shuffle_index = np.random.permutation(len(x1))
        x1 = x1[shuffle_index]
        x2 = x2[shuffle_index]
        labels = labels[shuffle_index]
        x1_train, x1_valid, x1_test = split_train_val_test(x1)
        x2_train, x2_valid, x2_test = split_train_val_test(x2)
        labels_train, labels_valid, labels_test = split_train_val_test(labels)
    else:
        x1_train, x1_valid, x1_test = [data[f"x1_{x}"] for x in ["train", "valid", "test"]]
        if "x2_train" in data:
            x2_train, x2_valid, x2_test = [data[f"x2_{x}"] for x in ["train", "valid", "test"]]
        else:
            x2_train, x2_valid, x2_test = None, None, None
        labels_train, labels_valid, labels_test = [data[f"label_{x}"] for x in ["train", "valid", "test"]]
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


def get_dataloaders(task_name, mol_emd, protein_emd, batch_size):
    x1_train, x2_train, labels_train, x1_valid, x2_valid, labels_valid, x1_test, x2_test, labels_test = load_data(
        task_name, mol_emd, protein_emd)
    train_loader = DataLoader(TaskPrepDataset(x1_train, x2_train, labels_train), batch_size=batch_size, shuffle=False,
                              drop_last=False)
    valid_loader = DataLoader(TaskPrepDataset(x1_valid, x2_valid, labels_valid), batch_size=batch_size, shuffle=False,
                              drop_last=False)
    test_loader = DataLoader(TaskPrepDataset(x1_test, x2_test, labels_test), batch_size=batch_size, shuffle=False,
                             drop_last=False)
    return train_loader, valid_loader, test_loader
