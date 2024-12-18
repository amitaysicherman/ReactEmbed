from os.path import join as pjoin

import numpy as np
from torch.utils.data import Dataset, DataLoader

from common.path_manager import data_path


def load_data(task_name, mol_emd, protein_emd):
    base_dir = f"{data_path}/torchdrug/"
    data_file = pjoin(base_dir, f"{task_name}/{protein_emd}_{mol_emd}.npz")
    data = np.load(data_file)
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
