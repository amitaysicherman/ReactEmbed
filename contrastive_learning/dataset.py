import os
import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from common.path_manager import item_path, reactions_file

TYPES = ["P-P", "P-M", "M-P", "M-M"]


# TYPES = ["P-P", "M-P"]  # Proteins anchor the triplets


def prep_entity(entities, empty_list):
    if entities == "" or entities == " ":
        return []
    else:
        return [int(x) for x in entities.split(",") if int(x) not in empty_list]


class TripletsDataset(Dataset):
    def __init__(self, split, p_model="ProtBert", m_model="MoLFormer", n_duplicates=10, flip_prob=0):
        self.split = split
        self.flip_prob = flip_prob
        self.proteins = np.load(pjoin(item_path, f"{p_model}_vectors.npy"))
        self.molecules = np.load(pjoin(item_path, f"{m_model}_vectors.npy"))
        # all the proteins with zero vectors are empty proteins
        self.empty_protein_index = set(np.where(np.all(self.proteins == 0, axis=1))[0].tolist())
        self.empty_molecule_index = set(np.where(np.all(self.molecules == 0, axis=1))[0].tolist())
        print(f"Empty proteins: {len(self.empty_protein_index)}")
        print(f"Empty molecules: {len(self.empty_molecule_index)}")
        self.types = TYPES

        loaded = self.load_triples_if_exists()
        if loaded:
            return

        self.pairs = {t: set() for t in TYPES}

        with open(reactions_file) as f:
            lines = f.read().splitlines()
        for line in tqdm(lines):
            if line.startswith(" "):
                proteins, molecules = "", line
            elif line.endswith(" "):
                proteins, molecules = line, ""
            else:
                proteins, molecules = line.split()
            proteins = prep_entity(proteins, self.empty_protein_index)
            molecules = prep_entity(molecules, self.empty_molecule_index)
            types = ["P"] * len(proteins) + ["M"] * len(molecules)
            elements = proteins + molecules
            for i, e1 in enumerate(elements):
                for j, e2 in enumerate(elements[i + 1:], start=i + 1):
                    t = types[i] + "-" + types[j]
                    self.pairs[t].add((e1, e2))
                    if t == "P-M":
                        self.pairs["M-P"].add((e2, e1))

        self.split_pair = {}
        for t in TYPES:
            t_pairs = list(self.pairs[t])
            t_pairs.sort()
            random.seed(42)
            random.shuffle(t_pairs)
            if self.split == "all":
                self.split_pair[t] = t_pairs
            elif self.split == "train":
                self.split_pair[t] = t_pairs[:int(len(t_pairs) * 0.8)]
            elif self.split == "valid":
                self.split_pair[t] = t_pairs[int(len(t_pairs) * 0.8):int(len(t_pairs) * 0.9)]
            elif self.split == "test":
                self.split_pair[t] = t_pairs[int(len(t_pairs) * 0.9):]
            else:
                raise ValueError("Unknown split")

        self.triples = {t: set() for t in TYPES}
        for t in TYPES:
            type1, type2 = t.split("-")
            for e1, e2 in tqdm(self.split_pair[t]):
                for _ in range(n_duplicates):
                    e3 = self.sample_neg_element(e1, type1, type2)
                    if self.flip_prob > 0 and random.random() < self.flip_prob:
                        self.triples[t].add((e1, e3, e2))
                    else:
                        self.triples[t].add((e1, e2, e3))
        self.triples = {t: list(self.triples[t]) for t in TYPES}
        # shuffle the triples
        for t in TYPES:
            random.seed(42)
            random.shuffle(self.triples[t])
        for t in TYPES:
            print(f"Number of {t} triples: {len(self.triples[t])}")
        self.save_triples()
    def sample_neg_element(self, e1, e1_type, e2_type):
        is_positive, is_empty = True, True
        e3 = None
        while is_positive or is_empty:
            if e2_type == "P":
                e3 = random.choice(list(range(len(self.proteins))))
                is_empty = e3 in self.empty_protein_index
                is_positive = (e1, e3) in self.pairs[f"{e1_type}-{e2_type}"]
            else:
                e3 = random.choice(list(range(len(self.molecules))))
                is_empty = e3 in self.empty_molecule_index
                is_positive = (e1, e3) in self.pairs[f"{e1_type}-{e2_type}"]
        return e3

    def __len__(self):
        return sum(len(self.triples[t]) for t in TYPES)

    def type_to_start_index(self, t):
        return sum(len(self.triples[x]) for x in TYPES[:TYPES.index(t)])

    def idx_type_to_vec(self, idx, t):
        if t == "P":
            return torch.tensor(self.proteins[idx]).float()
        return torch.tensor(self.molecules[idx]).float()

    def __len__(self):
        return sum(len(self.triples[t]) for t in TYPES)

    def __getitem__(self, t_idx):
        t, idx = t_idx
        e1, e2, e3 = self.triples[t][idx]
        t1, t2 = t.split("-")
        v1, v2, v3 = self.idx_type_to_vec(e1, t1), self.idx_type_to_vec(e2, t2), self.idx_type_to_vec(e3, t2)
        return t1, t2, v1, v2, v3

    def save_triples(self):
        base_dir = pjoin(item_path, "triplets")
        os.makedirs(base_dir, exist_ok=True)
        for t in TYPES:
            with open(pjoin(base_dir, f"{t}_{self.split}.txt"), "w") as f:
                for e1, e2, e3 in self.triples[t]:
                    f.write(f"{e1} {e2} {e3}\n")

    def load_triples_if_exists(self):
        base_dir = pjoin(item_path, "triplets")
        items_file = pjoin(base_dir, f"{TYPES[0]}_{self.split}.txt")
        if not os.path.exists(items_file):
            return False
        self.triples = {t: [] for t in TYPES}
        for t in TYPES:
            with open(pjoin(base_dir, f"{t}_{self.split}.txt")) as f:
                for line in f:
                    e1, e2, e3 = line.split()
                    self.triples[t].append((int(e1), int(e2), int(e3)))
        for t in TYPES:
            print(f"Number of {t} triples: {len(self.triples[t])}")
        return True


class TripletsBatchSampler(Sampler):
    def __init__(self, dataset: TripletsDataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for t in self.dataset.types:
            indices = list(range(len(self.dataset.triples[t])))
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield [(t, idx) for idx in indices[i:i + self.batch_size]]

    def __len__(self):
        return len(self.dataset) // self.batch_size
