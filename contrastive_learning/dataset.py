import random
from collections import Counter
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

# TYPES = ["P-P", "P-M", "M-P", "M-M"]

TYPES = ["M", "P"]

PAIR_TYPES = ["P-P", "P-M", "M-P", "M-M"]
TRIPLET_TYPES = ["P-P-M", "P-M-P", "M-P-P", "M-M-P", "P-P-P", "M-M-M", "M-P-M", "P-M-M"]


def print_hist_as_csv(ags):
    hist, bins = ags
    print("bin, count")
    for i, c in enumerate(hist):
        print(f"{(bins[i + 1] + bins[i]) / 2:.2f}, {c:,}")

def prep_entity(entities, empty_list):
    if entities == "" or entities == " ":
        return []
    else:
        return [int(x) for x in entities.split(",") if int(x) not in empty_list]


class TripletsDataset(Dataset):
    def __init__(self, data_name, split, p_model="ProtBert", m_model="ChemBERTa", n_duplicates=10, flip_prob=0,
                 samples_ratio=1, no_pp_mm=0):
        self.split = split

        self.flip_prob = flip_prob
        self.item_path = f"data/{data_name}"
        reactions_file = pjoin(self.item_path, "reaction.txt")
        self.proteins = np.load(pjoin(self.item_path, f"{p_model}_vectors.npy"))
        self.molecules = np.load(pjoin(self.item_path, f"{m_model}_vectors.npy"))
        self.empty_protein_index = set(np.where(np.all(self.proteins == 0, axis=1))[0].tolist())
        self.empty_molecule_index = set(np.where(np.all(self.molecules == 0, axis=1))[0].tolist())
        self.proteins_non_empty = [i for i in range(len(self.proteins)) if i not in self.empty_protein_index]
        self.molecules_non_empty = [i for i in range(len(self.molecules)) if i not in self.empty_molecule_index]
        print(f"Empty proteins: {len(self.empty_protein_index)}")
        print(f"Empty molecules: {len(self.empty_molecule_index)}")
        print("Not empty proteins:", len(self.proteins_non_empty))
        print("Not empty molecules:", len(self.molecules_non_empty))
        self.types = PAIR_TYPES
        self.pair_counts = {t: Counter() for t in self.types}
        self.source_edge_counter_p = Counter()
        self.source_edge_counter_m = Counter()
        # Count pair frequencies
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
                    if types[i] == "P":
                        self.source_edge_counter_p[e1] += 1
                    else:
                        self.source_edge_counter_m[e1] += 1
                    if types[j] == "P":
                        self.source_edge_counter_p[e2] += 1
                    else:
                        self.source_edge_counter_m[e2] += 1

                    if no_pp_mm == 1 and types[i] == types[j]:
                        continue
                    self.pair_counts[f"{types[i]}-{types[j]}"][(e1, e2)] += 1
                    self.pair_counts[f"{types[j]}-{types[i]}"][(e2, e1)] += 1
        # print molecule and protein source edge counts
        print_hist_as_csv(np.histogram(list(self.source_edge_counter_p.values())))
        print_hist_as_csv(np.histogram(list(self.source_edge_counter_m.values())))


        for t in self.pair_counts:
            print(f"Number of {t} pairs: {len(self.pair_counts[t]):,}")
            # print sum of all pairs count per type
            print(f"Sum of {t} pairs: {sum(self.pair_counts[t].values()):,}")
            counts = np.array(list(self.pair_counts[t].values()))
            counts = np.clip(counts, np.quantile(counts, 0.10), np.quantile(counts, 0.90))
            print_hist_as_csv(np.histogram(counts, bins=10))

        # Split the valid pairs
        self.split_pair = {}
        for t in self.types:
            t_pairs = list(self.pair_counts[t].keys())
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

        self.triples = {t: set() for t in TRIPLET_TYPES}
        for t in self.types:
            t1, t2 = t.split("-")
            ttag = "P" if t2 == "M" else "M"
            for e1, e2 in tqdm(self.split_pair[t], desc=f"Generating {t} triplets"):
                if samples_ratio < 1 and random.random() > samples_ratio:
                    continue
                for _ in range(n_duplicates):
                    pair_count = self.pair_counts[t][(e1, e2)]
                    pair_count = min(pair_count, 10)
                    for _ in range(pair_count):
                        e3_a = self.sample_neg_element(e1, t1, t2)
                        e3_b = self.sample_neg_element(e1, t1, ttag)

                        if self.flip_prob > 0 and random.random() < self.flip_prob:
                            self.triples[f"{t1}-{t2}-{t2}"].add((e1, e3_a, e2))
                            self.triples[f"{t1}-{ttag}-{t2}"].add((e1, e3_b, e2))
                        else:
                            self.triples[f"{t1}-{t2}-{t2}"].add((e1, e2, e3_a))
                            self.triples[f"{t1}-{t2}-{ttag}"].add((e1, e2, e3_b))
        self.triples = {t: list(self.triples[t]) for t in TRIPLET_TYPES}
        # shuffle the triples
        for t in self.triples:
            random.seed(42)
            random.shuffle(self.triples[t])
            print(f"Number of {t} triples: {len(self.triples[t])}")

    def sample_neg_element(self, e1, e1_type, e2_type):
        """Sample negative element that has never appeared with e1"""
        pair_type = f"{e1_type}-{e2_type}"
        while True:
            if e2_type == "P":
                e3 = random.choice(self.proteins_non_empty)
            else:
                e3 = random.choice(self.molecules_non_empty)
            if (e1, e3) not in self.pair_counts[pair_type]:
                return e3

    def __len__(self):
        return sum(len(self.triples[t]) for t in self.triples)

    def type_to_start_index(self, t):
        return sum(len(self.triples[x]) for x in TRIPLET_TYPES[:self.types.index(t)])

    def idx_type_to_vec(self, idx, t):
        if t == "P":
            return torch.tensor(self.proteins[idx]).float()
        return torch.tensor(self.molecules[idx]).float()

    def __getitem__(self, t_idx):
        t, idx = t_idx
        e1, e2, e3 = self.triples[t][idx]
        t1, t2, t3 = t.split("-")
        v1, v2, v3 = self.idx_type_to_vec(e1, t1), self.idx_type_to_vec(e2, t2), self.idx_type_to_vec(e3, t3)
        return t1, t2, t3, v1, v2, v3


class TripletsBatchSampler(Sampler):
    def __init__(self, dataset: TripletsDataset, batch_size, max_num_steps=5_000):
        self.dataset = dataset
        self.batch_size = batch_size
        max_len = max(len(self.dataset.triples[t]) for t in self.dataset.triples)
        print(f"Max length: {max_len}")
        # self.types_upsample = {t: max_len // len(self.dataset.triples[t]) for t in self.dataset.triples}
        self.max_num_steps = max_num_steps

    def __iter__(self):
        for _ in range(self.max_num_steps):
            t = random.choice(TRIPLET_TYPES)
            if len(self.dataset.triples[t]) == 0:
                continue
            if len(self.dataset.triples[t]) < self.batch_size:
                yield [(t, i) for i in range(len(self.dataset.triples[t]))]
            else:
                idx = random.choice(range(0, len(self.dataset.triples[t]) - self.batch_size - 1))
                yield [(t, idx + i) for i in range(self.batch_size)]

    def __len__(self):
        return self.max_num_steps


if __name__ == "__main__":
    dataset = TripletsDataset("reactome", "all", p_model="ProtBert", m_model="ChemBERTa")
