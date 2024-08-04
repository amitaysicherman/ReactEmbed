import math
import random
from collections import defaultdict
from itertools import combinations

import numpy as np
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from common.data_types import DNA
from common.data_types import MOLECULE, PROTEIN
from common.data_types import Reaction
from common.path_manager import reactions_file
from common.utils import reaction_from_str
from contrastive_learning.index_manger import NodesIndexManager

EMBEDDING_DATA_TYPES = [MOLECULE, PROTEIN]


def pairs_from_reaction(reaction: Reaction, nodes_index_manager: NodesIndexManager):
    elements = []

    reaction_elements = reaction.inputs + sum([x.entities for x in reaction.catalysis], []) + reaction.outputs

    for reaction_element in reaction_elements:
        node = nodes_index_manager.name_to_node[reaction_element.get_db_identifier()]
        elements.append(node.index)
    elements = [e for e in elements if nodes_index_manager.index_to_node[e].have_seq]
    elements = list(set(elements))
    pairs = []
    for e1, e2 in combinations(elements, 2):
        type_1 = nodes_index_manager.index_to_node[e1].type
        type_2 = nodes_index_manager.index_to_node[e2].type

        if type_1 >= type_2:
            pairs.append((e2, e1))
        else:
            pairs.append((e1, e2))
    return elements, pairs


class PairsDataset(Dataset):
    def __init__(self, reactions, nodes_index_manager: NodesIndexManager, neg_count=1, triples=False):
        self.nodes_index_manager = nodes_index_manager
        self.all_pairs = []
        self.all_elements = []
        for reaction in tqdm(reactions):
            elements, pairs = pairs_from_reaction(reaction, nodes_index_manager)
            self.all_elements.extend(elements)
            self.all_pairs.extend(pairs)

        self.pairs_unique = set(self.all_pairs)
        self.all_pairs = list(self.pairs_unique)
        elements_unique, elements_count = np.unique(self.all_elements, return_counts=True)
        self.elements_unique = elements_unique
        for dtype in EMBEDDING_DATA_TYPES:
            dtype_indexes = [i for i in range(len(elements_unique)) if
                             nodes_index_manager.index_to_node[elements_unique[i]].type == dtype]
            dtype_unique = elements_unique[dtype_indexes]

            setattr(self, f"{dtype}_unique", dtype_unique)
        self.data = []
        for i in tqdm(range(len(self.all_pairs))):
            a, b = self.all_pairs[i]
            a_type = nodes_index_manager.index_to_node[a].type
            b_type = nodes_index_manager.index_to_node[b].type
            if triples:
                fake_a = self.sample_neg_pair(b_=b, other_dtype=a_type)[0]
                fake_b = self.sample_neg_pair(a_=a, other_dtype=b_type)[-1]
                self.data.append((a, b, fake_b))
                self.data.append((b, a, fake_a))
            else:
                self.data.append((a, b, 1))
                self.data.append((b, a, 1))
                for j in range(neg_count):
                    self.data.append((*self.sample_neg_pair(a_=a, other_dtype=b_type), -1))
                    self.data.append((*self.sample_neg_pair(b_=b, other_dtype=a_type), -1))
                    self.data.append((*self.sample_neg_pair(a_=b, other_dtype=a_type), -1))
                    self.data.append((*self.sample_neg_pair(b_=a, other_dtype=b_type), -1))

    def __len__(self):
        return len(self.data)

    def sample_neg_pair(self, a_=None, b_=None, other_dtype=None):
        while True:
            elements = getattr(self, f'{other_dtype}_unique')
            a = random.choice(elements) if a_ is None else a_
            b = random.choice(elements) if b_ is None else b_
            if (a, b) not in self.pairs_unique:
                return a, b

    def __getitem__(self, idx):
        return self.data[idx]


class SameNameBatchSampler(Sampler):
    def __init__(self, dataset: PairsDataset, batch_size, shuffle=False):

        self.dataset = dataset
        self.batch_size = batch_size
        name_to_indices = defaultdict(list)
        self.shuffle = shuffle
        for idx in range(len(dataset)):
            idx1, idx2, _ = dataset[idx]
            type_1 = dataset.nodes_index_manager.index_to_node[idx1].type
            type_2 = dataset.nodes_index_manager.index_to_node[idx2].type
            name_to_indices[(type_1, type_2)].append(idx)
        self.names = list(name_to_indices.keys())
        self.name_to_indices = dict()
        if shuffle:
            random.shuffle(self.names)
        for name in name_to_indices:
            if shuffle:
                random.shuffle(name_to_indices[name])
            self.name_to_indices[name] = np.array(name_to_indices[name])

    def __iter__(self):

        for name in self.names:
            indices = self.name_to_indices[name]
            if self.batch_size > len(indices):
                yield indices
                continue
            for i in range(0, len(indices) - self.batch_size, self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)


def get_reaction_entities(reaction, check_output):
    if check_output:
        return reaction.inputs + reaction.outputs + sum([c.entities for c in reaction.catalysis], [])
    return reaction.inputs + sum([c.entities for c in reaction.catalysis], [])


def have_no_seq_nodes(reaction, node_index_manager: NodesIndexManager, check_output=False):
    entitites = get_reaction_entities(reaction, check_output)
    for e in entitites:
        if not node_index_manager.name_to_node[e.get_db_identifier()].have_seq:
            return True
    return False


def have_dna_nodes(reaction, node_index_manager: NodesIndexManager, check_output=False):
    entitites = get_reaction_entities(reaction, check_output)
    for e in entitites:
        if node_index_manager.name_to_node[e.get_db_identifier()].type == DNA:
            return True
    return False


def get_reactions():
    with open(reactions_file) as f:
        lines = f.readlines()
    reactions = [reaction_from_str(line) for line in lines]
    node_index_manager = NodesIndexManager()
    reactions = [reaction for reaction in reactions if
                 not have_no_seq_nodes(reaction, node_index_manager, check_output=True)]

    reactions = [reaction for reaction in reactions if
                 not have_dna_nodes(reaction, node_index_manager, check_output=True)]

    reactions = sorted(reactions, key=lambda x: x.date)
    train_val_index = int(0.7 * len(reactions))
    val_test_index = int(0.85 * len(reactions))
    train_lines = reactions[:train_val_index]
    val_lines = reactions[train_val_index:val_test_index]
    test_lines = reactions[val_test_index:]
    return train_lines, val_lines, test_lines
