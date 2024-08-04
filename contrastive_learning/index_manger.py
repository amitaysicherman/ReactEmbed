import os
import random
from typing import Dict

import numpy as np
import torch

from common.data_types import REACTION, COMPLEX, UNKNOWN_ENTITY_TYPE, PROTEIN, LOCATION, DATA_TYPES, \
    NO_PRETRAINED_EMD, PRETRAINED_EMD, PRETRAINED_EMD_FUSE, MOLECULE, TEXT, P_T5_XL, ROBERTA
from common.path_manager import item_path
from common.utils import get_type_to_vec_dim, load_fuse_model
from contrastive_learning.model import apply_model

REACTION_NODE_ID = 0
COMPLEX_NODE_ID = 1
UNKNOWN_ENTITY_TYPE = UNKNOWN_ENTITY_TYPE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(42)


class NodeData:
    def __init__(self, index, name, type_, vec=None, have_seq=True):
        self.index = index
        self.name = name
        self.type = type_
        self.vec = vec
        self.have_seq = have_seq


class NodesIndexManager:
    def __init__(self, pretrained_method=PRETRAINED_EMD, fuse_name="",
                 prot_emd_type=P_T5_XL, mol_emd_type=ROBERTA, fuse_model=None):
        reaction_node = NodeData(REACTION_NODE_ID, REACTION, REACTION)
        complex_node = NodeData(COMPLEX_NODE_ID, COMPLEX, COMPLEX)
        self.nodes = [reaction_node, complex_node]
        locations_counts = {}
        self.index_count = 2
        self.dtype_to_first_index = dict()
        self.dtype_to_last_index = dict()
        self.type_to_vec_dim = get_type_to_vec_dim(prot_emd_type)
        if fuse_model is not None:
            self.fuse_model = fuse_model
        else:
            self.fuse_model = load_fuse_model(fuse_name)
        if self.fuse_model is not None:
            self.fuse_model = self.fuse_model.eval().to(device)
        for dt in DATA_TYPES:
            self.dtype_to_first_index[dt] = self.index_count
            names_file = f'{item_path}/{dt}.txt'
            with open(names_file) as f:
                lines = f.read().splitlines()

            if dt in [PROTEIN, MOLECULE]:  # EMBEDDING_DATA_TYPES:
                if pretrained_method == NO_PRETRAINED_EMD:
                    vectors = np.stack([np.random.rand(self.type_to_vec_dim[dt]) for _ in range(len(lines))])
                else:
                    prefix = ""
                    if dt == PROTEIN and prot_emd_type:
                        prefix = f"{prot_emd_type}_"
                    if dt == MOLECULE and mol_emd_type:
                        prefix = f"{mol_emd_type}_"
                    pretrained_vec_file = f'{item_path}/{dt}_{prefix}vec.npy'
                    vectors = np.load(pretrained_vec_file)
                    if pretrained_method == PRETRAINED_EMD_FUSE:
                        with torch.no_grad():
                            vectors = apply_model(self.fuse_model, vectors, dt).detach().cpu().numpy()
            elif dt == UNKNOWN_ENTITY_TYPE:
                vectors = [np.zeros(self.type_to_vec_dim[PROTEIN]) for _ in range(len(lines))]
            else:
                vectors = [None] * len(lines)

            seq_file = f'{item_path}/{dt}_sequences.txt'
            if not os.path.exists(seq_file):
                seqs = [False] * len(lines)
            else:
                with open(seq_file) as f:
                    seqs = f.read().splitlines()
                    seqs = [True if len(seq) > 0 else False for seq in seqs]

            for i, line in enumerate(lines):
                name = "@".join(line.split("@")[:-1])
                node = NodeData(self.index_count, name, dt, vectors[i], seqs[i])
                self.nodes.append(node)
                self.index_count += 1
                if dt == LOCATION:
                    count = line.split("@")[-1]
                    locations_counts[node.index] = int(count)
            self.dtype_to_last_index[dt] = self.index_count
        self.locations = list(locations_counts.keys())
        self.locations_probs = np.array([locations_counts[l] for l in self.locations]) / sum(locations_counts.values())
        self.index_to_node = {node.index: node for node in self.nodes}
        self.name_to_node: Dict[str, NodeData] = {node.name: node for node in self.nodes}

        mulecules = [node for node in self.nodes if node.type == MOLECULE]
        self.molecule_indexes = [node.index for node in mulecules]
        self.molecule_array = np.array([node.vec for node in mulecules])

        proteins = [node for node in self.nodes if node.type == PROTEIN]
        self.protein_indexes = [node.index for node in proteins]
        self.protein_array = np.array([node.vec for node in proteins])

        texts = [node for node in self.nodes if node.type == TEXT]
        self.text_indexes = [node.index for node in texts]
        self.text_array = np.array([node.vec for node in texts])
        self.type_to_indexes = {PROTEIN: self.protein_indexes, MOLECULE: self.molecule_indexes, TEXT: self.text_indexes}
