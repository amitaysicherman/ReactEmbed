import os

import numpy as np
import torch
from rdkit import Chem
from torchdrug import models, layers, data, transforms
from torchdrug.layers import geometry
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GearNet3Embedder:
    def __init__(self, base_fold_path, gearnet_cp_file="data/models/gearnet/mc_gearnet_edge.pth"):

        self.base_fold_path = base_fold_path
        self.gearnet_model = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], num_relation=7,
                                            edge_input_dim=59,
                                            num_angle_bin=8,
                                            batch_norm=True, concat_hidden=True, short_cut=True, readout="sum").to(
            device).eval()
        checkpoint = torch.load(gearnet_cp_file)
        self.gearnet_model.load_state_dict(checkpoint)
        self.graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                                 edge_layers=[
                                                                     geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                                 edge_feature="gearnet")

    def to_vec(self, fold_index):
        pdb_file = os.path.join(self.base_fold_path, f"{fold_index}.pdb")
        with open(pdb_file, "r") as f:
            pdb_data = f.read()
        mol = Chem.MolFromPDBBlock(pdb_data, sanitize=False)
        if mol is None:
            return None
        try:
            protein = data.Protein.from_molecule(mol)
        except Exception as e:
            print(e)
            return None
        truncate_transform = transforms.TruncateProtein(max_length=550, random=False)
        protein_view_transform = transforms.ProteinView(view="residue")
        transform = transforms.Compose([truncate_transform, protein_view_transform])
        protein = {"graph": protein}
        protein = transform(protein)
        protein = protein["graph"]
        protein = data.Protein.pack([protein])
        protein = self.graph_construction_model(protein)
        output = self.gearnet_model(protein.to(device), protein.node_feature.float().to(device))
        output = output['node_feature'].mean(dim=0)
        return output.detach().cpu().numpy().flatten()


class SeqToVec:
    def __init__(self):
        self.mem = dict()
        self.model = GearNet3Embedder()

    def to_vec(self, seq: str):

        if len(seq) == 0:
            return None
        seq = seq.replace(".", "")
        if seq in self.mem:
            return self.mem[seq]
        vec = self.model.to_vec(seq)
        self.mem[seq] = vec
        return vec

    def lines_to_vecs(self, lines):
        all_vecs = []
        for line in tqdm(lines):
            if len(line.strip()) == 0:
                all_vecs.append(None)
                continue
            seq = line.strip()
            vec = self.to_vec(seq)
            all_vecs.append(vec)
        all_vecs = fill_none_with_zeros(all_vecs)
        all_vecs = np.array(all_vecs)
        return all_vecs


def model_to_type(model_name):
    if model_name in ["ChemBERTa", "MoLFormer", "MolCLR"]:
        return "molecule"
    elif model_name in ["ProtBert", "esm3-small", "esm3-medium", "GearNet"]:
        return "protein"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def fill_none_with_zeros(vecs):
    first_non_none = None
    for i, vec in enumerate(vecs):
        if vec is not None:
            first_non_none = vec
            break
    zeroes = np.zeros_like(first_non_none)
    for i, vec in enumerate(vecs):
        if vec is None:
            vecs[i] = zeroes
    return vecs


def main(model, data_name, start_index=-1, end_index=-1):
    if "esm3" in model:
        pass
    proteins_file = f'data/{data_name}/proteins.txt'
    molecules_file = f'data/{data_name}/molecules.txt'
    data_types = model_to_type(model)
    seq_to_vec = SeqToVec(model)

    if data_types == "protein":
        file = proteins_file.replace(".txt", "_sequences.txt")
    else:
        file = molecules_file.replace(".txt", "_sequences.txt")
    with open(file, "r") as f:
        lines = f.readlines()

    if start_index != -1 and end_index != -1:
        end_index = min(end_index, len(lines))
        lines = lines[start_index:end_index]
        output_file = f"data/{data_name}/{model}_vectors_{start_index}_{end_index}.npy"
    else:
        output_file = f"data/{data_name}/{model}_vectors.npy"
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
        return None
    all_vecs = seq_to_vec.lines_to_vecs(lines)
    np.save(output_file, all_vecs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert sequence to vector')
    parser.add_argument('--model', type=str, help='Model to use', default="MolCLR",
                        choices=["ProtBert", "ChemBERTa", "MoLFormer", "esm3-small", "esm3-medium", "GearNet",
                                 "MolCLR"])
    parser.add_argument('--data_name', type=str, help='Data name', default="reactome")
    parser.add_argument('--start_index', type=int, default=-1)
    parser.add_argument('--end_index', type=int, default=-1)

    args = parser.parse_args()
    main(args.model, args.data_name, args.start_index, args.end_index)
