import io
import os

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

from common.utils import fold_to_pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class GearNet3Embedder:
    def __init__(self, gearnet_cp_file="data/models/gearnet/mc_gearnet_edge.pth"):

        from torchdrug import models, layers, data, transforms
        from torchdrug.layers import geometry
        self.data = data
        self.transforms = transforms
        self.fold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self.fold_model = self.fold_model.to(device).eval()
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

    def fold_seq(self, seq: str, output_io):
        tokenized_input = self.fold_tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            output = self.fold_model(tokenized_input)
        pdbs = fold_to_pdb(output)
        output_io.write(pdbs[0])

    def to_vec(self, seq: str):
        if len(seq) > 550:
            seq = seq[:550]
        try:
            fold_tmp_io = io.StringIO()
            self.fold_seq(seq, fold_tmp_io)
            pdb_content = fold_tmp_io.getvalue()
            fold_tmp_io.close()
        except Exception as e:
            print(e)
            return None
        if not pdb_content:
            return None

        mol = Chem.MolFromPDBBlock(pdb_content, sanitize=False)
        if mol is None:
            return None
        try:
            protein = self.data.Protein.from_molecule(mol)
        except Exception as e:
            print(e)
            return None
        truncate_transform = self.transforms.TruncateProtein(max_length=550, random=False)
        protein_view_transform = self.transforms.ProteinView(view="residue")
        transform = self.transforms.Compose([truncate_transform, protein_view_transform])
        protein = {"graph": protein}
        protein = transform(protein)
        protein = protein["graph"]
        protein = self.data.Protein.pack([protein])
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
