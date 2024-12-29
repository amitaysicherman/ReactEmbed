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
        if not os.path.exists(pdb_file):
            return np.zeros(3072)
        with open(pdb_file, "r") as f:
            pdb_data = f.read()
        mol = Chem.MolFromPDBBlock(pdb_data, sanitize=False)
        if mol is None:
            return np.zeros(3072)
        try:
            protein = data.Protein.from_molecule(mol)
        except Exception as e:
            print(e)
            return np.zeros(3072)
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

    def lines_to_vecs(self, start_index, end_index):
        all_vecs = []
        for i in tqdm(range(start_index, end_index)):
            vec = self.to_vec(i)
            all_vecs.append(vec)
        all_vecs = np.array(all_vecs)
        return all_vecs


def main(model, data_name, start_index=-1, end_index=-1):
    GearNetEMB = GearNet3Embedder(f"data/{data_name}/fold")
    vecs = GearNetEMB.lines_to_vecs(start_index, end_index)
    np.save(f"data/{data_name}/GearNet_vectors.npy", vecs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert sequence to vector')
    parser.add_argument('--data_name', type=str, help='Data name', default="reactome")
    parser.add_argument('--start_index', type=int, default=-1)
    parser.add_argument('--end_index', type=int, default=-1)
    args = parser.parse_args()
    main(args.model, args.data_name, args.start_index, args.end_index)
