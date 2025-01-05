from os.path import join as pjoin

import numpy as np
import torch
from torchdrug import datasets
from torchdrug import layers
from torchdrug import models
from torchdrug import transforms
from torchdrug.layers import geometry
from tqdm import tqdm

from common.path_manager import data_path

name = "EnzymeCommission"
device = "cuda" if torch.cuda.is_available() else "cpu"


base_dir = f"{data_path}/torchdrug/"
output_base = pjoin(base_dir, name)


def task_name_to_dataset_class(task_name):
    if task_name.startswith("GeneOntology"):
        return getattr(datasets, "GeneOntology")
    return getattr(datasets, task_name)


gearnet_cp_file = "data/models/gearnet/mc_gearnet_edge.pth"
gearnet_model = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], num_relation=7,
                               edge_input_dim=59,
                               num_angle_bin=8,
                               batch_norm=True, concat_hidden=True, short_cut=True, readout="sum").to(device).eval()
checkpoint = torch.load(gearnet_cp_file, map_location=torch.device(device))
gearnet_model.load_state_dict(checkpoint)

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[
                                                        geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                        geometry.KNNEdge(k=10, min_distance=5),
                                                        geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")
dataset_class = task_name_to_dataset_class(name)
truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

dataset = dataset_class(output_base, atom_feature=None, bond_feature=None)
splits = dataset.split()
train, valid, test, *unused_test = splits

for split, name in zip([train, valid, test], ["train", "valid", "test"]):
    vecs = []
    for protein in tqdm(split):
        protein = transform(protein)
        protein = protein["graph"]
        protein = protein.Protein.pack([protein])
        protein = graph_construction_model(protein)
        output = gearnet_model(protein.to(device), protein.node_feature.float().to(device))['node_feature'].mean(dim=0)
        output = output.cpu().detach().numpy().flatten()
        vecs.append(output)
    vecs = np.array(vecs)
    output_file = f"{output_base}/{name}_GearNet_1.npy"
    np.save(output_file, vecs)
