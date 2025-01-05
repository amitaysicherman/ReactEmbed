from os.path import join as pjoin

import torch
from torchdrug import datasets
from torchdrug import models
from torchdrug import transforms

from common.path_manager import data_path

device = "cuda" if torch.cuda.is_available() else "cpu"

truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])

name = "EnzymeCommission"
base_dir = f"{data_path}/torchdrug/"
output_base = pjoin(base_dir, name)


def task_name_to_dataset_class(task_name):
    if task_name.startswith("GO"):
        return getattr(datasets, "GO")
    return getattr(datasets, task_name)


gearnet_model = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], num_relation=7,
                               edge_input_dim=59,
                               num_angle_bin=8,
                               batch_norm=True, concat_hidden=True, short_cut=True, readout="sum").to(device).eval()

dataset = task_name_to_dataset_class(name)(output_base, transform=transform, atom_feature=None, bond_feature=None)


class GearNet3Embedder:
    def __init__(self, gearnet_cp_file="data/models/gearnet/mc_gearnet_edge.pth"):

        from torchdrug import layers, data, transforms
        from torchdrug.layers import geometry
        self.data = data
        self.transforms = transforms
        self.fold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self.fold_model = self.fold_model.to(device).eval()
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
