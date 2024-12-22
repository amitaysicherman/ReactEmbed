import os
import re

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from rdkit import Chem
from torchdrug import models, layers, data, transforms
from torchdrug.layers import geometry
from transformers import AutoModel, BertModel, BertTokenizer
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein

from common.path_manager import proteins_file, molecules_file, item_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
name_to_hf_cp = {
    "ProtBert": 'Rostlab/prot_bert',
    "ChemBERTa": "seyonec/ChemBERTa-zinc-base-v1",
    "MoLFormer": "ibm/MoLFormer-XL-both-10pct"
}


def fold_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


class GearNet3Embedder:
    def __init__(self, gearnet_cp_file="data/models/gearnet/mc_gearnet_edge.pth"):
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

    def fold_seq(self, seq: str, output_file):
        tokenized_input = self.fold_tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            output = self.fold_model(tokenized_input)
        pdbs = fold_to_pdb(output)
        with open(output_file, 'w') as f:
            f.write(pdbs[0])

    def to_vec(self, seq: str, fold_tmp_file="tmp.pdb"):
        self.fold_seq(seq, fold_tmp_file)
        mol = Chem.MolFromPDBFile(fold_tmp_file, sanitize=False)
        os.remove(fold_tmp_file)
        if mol is None:
            raise ValueError("RDKit cannot read PDB file 'tmp.pdb'")
        protein = data.Protein.from_molecule(mol)
        truncate_transform = transforms.TruncateProtein(max_length=550, random=False)
        protein_view_transform = transforms.ProteinView(view="residue")
        transform = transforms.Compose([truncate_transform, protein_view_transform])
        protein = {"graph": protein}
        protein = transform(protein)
        protein = protein["graph"]
        protein = data.Protein.pack([protein])
        protein = self.graph_construction_model(protein)
        output = self.gearnet_model(protein.cuda(), protein.node_feature.float().cuda())
        output = output['node_feature'].mean(dim=0)
        return output.cpu().numpy().flatten()


class Esm3Embedder:
    def __init__(self, size):
        self.size = size
        if size == "small":
            self.model = ESMC.from_pretrained("esmc_300m", device=device).eval()
        elif size == "medium":
            self.model = ESMC.from_pretrained("esmc_600m", device=device).eval()
        else:
            raise ValueError(f"Unknown size: {size}")

    def to_vec(self, seq: str):
        if len(seq) > 1023:
            seq = seq[:1023]
        try:
            protein = ESMProtein(sequence=seq)
            protein = self.model.encode(protein)
            conf = LogitsConfig(return_embeddings=True, sequence=True)
            vec = self.model.logits(protein, conf).embeddings[0]
            return vec.mean(dim=0).cpu().numpy().flatten()
        except Exception as e:
            print(e)
            return None


class PortBert:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert").to(device).eval()

    def to_vec(self, seq: str):
        if len(seq) > 1023:
            seq = seq[:1023]
        seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
        ids = self.tokenizer(seq, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
        vec = embedding_repr.last_hidden_state[0].mean(dim=0)
        return vec.detach().cpu().numpy().flatten()


class MoLFormer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True,
                                               trust_remote_code=True).to(device).eval()

    def to_vec(self, seq: str):
        if len(seq) > 510:
            seq = seq[:510]
        inputs = self.tokenizer([seq], return_tensors='pt')  # ["input_ids"].to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        vec = outputs.pooler_output
        return vec.detach().cpu().numpy().flatten()


class ChemBERTa:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device).eval()

    def to_vec(self, seq: str):
        if len(seq) > 510:
            seq = seq[:510]
        inputs = self.tokenizer([seq], return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden_states = self.model(**inputs)[0]
        vec = torch.mean(hidden_states[0], dim=0)
        return vec.detach().cpu().numpy().flatten()


class SeqToVec:
    def __init__(self, model_name):
        if model_name == "ProtBert":
            self.model = PortBert()
        elif model_name == "ChemBERTa":
            self.model = ChemBERTa()
        elif model_name == "MoLFormer":
            self.model = MoLFormer()
        elif model_name in ["esm3-small", "esm3-medium"]:
            size = model_name.split("-")[-1]
            self.model = Esm3Embedder(size)
        elif model_name == "GearNet":
            self.model = GearNet3Embedder()
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def to_vec(self, seq: str):
        return self.model.to_vec(seq)


def model_to_type(model_name):
    if model_name in ["ChemBERTa", "MoLFormer"]:
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


def main(model):
    data_types = model_to_type(model)
    seq_to_vec = SeqToVec(model)
    if data_types == "protein":
        file = proteins_file.replace(".txt", "_sequences.txt")
    else:
        file = molecules_file.replace(".txt", "_sequences.txt")
    with open(file, "r") as f:
        lines = f.readlines()
    all_vecs = []
    output_file = f"{item_path}/{model}_vectors.npy"
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
        return None
    for line in tqdm(lines):
        if len(line.strip()) == 0:
            all_vecs.append(None)
        seq = line.strip()
        vec = seq_to_vec.to_vec(seq)
        all_vecs.append(vec)
    all_vecs = fill_none_with_zeros(all_vecs)
    all_vecs = np.array(all_vecs)
    np.save(output_file, all_vecs)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Convert sequence to vector')
    parser.add_argument('--model', type=str, help='Model to use', default="ChemBERTa",
                        choices=["ProtBert", "ChemBERTa", "MoLFormer", "esm3-small", "esm3-medium", "GearNet"])
    args = parser.parse_args()
    main(args.model)
