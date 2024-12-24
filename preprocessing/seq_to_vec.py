import io
import os
import re

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import AutoModel, BertModel, BertTokenizer
from transformers import AutoTokenizer, EsmForProteinFolding

from common.utils import fold_to_pdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
name_to_hf_cp = {
    "ProtBert": 'Rostlab/prot_bert',
    "ChemBERTa": "seyonec/ChemBERTa-zinc-base-v1",
    "MoLFormer": "ibm/MoLFormer-XL-both-10pct"
}


class MolCLREmbedder:
    def __init__(self, cp_file="data/models/MolCLR/model.pth"):
        if not os.path.exists(cp_file):
            os.makedirs(os.path.dirname(cp_file), exist_ok=True)
            url = "https://github.com/yuyangw/MolCLR/blob/master/ckpt/pretrained_gin/checkpoints/model.pth"
            os.system(f"wget {url} -O {cp_file}")
        from preprocessing.molCLR import GINet, smiles_to_data
        self.smiles_to_data = smiles_to_data
        self.model = GINet()

        self.model.load_my_state_dict(torch.load("data/models/MolCLR/model.pth", map_location="cpu"))
        self.model.eval()  # .to(device)

    def to_vec(self, seq: str):
        data = self.smiles_to_data(seq)
        if data is None:
            return None
        data = data  # `.to(device)
        with torch.no_grad():
            emb, _ = self.model(data)
        return emb.detach().cpu().numpy().flatten()


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
        output = self.gearnet_model(protein.cuda(), protein.node_feature.float().cuda())
        output = output['node_feature'].mean(dim=0)
        return output.detach().cpu().numpy().flatten()


class Esm3Embedder:
    def __init__(self, size):
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        self.ESMProtein = ESMProtein
        self.LogitsConfig = LogitsConfig
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
            protein = self.ESMProtein(sequence=seq)
            protein = self.model.encode(protein)
            conf = self.LogitsConfig(return_embeddings=True, sequence=True)
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
        self.mem = dict()
        if model_name == "ProtBert":
            self.model = PortBert()
            self.dtype = "protein"
        elif model_name == "ChemBERTa":
            self.model = ChemBERTa()
            self.dtype = "molecule"
        elif model_name == "MoLFormer":
            self.model = MoLFormer()
            self.dtype = "molecule"
        elif model_name in ["esm3-small", "esm3-medium"]:
            size = model_name.split("-")[-1]
            self.model = Esm3Embedder(size)
            self.dtype = "protein"
        elif model_name == "GearNet":
            self.model = GearNet3Embedder()
            self.dtype = "protein"
        elif model_name == "MolCLR":
            self.model = MolCLREmbedder()
            self.dtype = "molecule"
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def to_vec(self, seq: str):

        if len(seq) == 0:
            return None
        if self.dtype == "protein":
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


def main(model, data_name):
    if "esm3" in model:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
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
    args = parser.parse_args()
    main(args.model, args.data_name)
