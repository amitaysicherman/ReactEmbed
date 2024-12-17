import os
import re

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer

from common.path_manager import proteins_file, molecules_file, item_path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
name_to_hf_cp = {
    "ProtBert": 'Rostlab/prot_bert',
    "ChemBERTa": "seyonec/ChemBERTa-zinc-base-v1",
    "MoLFormer": "ibm/MoLFormer-XL-both-10pct"
}


class Esm3Embedder:
    def __init__(self, size):
        self.size = size
        if size == "small":
            self.model = ESMC.from_pretrained("esmc_300m").to(device).eval()
        elif size == "medium":
            self.model = ESMC.from_pretrained("esmc_600m").to(device).eval()
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
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def to_vec(self, seq: str):
        return self.model.to_vec(seq)


def model_to_type(model_name):
    if model_name in ["ChemBERTa", "MoLFormer"]:
        return "molecule"
    elif model_name in ["ProtBert", "esm3-small", "esm3-medium"]:
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
                        choices=["ProtBert", "ChemBERTa", "MoLFormer", "esm3-small", "esm3-medium"])
    args = parser.parse_args()
    main(args.model)
