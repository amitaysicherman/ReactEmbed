import os
import re
from abc import ABC

import numpy as np
import torch
from npy_append_array import NpyAppendArray
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoModel, AutoModelForMaskedLM, AutoTokenizer, BertConfig, \
    EsmModel

from common.data_types import DNA, PROTEIN, MOLECULE, TEXT, EMBEDDING_DATA_TYPES, P_BFD, P_T5_XL, ESM_1B, ESM_2, ESM_3, \
    PEBCHEM10M, ROBERTA, CHEMBERTA
from common.utils import get_type_to_vec_dim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 510
PROTEIN_MAX_LEN = 1023

protein_name_to_cp = {
    P_BFD: 'Rostlab/prot_bert_bfd',
    P_T5_XL: 'Rostlab/prot_t5_xl_half_uniref50-enc',
    ESM_1B: 'facebook/esm1b_t33_650M_UR50S',
    ESM_2: 'facebook/esm2_t12_35M_UR50D',
    ESM_3: 'esm3'
}

mol_name_to_cp = {
    PEBCHEM10M: "seyonec/PubChem10M_SMILES_BPE_450k",
    ROBERTA: "entropy/roberta_zinc_480m",
    CHEMBERTA: "seyonec/ChemBERTa-zinc-base-v1",
}


def clip_to_max_len(x: torch.Tensor, max_len: int = MAX_LEN):
    if x.shape[1] <= max_len:
        return x
    last_token = x[:, -1:]
    clipped_x = x[:, :max_len - 1]
    result = torch.cat([clipped_x, last_token], dim=1)
    return result


class ABCSeq2Vec(ABC):
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def to_vec(self, seq: str):
        inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
        inputs = clip_to_max_len(inputs)
        with torch.no_grad():
            hidden_states = self.model(inputs)[0]
        vec = torch.mean(hidden_states[0], dim=0)
        return self.post_process(vec)

    def post_process(self, vec):
        vec_flat = vec.detach().cpu().numpy().flatten()
        del vec
        return vec_flat.reshape(1, -1)


class Prot2vec(ABCSeq2Vec):
    def __init__(self, token="", name=""):
        super().__init__()
        self.cp_name = protein_name_to_cp[name]
        self.name = name
        self.token = token
        self.get_model_tokenizer()
        self.prot_dim = None

    def get_model_tokenizer(self):
        if self.name == P_BFD:
            self.tokenizer = BertTokenizer.from_pretrained(self.cp_name, do_lower_case=False)
            self.model = BertForMaskedLM.from_pretrained(self.cp_name, output_hidden_states=True).eval().to(device)
        elif self.name == ESM_3:
            from huggingface_hub import login
            from esm.models.esm3 import ESM3
            login(token=self.token)
            self.model = ESM3.from_pretrained("esm3_sm_open_v1", device=device)
        elif self.name == ESM_1B or self.name == ESM_2:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cp_name)
            self.model = EsmModel.from_pretrained(self.cp_name).eval().to(device)
        elif self.name == P_T5_XL:
            self.tokenizer = T5Tokenizer.from_pretrained(self.cp_name, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(self.cp_name).eval().to(device)
        else:
            raise ValueError(f"Unknown protein embedding: {self.name}")
        if device == torch.device("cpu"):
            self.model.to(torch.float32)

    def to_vec(self, seq: str):
        if self.name == ESM_3:
            if len(seq) > 2595:  # TODO : is it better sulution?
                seq = seq[:2595]
            from esm.sdk.api import ESMProtein

            protein = ESMProtein(sequence=seq)
            with torch.no_grad():
                protein = self.model.encode(protein)
                vec = self.model.forward(sequence_tokens=protein.sequence.unsqueeze(0).cuda()).embeddings[0].mean(dim=0)
        elif self.name in [ESM_1B, ESM_2]:
            inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
            inputs = clip_to_max_len(inputs)
            with torch.no_grad():
                vec = self.model(inputs)['pooler_output'][0]
        else:
            seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
            ids = self.tokenizer(seq, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            input_ids = clip_to_max_len(input_ids, PROTEIN_MAX_LEN)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            attention_mask = clip_to_max_len(attention_mask, PROTEIN_MAX_LEN)

            with torch.no_grad():
                embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if self.name == P_BFD:
                vec = embedding_repr.hidden_states[-1][0].mean(dim=0)
            else:
                vec = embedding_repr.last_hidden_state[0].mean(dim=0)
        self.prot_dim = vec.shape[-1]
        return self.post_process(vec)


class DNA2Vec(ABCSeq2Vec):
    def __init__(self):
        super().__init__()

        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.model = AutoModel.from_config(config).eval().to(device)
        if device == torch.device("cpu"):
            self.model.to(torch.float32)


class Mol2Vec(ABCSeq2Vec):
    def __init__(self, name):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(mol_name_to_cp[name]).base_model.eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(mol_name_to_cp[name])


class BioText2Vec(ABCSeq2Vec):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")
        self.model = AutoModel.from_pretrained("gsarti/biobert-nli").eval().to(device)
        if device == torch.device("cpu"):
            self.model.to(torch.float32)


class Seq2Vec:
    def __init__(self, self_token, protein_name=P_T5_XL, mol_name=PEBCHEM10M, use_cache=False):
        self.prot2vec = Prot2vec(self_token, protein_name)
        self.dna2vec = DNA2Vec()
        self.mol2vec = Mol2Vec(mol_name)
        self.text2vec = BioText2Vec()
        self.use_cache = use_cache
        if use_cache:
            self.cache = {}
        self.type_to_vec_dim = get_type_to_vec_dim(protein_name)

    def to_vec(self, seq: str, seq_type: str):
        if self.use_cache and seq in self.cache:
            return self.cache[seq]

        zeros = np.zeros((1, self.type_to_vec_dim[seq_type]))
        if seq_type == PROTEIN:
            vec = self.prot2vec.to_vec(seq)
        elif seq_type == DNA:
            vec = self.dna2vec.to_vec(seq)
        elif seq_type == MOLECULE:
            vec = self.mol2vec.to_vec(seq)
        elif seq_type == TEXT:
            vec = self.text2vec.to_vec(seq)
        else:
            print(f"Unknown sequence type: {seq_type}")
            return zeros
        if vec is None:
            return zeros
        if self.use_cache:
            self.cache[seq] = vec
        return vec


def read_seq_write_vec(seq2vec, input_file_name, output_file_name, seq_type):
    with open(input_file_name) as f:
        seqs = f.read().splitlines()
    missing_count = 0
    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    for seq in tqdm(seqs):
        vec = seq2vec.to_vec(seq, seq_type)
        if vec is None:
            missing_count += 1
        with NpyAppendArray(output_file_name) as f:
            f.append(vec)
    print(f"Missing {missing_count}({missing_count / len(seqs):%}) sequences for {seq_type}")


if __name__ == "__main__":
    from common.path_manager import item_path
    from common.args_manager import get_args

    args = get_args()
    self_token = args.auth_token
    protein_emd = args.protein_embedding
    mol_emd = args.molecule_embedding
    assert protein_emd in protein_name_to_cp, f"Unknown protein embedding: {protein_emd}"
    seq2vec = Seq2Vec(self_token, protein_name=protein_emd, mol_name=mol_emd)
    dtypes = EMBEDDING_DATA_TYPES

    for dt in dtypes:
        suf = ""
        if dt == PROTEIN:
            suf = f"_{protein_emd}"
        elif dt == MOLECULE:
            suf = f"_{mol_emd}"
        input_file = f'{item_path}/{dt}_sequences.txt'
        output_file_name = f'{item_path}/{dt}{suf}_vec.npy'
        read_seq_write_vec(seq2vec, input_file, output_file_name, dt)
