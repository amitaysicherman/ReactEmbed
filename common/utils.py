import datetime
import os

import torch

from common.data_types import CatalystOBJ, Entity, Reaction, UNKNOWN_ENTITY_TYPE, DNA, PROTEIN, MOLECULE, TEXT, P_T5_XL, \
    P_BFD, ESM_1B, ESM_2, ESM_3
from common.path_manager import fuse_path
from contrastive_learning.model import ReactEmbedModel, ReactEmbedConfig


def get_type_to_vec_dim(prot_emd_type=P_T5_XL):
    type_to_dim = {DNA: 768, MOLECULE: 768, TEXT: 768, PROTEIN: 1024}
    if prot_emd_type == P_T5_XL:
        type_to_dim[PROTEIN] = 1024
    elif prot_emd_type == P_BFD:
        type_to_dim[PROTEIN] = 1024
    elif prot_emd_type == ESM_1B:
        type_to_dim[PROTEIN] = 1280
    elif prot_emd_type == ESM_2:
        type_to_dim[PROTEIN] = 480
    elif prot_emd_type == ESM_3:
        type_to_dim[PROTEIN] = 1536
    return type_to_dim


def db_to_type(db_name):
    db_name = db_name.lower()
    if db_name == "ensembl":
        return DNA
    elif db_name == "embl":
        return PROTEIN
    elif db_name == "uniprot" or db_name == "uniprot isoform":
        return PROTEIN
    elif db_name == "chebi":
        return MOLECULE
    elif db_name == "guide to pharmacology":
        return MOLECULE
    elif db_name == "go":
        return TEXT
    elif db_name == "text":
        return TEXT
    elif db_name == "ncbi nucleotide":
        return DNA
    elif db_name == "pubchem compound":
        return MOLECULE
    else:
        return UNKNOWN_ENTITY_TYPE


def catalyst_from_dict(d: dict) -> CatalystOBJ:
    entities = [Entity(**e) for e in d["entities"]]
    activity = d["activity"]
    return CatalystOBJ(entities, activity)


def reaction_from_dict(d: dict) -> Reaction:
    name = d["name"]
    inputs = [Entity(**e) for e in d["inputs"]]
    outputs = [Entity(**e) for e in d["outputs"]]
    catalysis = [catalyst_from_dict(c) for c in d["catalysis"]]
    year, month, day = d["date"].split("_")
    date = datetime.date(int(year), int(month), int(day))
    reactome_id = int(d["reactome_id"])
    biological_process = d["biological_process"].split("_")
    return Reaction(name, inputs, outputs, catalysis, date, reactome_id, biological_process)


def reaction_from_str(s: str) -> Reaction:
    return reaction_from_dict(eval(s))


def load_fuse_model(name):
    name = str(os.path.join(fuse_path, name))
    cp_names = os.listdir(name)
    cp_name = [x for x in cp_names if x.endswith(".pt")][0]
    cp_data = torch.load(f"{name}/{cp_name}", map_location=torch.device('cpu'))
    config_file = os.path.join(name, 'config.txt')
    config = ReactEmbedConfig.load_from_file(config_file)
    dim = config.output_dim[0]
    model = ReactEmbedModel(config)
    model.load_state_dict(cp_data)
    model = model.eval()
    return model, dim


def model_args_to_name(**kwargs):
    names_to_check = ["batch_size", "p_model", "m_model", "n_layers", "hidden_dim", "dropout", "epochs",
                      "lr", "flip_prob"]
    for name in names_to_check:
        if name not in kwargs:
            raise ValueError(f"Missing argument: {name}")
    for k in kwargs:
        if k not in names_to_check:
            raise ValueError(f"Extra argument: {k}")
    batch_size = kwargs["batch_size"]
    p_model = kwargs["p_model"]
    m_model = kwargs["m_model"]
    n_layers = kwargs["n_layers"]
    hidden_dim = kwargs["hidden_dim"]
    dropout = kwargs["dropout"]
    epochs = kwargs["epochs"]
    lr = kwargs["lr"]
    flip_prob = kwargs["flip_prob"]

    return f"{p_model}-{m_model}-{n_layers}-{hidden_dim}-{dropout}-{epochs}-{lr}-{batch_size}-{flip_prob}"
