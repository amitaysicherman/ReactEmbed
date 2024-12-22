import datetime
import os

import torch
from transformers.models.esm.openfold_utils import atom14_to_atom37, to_pdb
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein

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


def name_to_model_args(name):
    names = name.split("-")
    if name.startswith("esm"):
        p_model = f"esm-{names[1]}"
        names = names[1:]
    else:
        p_model = names[0]
    m_model = names[1]
    n_layers = int(names[2])
    hidden_dim = int(names[3])
    dropout = float(names[4])
    epochs = int(names[5])
    lr = float(names[6])
    batch_size = int(names[7])
    flip_prob = float(names[8])
    return {"p_model": p_model, "m_model": m_model, "n_layers": n_layers, "hidden_dim": hidden_dim, "dropout": dropout,
            "epochs": epochs, "lr": lr, "batch_size": batch_size, "flip_prob": flip_prob}


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
