from enum import Enum

from transformers.models.esm.openfold_utils import atom14_to_atom37, to_pdb
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein


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


class Config(Enum):
    PRE = "pre"
    our = "our"
    both = "both"
