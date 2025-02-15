from enum import Enum


def model_args_to_name(p_model, m_model, n_layers, hidden_dim, dropout, epochs, lr, batch_size, flip_prob, shared_dim,
                       samples_ratio, no_pp_mm):
    return f"{p_model}-{m_model}-{n_layers}-{hidden_dim}-{dropout}-{epochs}-{lr}-{batch_size}-{flip_prob}-{shared_dim}-{samples_ratio}-{no_pp_mm}"


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
    shared_dim = int(names[9])
    samples_ratio = int(names[10])
    no_pp_mm = int(names[11])
    return {"p_model": p_model, "m_model": m_model, "n_layers": n_layers, "hidden_dim": hidden_dim, "dropout": dropout,
            "epochs": epochs, "lr": lr, "batch_size": batch_size, "flip_prob": flip_prob, "shared_dim": shared_dim,
            "samples_ratio": samples_ratio, "no_pp_mm": no_pp_mm}


class Config(Enum):
    PRE = "pre"
    our = "our"
    both = "both"
