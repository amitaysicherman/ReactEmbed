from enum import Enum


def model_args_to_name(**kwargs):
    names_to_check = ["batch_size", "p_model", "m_model", "n_layers", "hidden_dim", "dropout", "epochs",
                      "lr", "flip_prob", "shared_dim"]
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
    shared_dim = kwargs["shared_dim"]

    return f"{p_model}-{m_model}-{n_layers}-{hidden_dim}-{dropout}-{epochs}-{lr}-{batch_size}-{flip_prob}-{shared_dim}"


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
    min_value = int(names[9])
    return {"p_model": p_model, "m_model": m_model, "n_layers": n_layers, "hidden_dim": hidden_dim, "dropout": dropout,
            "epochs": epochs, "lr": lr, "batch_size": batch_size, "flip_prob": flip_prob, "min_value": min_value}




class Config(Enum):
    PRE = "pre"
    our = "our"
    both = "both"
