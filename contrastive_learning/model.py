import dataclasses

import torch
from torch import nn as nn
from torch.nn import functional as F


@dataclasses.dataclass
class ReactEmbedConfig:
    p_dim: int
    m_dim: int
    shared_dim: int
    n_layers: int
    hidden_dim: int
    dropout: float
    normalize_last: int = 0

    def save_to_file(self, file_name):
        with open(file_name, "w") as f:
            f.write(f"p_dim={self.p_dim}\n")
            f.write(f"m_dim={self.m_dim}\n")
            f.write(f"shared_dim={self.shared_dim}\n")
            f.write(f"n_layers={self.n_layers}\n")
            f.write(f"hidden_dim={self.hidden_dim}\n")
            f.write(f"dropout={self.dropout}\n")
            f.write(f"normalize_last={self.normalize_last}\n")

    @staticmethod
    def load_from_file(file_name):
        with open(file_name) as f:
            lines = f.readlines()
        d = {}
        for line in lines:
            k, v = line.strip().split("=")
            d[k] = float(v) if "." in v else int(v)
        return ReactEmbedConfig(**d)


def get_layers(dims, dropout=0.0):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        if dropout > 0:
            layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
    return layers


class ReactEmbedModel(nn.Module):
    def __init__(self, config: ReactEmbedConfig):
        super(ReactEmbedModel, self).__init__()
        self.config = config
        if config.n_layers < 1:
            raise ValueError("n_layers must be at least 1")

        # Create just two transformation networks: P→shared and M→shared
        self.p_to_shared = get_layers(
            [config.p_dim] + [config.hidden_dim] * (config.n_layers - 1) + [config.shared_dim],
            config.dropout
        )
        self.m_to_shared = get_layers(
            [config.m_dim] + [config.hidden_dim] * (config.n_layers - 1) + [config.shared_dim],
            config.dropout
        )

    def forward(self, x, type_):
        """
        Transform input to shared space based on type
        type_ should be either 'P' or 'M'
        """
        if type_ == "P":
            x = self.p_to_shared(x)
        elif type_ == "M":
            x = self.m_to_shared(x)
        else:
            raise ValueError(f"Invalid type: {type_}")

        if self.config.normalize_last:
            return F.normalize(x, dim=-1)
        return x
