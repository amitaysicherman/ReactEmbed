import dataclasses

import torch
from torch import nn as nn
from torch.nn import functional as F


@dataclasses.dataclass
class ReactEmbedConfig:
    p_dim: int
    m_dim: int
    n_layers: int
    hidden_dim: int
    dropout: float
    normalize_last: int = 1

    def save_to_file(self, file_name):
        with open(file_name, "w") as f:
            f.write(f"p_dim={self.p_dim}\n")
            f.write(f"m_dim={self.m_dim}\n")
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
        layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
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
        self.layers_dict = nn.ModuleDict()
        for src_dim, src in zip([config.p_dim, config.m_dim], ["P", "M"]):
            for dst_dim, dst in zip([config.p_dim, config.m_dim], ["P", "M"]):
                name = f"{src}-{dst}"
                dims = [src_dim] + [config.hidden_dim] * (config.n_layers - 1) + [dst_dim]
                self.layers_dict[name] = get_layers(dims, config.dropout)

    def forward(self, x, type_):
        # check the device of the input tensor
        x = self.layers_dict[type_](x)
        if self.config.normalize_last:
            return F.normalize(x, dim=-1)
        else:
            return x

    def dual_forward(self, x, type_1):
        if type_1 == "P":
            y1 = self.forward(x, "P-P")
            y2 = self.forward(x, "P-M")
        else:
            y1 = self.forward(x, "M-P")
            y2 = self.forward(x, "M-M")
        # concatenate the two outputs
        y = torch.cat([y1, y2], dim=-1)
        return y
