import dataclasses
from typing import List

import torch
from torch import nn as nn
from torch.nn import functional as F


@dataclasses.dataclass
class MultiModalLinearConfig:
    embedding_dim: List[int]
    n_layers: int
    names: List[str]
    hidden_dim: int
    output_dim: int
    dropout: float
    normalize_last: int

    def save_to_file(self, file_name):
        with open(file_name, "w") as f:
            for k, v in dataclasses.asdict(self).items():
                if isinstance(v, list) or isinstance(v, tuple):
                    if isinstance(v[0], tuple):
                        v = ["_".join([str(x) for x in y]) for y in v]
                    v = ",".join([str(x) for x in v])
                f.write(f"{k}={v}\n")

    @staticmethod
    def load_from_file(file_name):
        with open(file_name) as f:
            data = {}
            for line in f:
                k, v = line.strip().split("=")
                if k == "names":
                    v = v.split(",")
                data[k] = v
        return MultiModalLinearConfig(embedding_dim=[int(x) for x in data["embedding_dim"].split(",")],
                                      n_layers=int(data["n_layers"]), names=data["names"],
                                      hidden_dim=int(data["hidden_dim"]),
                                      output_dim=int(data["output_dim"]),
                                      dropout=float(data["dropout"]), normalize_last=int(data["normalize_last"]))


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


class MiltyModalLinear(nn.Module):
    def __init__(self, config: MultiModalLinearConfig):
        super(MiltyModalLinear, self).__init__()
        self.names = config.names
        self.normalize_last = config.normalize_last
        if config.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.layers_dict = nn.ModuleDict()
        self.output_dim = config.output_dim
        for name, input_dim in zip(self.names, config.embedding_dim):
            dims = [input_dim] + [config.hidden_dim] * (config.n_layers - 1) + [self.output_dim]
            self.layers_dict[name] = get_layers(dims, config.dropout)

    def forward(self, x, type_):
        # check the device of the input tensor
        x = self.layers_dict[type_](x)
        if self.normalize_last:
            return F.normalize(x, dim=-1)
        else:
            return x
