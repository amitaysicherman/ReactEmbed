import dataclasses
from typing import List

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from common.data_types import DNA


@dataclasses.dataclass
class MultiModalLinearConfig:
    embedding_dim: List[int]
    n_layers: int
    names: List[str]
    hidden_dim: int
    output_dim: List[int]
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
                    v = [tuple(v_.split("_")) for v_ in v.split(",")]
                data[k] = v
        return MultiModalLinearConfig(embedding_dim=[int(x) for x in data["embedding_dim"].split(",")],
                                      n_layers=int(data["n_layers"]), names=data["names"],
                                      hidden_dim=int(data["hidden_dim"]),
                                      output_dim=[int(x) for x in data["output_dim"].split(",")],
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
        self.names = ["_".join(x) if isinstance(x, tuple) else x for x in config.names]
        self.normalize_last = config.normalize_last
        if config.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        self.layers_dict = nn.ModuleDict()
        self.output_dim = config.output_dim
        for name, input_dim, output_dim in zip(self.names, config.embedding_dim, config.output_dim):
            dims = [input_dim] + [config.hidden_dim] * (config.n_layers - 1) + [output_dim]
            self.layers_dict[name] = get_layers(dims, config.dropout)

    def have_type(self, type_):
        if isinstance(type_, tuple):
            type_ = "_".join(type_)
        return type_ in self.names

    def forward(self, x, type_):
        if isinstance(type_, tuple):
            type_ = "_".join(type_)

        if not self.have_type(type_):
            type_ = type_.split("_")[0]

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x).float()
        x = self.layers_dict[type_](x)
        if self.normalize_last:
            return F.normalize(x, dim=-1)
        else:
            return x


def concat_all_to_one_typs(model: MiltyModalLinear, x, src_type):
    res = []
    for name in model.names:
        src_name, dst_name = name.split("_")
        if src_name == src_type:
            if model.have_type(name):
                res.append(model(x, name))
            else:
                print(f"Warning: model does not have type {name}")
                res.append(x)
    return torch.cat(res, dim=-1)


def apply_model(model: MiltyModalLinear, x, type_):
    if type_ == DNA:
        return torch.Tensor(x)
    if "_" in model.names[0]:
        return concat_all_to_one_typs(model, x, type_)
    else:
        if not model.have_type(type_):
            print(f"Warning: model does not have type {type_}")
            return torch.Tensor(x)
        return model(x, type_)
