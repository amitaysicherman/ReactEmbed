import os
from enum import Enum

import torch

from common.utils import Config
from contrastive_learning.model import ReactEmbedConfig, ReactEmbedModel


class DataType(Enum):
    MOLECULE = 'M'
    PROTEIN = 'P'


def load_fuse_model(name):
    cp_data = torch.load(f"{name}/model.pt", map_location=torch.device('cpu'))
    config_file = os.path.join(name, 'config.txt')
    config = ReactEmbedConfig.load_from_file(config_file)
    dim = config.p_dim + config.m_dim
    model = ReactEmbedModel(config)
    model.load_state_dict(cp_data)
    model = model.eval()
    return model, dim


def get_layers(dims, dropout):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
    return layers


class FuseModel(torch.nn.Module):
    def __init__(self, conf: Config, fuse_model=None, fuse_base=""):
        super().__init__()
        if conf == Config.both:
            self.use_fuse = True
            self.use_model = True
        elif conf == Config.PRE:
            self.use_fuse = False
            self.use_model = True
        else:
            self.use_fuse = True
            self.use_model = False
        print(f"use_fuse: {self.use_fuse}, use_model: {self.use_model}")
        if self.use_fuse:
            if fuse_model is None:
                self.fuse_model, dim = load_fuse_model(fuse_base)
            else:
                self.fuse_model = fuse_model
                dim = fuse_model.config.p_dim + fuse_model.config.m_dim
            self.fuse_dim = dim


class LinFuseModel(FuseModel):
    def __init__(self, input_dim_1: int, dtype_1: DataType, output_dim: int, conf: Config,
                 fuse_model=None, fuse_base="", n_layers=2, drop_out=0.0, hidden_dim=-1):
        super().__init__(conf, fuse_model, fuse_base)
        self.dtype = dtype_1
        # Separate projection layers for each input
        if self.use_model:
            self.v1_projection = torch.nn.Linear(input_dim_1, hidden_dim)
        if self.use_fuse:
            self.v2_projection = torch.nn.Linear(self.fuse_dim, hidden_dim)

        # Gating mechanism
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, 1),
            torch.nn.Sigmoid()
        ) if self.use_fuse and self.use_model else None

        # Final layers
        hidden_layers = [hidden_dim] * (n_layers - 1)
        self.layers = get_layers([hidden_dim] + hidden_layers + [output_dim], dropout=drop_out)

    def forward(self, data):
        v1 = self.v1_projection(data) if self.use_model else None
        v2 = self.v2_projection(
            self.fuse_model.dual_forward(data, self.dtype.value).detach()) if self.use_fuse else None

        if self.use_fuse and self.use_model:
            # Compute attention/gating
            gate = self.gate(torch.cat([v1, v2], dim=1))
            print(gate)
            x = (1 - gate) * v1 + gate * v2
        else:
            x = v1 if self.use_model else v2

        return self.layers(x)

class PairsFuseModel(FuseModel):
    def __init__(self, input_dim_1: int, dtype_1: DataType, input_dim_2: int, dtype_2: DataType,
                 output_dim: int, conf: Config, hidden_dim=-1, n_layers=2, drop_out=0.5,
                 fuse_model=None, fuse_base=""):
        super().__init__(conf, fuse_model, fuse_base)
        self.x1_dtype = dtype_1
        self.x2_dtype = dtype_2
        if hidden_dim == -1:
            hidden_dim = max(input_dim_1, input_dim_2)

        # Projection layers for original inputs
        if self.use_model:
            self.v1_projection = torch.nn.Linear(input_dim_1, hidden_dim)
            self.v2_projection = torch.nn.Linear(input_dim_2, hidden_dim)

        # Projection layers for fuse model outputs
        if self.use_fuse:
            self.fuse1_projection = torch.nn.Linear(self.fuse_dim, hidden_dim)
            self.fuse2_projection = torch.nn.Linear(self.fuse_dim, hidden_dim)

        # Gating mechanisms - one for each pair
        if self.use_fuse and self.use_model:
            self.gate1 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim),
                torch.nn.Sigmoid()
            )
            self.gate2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim),
                torch.nn.Sigmoid()
            )

        # Final layers
        hidden_layers = [hidden_dim * 2] * (n_layers - 1)  # *2 because we'll concatenate pair features
        self.layers = get_layers([hidden_dim * 2] + hidden_layers + [output_dim], dropout=drop_out)

    def forward(self, x1, x2):
        # Process first input
        if self.use_model:
            v1 = self.v1_projection(x1)
        if self.use_fuse:
            v1_fuse = self.fuse1_projection(self.fuse_model.dual_forward(x1, self.x1_dtype.value).detach())

        # Process second input
        if self.use_model:
            v2 = self.v2_projection(x2)
        if self.use_fuse:
            v2_fuse = self.fuse2_projection(self.fuse_model.dual_forward(x2, self.x2_dtype.value).detach())

        # Apply gating if using both original and fuse
        if self.use_fuse and self.use_model:
            # Gate for first pair
            gate1 = self.gate1(torch.cat([v1, v1_fuse], dim=1))
            feat1 = v1 + gate1 * v1_fuse

            # Gate for second pair
            gate2 = self.gate2(torch.cat([v2, v2_fuse], dim=1))
            feat2 = v2 + gate2 * v2_fuse
        else:
            # If using only one type of input
            feat1 = v1 if self.use_model else v1_fuse
            feat2 = v2 if self.use_model else v2_fuse

        # Combine pair features
        x = torch.cat([feat1, feat2], dim=1)
        return self.layers(x)