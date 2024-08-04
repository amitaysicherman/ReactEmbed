import os
from enum import Enum

import torch

from common.data_types import Config
from common.path_manager import fuse_path
from contrastive_learning.model import MultiModalLinearConfig, MiltyModalLinear


class DataType(Enum):
    MOLECULE = 'molecule_protein'
    PROTEIN = 'protein_protein'


def load_fuse_model(name):
    name = str(os.path.join(fuse_path, name))
    cp_names = os.listdir(name)
    cp_name = [x for x in cp_names if x.endswith(".pt")][0]
    print(f"Load model {name}/{cp_name}")
    cp_data = torch.load(f"{name}/{cp_name}", map_location=torch.device('cpu'))
    config_file = os.path.join(name, 'config.txt')
    config = MultiModalLinearConfig.load_from_file(config_file)
    dim = config.output_dim[0]
    model = MiltyModalLinear(config)
    model.load_state_dict(cp_data)
    model = model.eval()
    return model, dim


def get_layers(dims, dropout):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        # layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        if dropout > 0:
            layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
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

        if self.use_fuse:
            if fuse_model is None:
                self.fuse_model, dim = load_fuse_model(fuse_base)
            else:
                self.fuse_model = fuse_model
                dim = fuse_model.output_dim[0]
            self.fuse_dim = dim


class LinFuseModel(FuseModel):
    def __init__(self, input_dim: int, input_type: DataType, output_dim: int, conf: Config, fuse_model=None,
                 fuse_base="", n_layers=2, drop_out=0.0, hidden_dim=-1):
        super().__init__(conf, fuse_model, fuse_base)
        self.input_dim = 0
        if self.use_fuse:
            self.input_dim += self.fuse_dim
        if self.use_model:
            self.input_dim += input_dim

        if hidden_dim == -1:
            hidden_dim = self.input_dim

        self.dtype = input_type
        hidden_layers = [hidden_dim] * (n_layers - 1)
        self.layers = get_layers([self.input_dim] + hidden_layers + [output_dim], dropout=drop_out)

    def forward(self, data):
        x = []
        if self.use_fuse:
            x.append(self.fuse_model(data, self.dtype.value).detach())
        if self.use_model:
            x.append(data)
        x = torch.concat(x, dim=1)
        return self.layers(x)


class PairTransFuseModel(FuseModel):
    def __init__(self, input_dim_1: int, dtpye_1: DataType, input_dim_2: int, dtype_2: DataType, output_dim: int,
                 conf: Config,
                 hidden_dim=-1, n_layers=2, drop_out=0.5, fuse_model=None,
                 fuse_base=""):
        super().__init__(conf, fuse_model, fuse_base)

        self.input_dim = 0
        if self.use_fuse:
            self.input_dim += self.fuse_dim * 2
        if self.use_model:
            self.input_dim += input_dim_1 + input_dim_2
        self.x1_dtype = dtpye_1
        self.x2_dtype = dtype_2

        if hidden_dim == -1:
            hidden_dim = self.input_dim

        hidden_layers = [hidden_dim] * (n_layers - 1)
        self.layers = get_layers([self.input_dim] + hidden_layers + [output_dim], dropout=drop_out)
        # if self.use_fuse:
        #     self.x1_fuse_linear = torch.nn.Linear(self.fuse_dim, hidden_dim)
        #     self.x2_fuse_linear = torch.nn.Linear(self.fuse_dim, hidden_dim)
        # if self.use_model:
        #     self.x1_model_linear = torch.nn.Linear(input_dim_1, hidden_dim)
        #     self.x2_model_linear = torch.nn.Linear(input_dim_2, hidden_dim)
        # nhead = 2
        # encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
        #                                                  dim_feedforward=hidden_dim * 2,
        #                                                  batch_first=True, dropout=drop_out)
        # self.trans = torch.nn.Sequential(
        #     torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers),
        #     torch.nn.Linear(hidden_dim, output_dim)
        # )
        # self.x1_type = dtpye_1
        # self.x2_type = dtype_2

    def forward(self, x1, x2):
        x = []
        if self.use_fuse:
            x1_fuse = self.fuse_model(x1, self.x1_dtype.value).detach()
            x.append(x1_fuse)
            x2_fuse = self.fuse_model(x2, self.x2_dtype.value).detach()
            x.append(x2_fuse)
        if self.use_model:
            x.append(x1)
            x.append(x2)
        # if self.use_fuse:
        #     x1_fuse = self.fuse_model(x1, self.x1_type.value).detach()
        #     x.append(self.x1_fuse_linear(x1_fuse))
        #     x2_fuse = self.fuse_model(x2, self.x2_type.value).detach()
        #     x.append(self.x2_fuse_linear(x2_fuse))
        # if self.use_model:
        #     x1_model = self.x1_model_linear(x1)
        #     x.append(x1_model)
        #     x2_model = self.x2_model_linear(x2)
        #     x.append(x2_model)

        # x = torch.stack(x, dim=1)
        x = torch.cat(x, dim=1)
        return self.layers(x)
        # return self.trans(x).mean(dim=1)
