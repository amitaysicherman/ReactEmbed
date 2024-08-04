import enum
from dataclasses import dataclass

import torch
from torch import nn
from torchdrug import datasets

from eval_tasks.models import DataType, LinFuseModel, PairTransFuseModel, FuseModel


class PrepType(enum.Enum):
    torchdrug = "torchdrug"
    drugtarget = "drugtarget"


@dataclass
class Task:
    name: str
    dataset: object
    model: FuseModel
    criterion: object
    metric: object
    dtype1: DataType
    output_dim: int
    dtype2: DataType = None
    prep_type: PrepType = PrepType.torchdrug
    n_layers: int = 1


def mse_metric(output, target):
    squared_diff = (output - target) ** 2
    mse = torch.mean(squared_diff)
    return -1 * mse


classification = "classification"
regression = "regression"

name_to_task = {
    "BetaLactamase": Task("BetaLactamase", datasets.BetaLactamase, LinFuseModel, nn.MSELoss, regression,
                          DataType.PROTEIN, 1),
    "Fluorescence": Task("Fluorescence", datasets.Fluorescence, LinFuseModel, nn.MSELoss, regression,
                         DataType.PROTEIN, 1),
    "Stability": Task("Stability", datasets.Stability, LinFuseModel, nn.MSELoss, regression,
                      DataType.PROTEIN, 1),
    "HumanPPI": Task("HumanPPI", datasets.HumanPPI, PairTransFuseModel, nn.BCEWithLogitsLoss, classification,
                     DataType.PROTEIN, 1, DataType.PROTEIN),
    "BindingDB": Task("BindingDB", datasets.BindingDB, PairTransFuseModel, nn.MSELoss, regression, DataType.PROTEIN, 1,
                      DataType.MOLECULE),
    "BACE": Task("BACE", datasets.BACE, LinFuseModel, nn.BCEWithLogitsLoss, classification, DataType.MOLECULE,
                 1, n_layers=3),
    "BBBP": Task("BBBP", datasets.BBBP, LinFuseModel, nn.BCEWithLogitsLoss, classification, DataType.MOLECULE,
                 1, n_layers=3),
    "ClinTox": Task("ClinTox", datasets.ClinTox, LinFuseModel, nn.BCEWithLogitsLoss, classification,
                    DataType.MOLECULE,
                    2, n_layers=3),
    "SIDER": Task("SIDER", datasets.SIDER, LinFuseModel, nn.BCEWithLogitsLoss, classification,
                  DataType.MOLECULE, 27, n_layers=3),
    "DrugBank": Task("DrugBank", None, PairTransFuseModel, nn.BCEWithLogitsLoss, classification,
                     DataType.PROTEIN, 1, DataType.MOLECULE, PrepType.drugtarget),
    "Davis": Task("Davis", None, PairTransFuseModel, nn.BCEWithLogitsLoss, classification, DataType.PROTEIN,
                  1, DataType.MOLECULE, PrepType.drugtarget),
}
