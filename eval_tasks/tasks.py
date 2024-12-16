import enum
from dataclasses import dataclass

import torch
from torch import nn
from torchdrug import datasets

from eval_tasks.models import DataType, LinFuseModel, PairsFuseModel


class PrepType(enum.Enum):
    torchdrug = "torchdrug"
    drugtarget = "drugtarget"


@dataclass
class Task:
    name: str
    dataset: object
    model: torch.nn.Module
    criterion: object
    dtype1: DataType
    output_dim: int
    dtype2: DataType = None
    prep_type: PrepType = PrepType.torchdrug


def mse_metric(output, target):
    squared_diff = (output - target) ** 2
    mse = torch.mean(squared_diff)
    return -1 * mse


name_to_task = {
    # Molecule tasks
    "BACE": Task("BACE", datasets.BACE, LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 1),
    "BBBP": Task("BBBP", datasets.BBBP, LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 1),
    "CEP": Task("CEP", datasets.CEP, LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "ClinTox": Task("ClinTox", datasets.ClinTox, LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 2),
    "Delaney": Task("Delaney", datasets.Delaney, LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "FreeSolv": Task("FreeSolv", datasets.FreeSolv, LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "HIV": Task("HIV", datasets.HIV, LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 1),
    "Lipophilicity": Task("Lipophilicity", datasets.Lipophilicity, LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "Malaria": Task("Malaria", datasets.Malaria, LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "SIDER": Task("SIDER", datasets.SIDER, LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 27),
    "Tox21": Task("Tox21", datasets.Tox21, LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 12),

    # Protein tasks
    "BetaLactamase": Task("BetaLactamase", datasets.BetaLactamase, LinFuseModel, nn.MSELoss, DataType.PROTEIN, 1),
    "Fluorescence": Task("Fluorescence", datasets.Fluorescence, LinFuseModel, nn.MSELoss, DataType.PROTEIN, 1),
    "Stability": Task("Stability", datasets.Stability, LinFuseModel, nn.MSELoss, DataType.PROTEIN, 1),
    "BinaryLocalization": Task("BinaryLocalization", datasets.BinaryLocalization, LinFuseModel, nn.BCEWithLogitsLoss,
                               DataType.PROTEIN, 1),
    # Pairs tasks
    "HumanPPI": Task("HumanPPI", datasets.HumanPPI, PairsFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1,
                     DataType.PROTEIN),
    "YeastPPI": Task("YeastPPI", datasets.YeastPPI, PairsFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1,
                     DataType.PROTEIN),
    "PPIAffinity": Task("PPIAffinity", datasets.PPIAffinity, PairsFuseModel, nn.MSELoss, DataType.PROTEIN, 1,
                        DataType.PROTEIN),
    "BindingDB": Task("BindingDB", datasets.BindingDB, PairsFuseModel, nn.MSELoss, DataType.PROTEIN, 1,
                      DataType.MOLECULE),
    "PDBBind": Task("PDBBind", datasets.PDBBind, PairsFuseModel, nn.MSELoss, DataType.PROTEIN, 1,
                    DataType.MOLECULE),

    # Drug target tasks
    "DrugBank": Task("DrugBank", None, PairsFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1, DataType.MOLECULE,
                     PrepType.drugtarget),
    "Davis": Task("Davis", None, PairsFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1, DataType.MOLECULE,
                  PrepType.drugtarget),
}
