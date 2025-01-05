import enum
from dataclasses import dataclass

import torch
from torch import nn

from eval_tasks.models import DataType, LinFuseModel, PairsFuseModel


class PrepType(enum.Enum):
    torchdrug = "torchdrug"
    drugtarget = "drugtarget"


@dataclass
class Task:
    name: str
    model: torch.nn.Module
    criterion: object
    dtype1: DataType
    output_dim: int
    dtype2: DataType = None
    prep_type: PrepType = PrepType.torchdrug
    branch: str = None


def mse_metric(output, target):
    squared_diff = (output - target) ** 2
    mse = torch.mean(squared_diff)
    return -1 * mse


name_to_task = {
    # Molecule tasks
    "BACE": Task("BACE", LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 1),
    "BBBP": Task("BBBP", LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 1),
    "CEP": Task("CEP", LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "ClinTox": Task("ClinTox", LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 2),
    "Delaney": Task("Delaney", LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "FreeSolv": Task("FreeSolv", LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "HIV": Task("HIV", LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 1),
    "Lipophilicity": Task("Lipophilicity", LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "Malaria": Task("Malaria", LinFuseModel, nn.MSELoss, DataType.MOLECULE, 1),
    "SIDER": Task("SIDER", LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 27),
    "Tox21": Task("Tox21", LinFuseModel, nn.BCEWithLogitsLoss, DataType.MOLECULE, 12),

    # Protein tasks
    "BetaLactamase": Task("BetaLactamase", LinFuseModel, nn.MSELoss, DataType.PROTEIN, 1),
    "Fluorescence": Task("Fluorescence", LinFuseModel, nn.MSELoss, DataType.PROTEIN, 1),
    "Stability": Task("Stability", LinFuseModel, nn.MSELoss, DataType.PROTEIN, 1),
    "Solubility": Task("Solubility", LinFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1),
    "BinaryLocalization": Task("BinaryLocalization", LinFuseModel, nn.BCEWithLogitsLoss,
                               DataType.PROTEIN, 1),
    "SubcellularLocalization": Task("SubcellularLocalization", LinFuseModel, nn.BCEWithLogitsLoss,
                                    DataType.PROTEIN, 10),
    "EnzymeCommission": Task("EnzymeCommission", LinFuseModel, nn.BCEWithLogitsLoss,
                             DataType.PROTEIN, 6),
    "GeneOntologyMF": Task("GeneOntologyMF", LinFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1, branch="MF"),
    "GeneOntologyBP": Task("GeneOntologyBP", LinFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1, branch="BP"),
    "GeneOntologyCC": Task("GeneOntologyCC", LinFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1, branch="CC"),

    # Pairs tasks
    "HumanPPI": Task("HumanPPI", PairsFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1,
                     DataType.PROTEIN),
    "YeastPPI": Task("YeastPPI", PairsFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1,
                     DataType.PROTEIN),
    "PPIAffinity": Task("PPIAffinity", PairsFuseModel, nn.MSELoss, DataType.PROTEIN, 1,
                        DataType.PROTEIN),
    "BindingDB": Task("BindingDB", PairsFuseModel, nn.MSELoss, DataType.PROTEIN, 1,
                      DataType.MOLECULE),
    "PDBBind": Task("PDBBind", PairsFuseModel, nn.MSELoss, DataType.PROTEIN, 1,
                    DataType.MOLECULE),

    # Drug target tasks
    "DrugBank": Task("DrugBank", PairsFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1, DataType.MOLECULE,
                     PrepType.drugtarget),
    "Davis": Task("Davis", PairsFuseModel, nn.BCEWithLogitsLoss, DataType.PROTEIN, 1, DataType.MOLECULE,
                  PrepType.drugtarget),
}


def task_to_metric(task, set=0):
    if task.criterion == nn.BCEWithLogitsLoss:
        return ['f1_max', "auprc"][set]
    else:
        return ['r2', "mse"][set]


if __name__ == '__main__':
    for name in name_to_task:
        print(name)
