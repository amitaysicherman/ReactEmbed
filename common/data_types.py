import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List


@dataclass
class Entity:
    name: str
    db: str
    db_id: str
    location: str
    modifications: tuple = ()
    complex_id: int = 0

    def get_db_identifier(self):
        return self.db + "@" + self.db_id

    def to_dict(self):
        return {"name": self.name, "db": self.db, "db_id": self.db_id, "location": self.location,
                "modifications": self.modifications, "complex_id": self.complex_id}


@dataclass
class CatalystOBJ:
    entities: List[Entity]
    activity: str

    def to_dict(self):
        return {"entities": [e.to_dict() for e in self.entities], "activity": self.activity}


class Reaction:
    def __init__(self, name, inputs: List[Entity], outputs: List[Entity], catalysis: List[CatalystOBJ],
                 date: datetime.date, reactome_id: int, biological_process: List[str]):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.catalysis = catalysis
        self.date = date
        self.reactome_id = reactome_id
        self.biological_process = biological_process

    def to_dict(self):
        return {"name": self.name, "inputs": [e.to_dict() for e in self.inputs],
                "outputs": [e.to_dict() for e in self.outputs], "catalysis": [c.to_dict() for c in self.catalysis],
                "date": f'{self.date.year}_{self.date.month}_{self.date.day}', "reactome_id": str(self.reactome_id),
                "biological_process": "_".join(self.biological_process)}


REACTION = "reaction"
COMPLEX = "complex"
UNKNOWN_ENTITY_TYPE = "_"
DNA = "dna"
PROTEIN = "protein"
MOLECULE = "molecule"
TEXT = "text"
EMBEDDING_DATA_TYPES = [PROTEIN, DNA, MOLECULE, TEXT]
LOCATION = "location"
DATA_TYPES = EMBEDDING_DATA_TYPES + [LOCATION] + [UNKNOWN_ENTITY_TYPE]
BIOLOGICAL_PROCESS = "bp"

NO_PRETRAINED_EMD = 0
PRETRAINED_EMD = 1
PRETRAINED_EMD_FUSE = 2

P_BFD = "ProtBert-BFD"
P_T5_XL = "ProtBertT5-xl"
ESM_1B = "ESM-1B"
ESM_2 = "ESM2"
ESM_3 = "ESM3"

PEBCHEM10M = "pebchem10m"
ROBERTA = "roberta"
CHEMBERTA = "chemberta"

NAME_TO_UI = {PEBCHEM10M: "PubChem", ROBERTA: "Roberta", CHEMBERTA: "ChemBerta", P_BFD: "ProtBertBFD",
              P_T5_XL: "ProtBertT5", ESM_1B: "ESM1", ESM_2: "ESM2", ESM_3: "ESM3", }

PROT_UI_ORDER = [ESM_2, P_BFD, ESM_1B, ESM_3, P_T5_XL]
MOL_UI_ORDER = [PEBCHEM10M, CHEMBERTA, ROBERTA]


class Config(Enum):
    PRE = "pre"
    our = "our"
    both = "both"
