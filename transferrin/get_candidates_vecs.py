import time

import numpy as np
import requests

from preprocessing.seq_to_vec import SeqToVec

output_seq_file = "transferrin/can_seqs.txt"


def get_req(url: str, to_json=False, ret=3):
    for i in range(ret):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json() if to_json else response.text
        print(f"Failed to retrieve url ({i}): {url}")
        time.sleep(1)
    return {} if to_json else ""


def from_second_line(seq):
    lines = seq.split("\n")
    return "".join(lines[1:]) if len(lines) > 1 else ""


def get_uniprot_sequence(protein_id):
    return from_second_line(get_req(f"https://www.uniprot.org/uniprot/{protein_id}.fasta"))


name_to_id = {'GLUT1': 'P11166',
              'SMIT': 'P53794',
              'LAT1': 'Q01650',
              'EAAT 1': 'P43003',
              'MCT1': 'O60669',
              'FATP-1': 'Q6PCB7',
              'MCT8': 'P36021',
              'Thyroid T4 hormone': 'P10828',
              'OAT2': 'Q8IVM8',
              'SMVT': 'Q9Y289',
              'TfR': 'P02786',
              'IR': 'P06213',
              'LEP-R': 'P48357',
              'LRP1': 'Q07954',
              'RAGE': 'Q15109'}

all_seqs = []
for name, protein_id in name_to_id.items():
    sequence = get_uniprot_sequence(protein_id)
    all_seqs.append(sequence)
    with open(output_seq_file, "a") as f:
        f.write(f"{name},{protein_id},{sequence}\n")

for p_model in ["ProtBert", "esm3-small", "esm3-medium"]:
    seq_to_vec = SeqToVec(model_name=p_model)
    vecs = seq_to_vec.lines_to_vecs(output_seq_file)
    np.save(f"transferrin/can_{p_model}.npy", vecs)
