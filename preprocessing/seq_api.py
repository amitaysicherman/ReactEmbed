import os
import time
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

from common.data_types import EMBEDDING_DATA_TYPES


def get_req(url: str, to_json=False):
    for i in range(3):
        response = requests.get(url)
        if response.status_code == 200:
            if to_json:
                return response.json()
            return response.text
        else:
            print(f"Failed to retrieve url ({i}): {url} ")
            time.sleep(1)

    if to_json:
        return {}
    return ""


def from_second_line(seq):
    seq = seq.split("\n")
    if len(seq) < 2:
        return ""
    return "".join(seq[1:])


def get_smiles_from_chebi(chebi_id, default_seq=""):
    chebi_id = chebi_id.replace("CHEBI:", "")
    url = f"https://www.ebi.ac.uk/chebi/saveStructure.do?xml=true&chebiId={chebi_id}&imageId=0"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            root = ET.fromstring(response.content)
            smiles_tag = root.find('.//SMILES')
            if smiles_tag is not None:
                return smiles_tag.text
            else:
                print(f"Failed to find SMILES for {chebi_id}")
                return default_seq
        except:
            print(f"Failed to find SMILES for {chebi_id}")

            return default_seq
    else:
        print(f"Failed to find SMILES for {chebi_id}")

        return default_seq


def get_sequence(identifier, db_name):
    db_name = db_name.lower()
    default_seq = ""

    # Define URLs and processing for each database
    db_handlers = {
        "ensembl": lambda id: get_req(
            f"https://rest.ensembl.org/sequence/id/{id}?content-type=text/plain;species=human"),
        "embl": lambda id: from_second_line(get_req(f"https://www.ebi.ac.uk/ena/browser/api/fasta/{id}")),
        "uniprot": lambda id: from_second_line(get_req(f"https://www.uniprot.org/uniprot/{id}.fasta")),
        "uniprot isoform": lambda id: from_second_line(get_req(f"https://www.uniprot.org/uniprot/{id}.fasta")),
        "chebi": lambda id: get_smiles_from_chebi(id, default_seq),
        "guide to pharmacology": lambda id: get_req(
            f"https://www.guidetopharmacology.org/services/ligands/{id}/structure",
            to_json=True
        ).get("smiles", default_seq),
        "go": lambda id: parse_go_response(
            get_req(f"https://api.geneontology.org/api/ontology/term/{id}", to_json=True)
        ),
        "ncbi nucleotide": lambda id: from_second_line(
            get_req(f"https://www.ncbi.nlm.nih.gov/nuccore/{id}?report=fasta&log$=seqview&format=text")),
        "pubchem compound": lambda id: get_req(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/property/CanonicalSMILES/JSON",
            to_json=True
        ).get("CanonicalSMILES", default_seq),
        "text": lambda id: id
    }

    def parse_go_response(json_resp):
        label = json_resp.get("label", "")
        definition = json_resp.get("definition", "")
        return f"{label}. {definition}" if label or definition else default_seq

    handler = db_handlers.get(db_name)
    if handler is None:
        raise ValueError(f"Unknown database name: {db_name}")

    return handler(identifier)


if __name__ == "__main__":
    from common.path_manager import item_path

    base_dir = item_path
    for dt in EMBEDDING_DATA_TYPES[2:]:
        fail_count = 0

        print(f"Processing {dt}")
        input_file = f"{base_dir}/{dt}.txt"
        output_file = f"{base_dir}/{dt}_sequences.txt"

        with open(input_file) as f:
            lines = f.read().splitlines()

        output_lines = [""] * len(lines)
        if os.path.exists(output_file):
            print(f"Output file already exists: {output_file}")
            with open(output_file) as f:
                output_lines = f.read().splitlines()
                output_lines = [line.strip() for line in output_lines]
                assert len(output_lines) == len(lines)
                missing_count = output_lines.count("")
                print(f"Missing {missing_count}({missing_count / len(lines):%}) sequences")

        seqs = []

        for line, output_line in tqdm(zip(lines, output_lines), total=len(lines)):
            if output_line:
                new_seq = output_line
            else:
                db_, id_, count_ = line.split("@")
                new_seq = get_sequence(id_, db_)
                if not new_seq:
                    fail_count += 1
            seqs.append(new_seq)
        print(f"Failed to retrieve {fail_count}({fail_count / len(lines):%}) sequences")

        with open(output_file, "w") as f:
            for seq in seqs:
                if "\n" in seq:
                    seq = seq.replace("\n", "\t")
                    print(seq)
                f.write(seq + "\n")
        print(f"Finished processing {dt}")
