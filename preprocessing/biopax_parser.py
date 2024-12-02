import os
import time
import xml.etree.ElementTree as ET

import pybiopax
import requests
from pybiopax.biopax import BiochemicalReaction
from tqdm import tqdm

from common.path_manager import data_path, reactions_file, proteins_file, molecules_file


def get_req(url: str, to_json=False):
    for i in range(3):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json() if to_json else response.text
        print(f"Failed to retrieve url ({i}): {url}")
        time.sleep(1)
    return {} if to_json else ""


def from_second_line(seq):
    lines = seq.split("\n")
    return "".join(lines[1:]) if len(lines) > 1 else ""


def get_smiles_from_chebi(chebi_id, default_seq=""):
    chebi_id = chebi_id.replace("CHEBI:", "")
    url = f"https://www.ebi.ac.uk/chebi/saveStructure.do?xml=true&chebiId={chebi_id}&imageId=0"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            root = ET.fromstring(response.content)
            smiles_tag = root.find('.//SMILES')
            return smiles_tag.text if smiles_tag is not None else default_seq
        except ET.ParseError:
            print(f"Failed to parse SMILES for {chebi_id}")
    else:
        print(f"Failed to retrieve SMILES for {chebi_id}")
    return default_seq


def get_sequence(identifier, db_name):
    db_name = db_name.lower()
    default_seq = ""

    db_handlers = {
        "ensembl": lambda id: get_req(
            f"https://rest.ensembl.org/sequence/id/{id}?content-type=text/plain;species=human"),
        "embl": lambda id: from_second_line(get_req(f"https://www.ebi.ac.uk/ena/browser/api/fasta/{id}")),
        "uniprot": lambda id: from_second_line(get_req(f"https://www.uniprot.org/uniprot/{id}.fasta")),
        "uniprot isoform": lambda id: from_second_line(get_req(f"https://www.uniprot.org/uniprot/{id}.fasta")),
        "chebi": lambda id: get_smiles_from_chebi(id, default_seq),
        "guide to pharmacology": lambda id: get_req(
            f"https://www.guidetopharmacology.org/services/ligands/{id}/structure", to_json=True).get("smiles",
                                                                                                      default_seq),
        "pubchem compound": lambda id: get_req(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/property/CanonicalSMILES/JSON",
            to_json=True).get("CanonicalSMILES", default_seq),
    }
    handler = db_handlers.get(db_name)
    if handler is None:
        raise ValueError(f"Unknown database name: {db_name}")

    return handler(identifier)


PROTEIN = "protein"
MOLECULE = "molecule"


def db_to_type(db_name):
    proteins_data_bases = ["embl", "ensembl", "uniprot", "uniprot isoform"]
    molecules_data_bases = ["chebi", "pubchem compound", "guide to pharmacology"]
    db_name = db_name.lower()
    if db_name in proteins_data_bases:
        return PROTEIN
    elif db_name in molecules_data_bases:
        return MOLECULE


def element_parser(element: pybiopax.biopax.PhysicalEntity):
    if not hasattr(element, "entity_reference") or not hasattr(element.entity_reference, "xref"):
        if hasattr(element, "xref"):
            for xref in element.xref:
                if "Reactome" not in xref.db:
                    ref_db = xref.db
                    ref_id = xref.id
                    break
            else:
                ref_db = "0"
                ref_id = element.display_name
        else:
            ref_db = "0"
            ref_id = element.display_name

    elif len(element.entity_reference.xref) > 1:
        print(len(element.entity_reference.xref), "xrefs")
        ref_db = element.entity_reference.xref[0].db
        ref_id = element.entity_reference.xref[0].id
    else:
        ref_db = element.entity_reference.xref[0].db
        ref_id = element.entity_reference.xref[0].id
    return ref_db, ref_id


def get_all_elements(entity):
    elements = []
    if entity.member_physical_entity:
        for entity in entity.member_physical_entity:
            elements.extend(get_all_elements(entity))
    if isinstance(entity, pybiopax.biopax.Complex):
        for entity in entity.component:
            elements.extend(get_all_elements(entity))
    elif isinstance(entity, pybiopax.biopax.PhysicalEntity):
        elements.append(element_parser(entity))
    else:
        print("Unknown entity", type(entity))
    return elements


def elements_to_prot_mols(elements, proteins_to_id, molecules_to_id):
    proteins = []
    molecules = []
    for db, db_id in elements:
        type_ = db_to_type(db)
        if type_ == PROTEIN:
            if db_id not in proteins_to_id:
                proteins_to_id[(db, db_id)] = len(proteins_to_id)
            proteins.append(proteins_to_id[(db, db_id)])
        elif type_ == MOLECULE:
            if db_id not in molecules_to_id:
                molecules_to_id[(db, db_id)] = len(molecules_to_id)
            molecules.append(molecules_to_id[(db, db_id)])
    return proteins, molecules


def save_all_sequences(data_dict, output_file):
    from joblib import Parallel, delayed

    def fetch_sequence(db, db_id):
        return get_sequence(db_id, db)

    all_seq = Parallel(n_jobs=-1)(
        delayed(fetch_sequence)(db, db_id) for (db, db_id), _ in tqdm(data_dict.items(), desc="Fetching sequences"))
    with open(output_file, "w") as f:
        f.write("\n".join(all_seq))


def main():
    proteins_to_id = {}
    molecules_to_id = {}

    input_file = os.path.join(data_path, "biopax", "Homo_sapiens.owl")
    if os.path.exists(reactions_file):
        os.remove(reactions_file)
    model = pybiopax.model_from_owl_file(input_file)
    reactions = []
    for i, reaction in tqdm(enumerate(model.get_objects_by_type(pybiopax.biopax.BiochemicalReaction))):
        assert reaction.conversion_direction == "LEFT-TO-RIGHT"
        elements = []
        for entity in reaction.left:
            elements.extend(get_all_elements(entity))

        for entity in reaction.right:
            elements.extend(get_all_elements(entity))
        proteins, molecules = elements_to_prot_mols(elements, proteins_to_id, molecules_to_id)
        proteins, molecules = list(set(proteins)), list(set(molecules))
        reactions.append(",".join([str(p) for p in proteins]) + " " + ",".join([str(m) for m in molecules]))
    with open(reactions_file, "w") as f:
        f.write("\n".join(reactions))
    with open(proteins_file, "w") as f:
        for k, v in proteins_to_id.items():
            f.write(f'{k[0]} {k[1]} {v}\n')
    with open(molecules_file, "w") as f:
        for k, v in molecules_to_id.items():
            f.write(f'{k[0]} {k[1]} {v}\n')
    save_all_sequences(proteins_to_id, proteins_file.replace(".txt", "_sequences.txt"))
    save_all_sequences(molecules_to_id, molecules_file.replace(".txt", "_sequences.txt"))


if __name__ == "__main__":
    main()
