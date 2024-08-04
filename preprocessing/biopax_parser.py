import datetime
from collections import defaultdict
from typing import List, Dict

import pybiopax
from pybiopax.biopax import BiochemicalReaction
from pybiopax.biopax.base import Pathway
from pybiopax.biopax.model import BioPaxModel
from pybiopax.biopax.util import RelationshipXref as RelXref
from tqdm import tqdm

from common.data_types import Entity, CatalystOBJ, Reaction

max_complex_id = 1


def feature_parser(feature) -> str:
    if isinstance(feature, pybiopax.biopax.FragmentFeature):
        return ""
    if isinstance(feature, pybiopax.biopax.ModificationFeature):
        if not hasattr(feature, "modification_type") or feature.modification_type is None:
            return feature.comment[0].split(" ")[0]
        return feature.modification_type.term[0]


def element_parser(element: pybiopax.biopax.PhysicalEntity, complex_location=None, complex_id=0):
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

    elif len(element.entity_reference.xref) > 1:
        print(len(element.entity_reference.xref), "xrefs")
        ref_db = element.entity_reference.xref[0].db
        ref_id = element.entity_reference.xref[0].id
    else:
        ref_db = element.entity_reference.xref[0].db
        ref_id = element.entity_reference.xref[0].id

    name = element.display_name
    if complex_location:
        location = complex_location
    else:
        if not hasattr(element, "cellular_location") or not hasattr(element.cellular_location, "term"):
            location = ""
        else:
            location = element.cellular_location.term[0]

    features = list(element.feature)
    modifications = [feature_parser(f) for f in features]
    modifications = [f for f in modifications if f != ""]
    modifications = tuple(modifications)
    return Entity(name, ref_db, ref_id, location, modifications, complex_id)


def add_protein_or_complex(entity, complex_location=None, complex_id=0):
    elements = []
    if entity.member_physical_entity:
        entity = entity.member_physical_entity[0]  # TODO: get all set elements

    if isinstance(entity, pybiopax.biopax.Complex):

        if complex_location is None:
            global max_complex_id
            complex_id = max_complex_id
            max_complex_id += 1
            complex_location = entity.cellular_location.term[0]
        for entity in entity.component:
            elements.extend(add_protein_or_complex(entity, complex_location, complex_id))
    elif isinstance(entity, pybiopax.biopax.PhysicalEntity):
        elements.append(element_parser(entity, complex_location, complex_id))

    else:
        print("Unknown entity", type(entity))
    return elements


def catalysis_parser(catalysis_list: List[pybiopax.biopax.Catalysis]) -> List[CatalystOBJ]:
    results = []
    for catalysis in catalysis_list:
        assert len(catalysis.controller) == 1, "More than one controller"
        catalysis_entities = add_protein_or_complex(catalysis.controller[0])
        catalysis_activity = catalysis.xref[0].id
        results.append(CatalystOBJ(catalysis_entities, catalysis_activity))
    return results


def get_reactome_id(reaction: BiochemicalReaction) -> str:
    if not hasattr(reaction, "xref"):
        return "0"
    for xref in reaction.xref:
        if "Reactome" in xref.db:
            return xref.id
    return "0"


def get_reaction_date(reaction: BiochemicalReaction, format='%Y-%m-%d',
                      default_date=datetime.date(1970, 1, 1)) -> datetime.date:
    if not hasattr(reaction, "comment"):
        return default_date
    comments = [c.split(", ")[-1].split(" ")[0] for c in reaction.comment if "Authored" in c]
    if not comments:
        return default_date
    date_str = comments[0]
    try:
        date = datetime.datetime.strptime(date_str, format).date()
    except:
        date = default_date
        print(f"Error in date: {date_str}")

    return date


def reactions_to_biological_process(model: BioPaxModel) -> Dict[str, List[str]]:
    pathways = list(model.get_objects_by_type(Pathway))
    inv_mapping = {}
    for pathway in pathways:
        for c in pathway.pathway_component:
            inv_mapping[c] = pathway
    mappings = defaultdict(list)
    for reaction in model.get_objects_by_type(BiochemicalReaction):
        parent = reaction
        while True:
            if parent not in inv_mapping:
                go_relations = []
                break
            parent = inv_mapping[parent]
            go_relations = [ref.id for ref in parent.xref if isinstance(ref, RelXref) and ref.db == "GENE ONTOLOGY"]
            if len(go_relations) > 0:
                break

        mappings[reaction.uid].extend(go_relations)
    return mappings


if __name__ == "__main__":
    from common.path_manager import data_path, reactions_file
    import os

    input_file = os.path.join(data_path, "biopax", "Homo_sapiens.owl")
    output_file = reactions_file
    if os.path.exists(output_file):
        os.remove(output_file)
    model = pybiopax.model_from_owl_file(input_file)
    reaction_to_go = reactions_to_biological_process(model)
    reactions = list(model.get_objects_by_type(pybiopax.biopax.BiochemicalReaction))
    all_catalysis = list(model.get_objects_by_type(pybiopax.biopax.Catalysis))
    print(len(reactions))
    for i, reaction in tqdm(enumerate(reactions)):
        assert reaction.conversion_direction == "LEFT-TO-RIGHT"
        left_elements = []
        for entity in reaction.left:
            left_elements.extend(add_protein_or_complex(entity))

        right_elements = []
        for entity in reaction.right:
            right_elements.extend(add_protein_or_complex(entity))
        catalys_activities = [c for c in all_catalysis if c.controlled == reaction]
        catalys_activities = catalysis_parser(catalys_activities)
        date = get_reaction_date(reaction)
        reactome_id = get_reactome_id(reaction)
        biological_process = reaction_to_go[reaction.uid]
        reaction_obj = Reaction(reaction.name[0], left_elements, right_elements, catalys_activities, date, reactome_id,
                                biological_process)
        with open(output_file, "a") as f:
            f.write(f'{reaction_obj.to_dict()}\n')
