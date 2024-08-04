import os
from collections import defaultdict

from tqdm import tqdm

from common.data_types import TEXT, LOCATION, DATA_TYPES
from common.path_manager import reactions_file, item_path
from common.utils import db_to_type
from common.utils import reaction_from_str

indexes = {dt: defaultdict(int) for dt in DATA_TYPES}
with open(reactions_file) as f:
    lines = f.readlines()
reactions = []

for line in tqdm(lines):
    reaction = reaction_from_str(line)
    catalyst_entities = sum([c.entities for c in reaction.catalysis], [])
    all_entities = reaction.inputs + reaction.outputs + catalyst_entities
    for entity in all_entities:
        dtype = db_to_type(entity.db)
        indexes[dtype][entity.get_db_identifier()] += 1
        indexes[LOCATION][entity.location] += 1
        for mod in entity.modifications:
            mod = f"TEXT@{mod}"
            indexes[TEXT][mod] += 1
    for catalyst in reaction.catalysis:
        catalyst_activity = f"GO@{catalyst.activity}"
        indexes[TEXT][catalyst_activity] += 1

for k, v in indexes.items():
    output_file = os.path.join(item_path, f"{k}.txt")
    with open(output_file, "w") as f:
        for k, v in v.items():
            f.write(f"{k}@{v}\n")
