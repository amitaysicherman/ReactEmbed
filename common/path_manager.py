import os
import pathlib

BASE_DIR = str(pathlib.Path(__file__).parent.parent.resolve())

data_path = os.path.join(BASE_DIR, "data")
os.makedirs(data_path, exist_ok=True)
item_path = os.path.join(data_path, "reactome")
os.makedirs(item_path, exist_ok=True)
model_path = os.path.join(data_path, "models")
os.makedirs(model_path, exist_ok=True)

scores_path = os.path.join(data_path, "scores")
os.makedirs(scores_path, exist_ok=True)

reactions_file = os.path.join(item_path, "reaction.txt")
proteins_file = os.path.join(item_path, "proteins.txt")
molecules_file = os.path.join(item_path, "molecules.txt")

fuse_path = os.path.join(model_path, "fuse")
os.makedirs(fuse_path, exist_ok=True)
