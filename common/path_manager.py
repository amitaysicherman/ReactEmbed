import os
import pathlib

BASE_DIR = str(pathlib.Path(__file__).parent.parent.resolve())

data_path = os.path.join(BASE_DIR, "data")
item_path = os.path.join(data_path, "reactome")
model_path = os.path.join(data_path, "models")
scores_path = os.path.join(data_path, "scores")
reactions_file = os.path.join(item_path, "reaction.txt")
fuse_path = os.path.join(model_path, "fuse")
