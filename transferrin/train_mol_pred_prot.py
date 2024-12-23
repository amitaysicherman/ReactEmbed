import numpy as np

from contrastive_learning.model import ReactEmbedModel
from eval_tasks.models import LinFuseModel
from eval_tasks.trainer import main as trainer_task_main

p_model = "ProtBert"
m_model = "MoLFormer"
fuse_base = "data/pathbank/model/ProtBert-MoLFormer-2-256-0.3-10-0.001-8192-0.0"
model: LinFuseModel = None
score, model = trainer_task_main(use_fuse=True, use_model=False, bs=32, lr=1e-4, drop_out=0.1, hidden_dim=128,
                                 task_name="BBBP", fuse_base=fuse_base, mol_emd=m_model, protein_emd=p_model,
                                 n_layers=2, metric="auc", max_no_improve=15, return_model=True)
print(score)
# load reactome proteins data
proteins = np.load("data/reactome/ProtBert_vectors.npy")
fuse_model: ReactEmbedModel = model.fuse_model
fuse_model.eval()
for p in proteins:
    x = fuse_model.dual_forward(p, "P")
    pred = model.layers(x)
    print(pred)
