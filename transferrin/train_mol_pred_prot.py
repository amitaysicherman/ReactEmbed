import torch

from contrastive_learning.model import ReactEmbedModel
from eval_tasks.models import LinFuseModel
from eval_tasks.trainer import main as trainer_task_main
from transferrin.utils import PreprocessManager, get_go_description, find_top_n_combinations

transferrin_id = "P02787"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
p_model = "esm3-medium"
m_model = "MoLFormer"
preprocess = PreprocessManager(p_model=p_model, reactome=True)
fuse_base = f"data/reactome/model/{p_model}-{m_model}-2-256-0.3-10-0.001-8192-0.0"
model: LinFuseModel = None
score, model = trainer_task_main(use_fuse=True, use_model=False, bs=32, lr=1e-4, drop_out=0.1, hidden_dim=128,
                                 task_name="BBBP", fuse_base=fuse_base, mol_emd=m_model, protein_emd=p_model,
                                 n_layers=2, metric="auc", max_no_improve=15, return_model=True)
print("Model loaded")
print("score", score)
vecs = preprocess.get_vecs()
proteins = torch.tensor(vecs).to(device)
fuse_model: ReactEmbedModel = model.fuse_model
fuse_model.eval()
x = fuse_model.dual_forward(proteins, "P")
pred = model.layers(x)

res = torch.sigmoid(pred).detach().cpu().numpy().flatten()
go_matrix = preprocess.get_go_matrix()
assert transferrin_id in go_matrix.index
assert len(go_matrix) == preprocess.get_vecs().shape[0]
go_matrix["S"] = res.flatten()
transferrin_index = go_matrix.index.get_loc(transferrin_id)
print("Predictions done")

print(f"Processing {p_model} - {m_model}")
print("Transferrin:", res.flatten()[transferrin_index], sum(res > res[transferrin_index]))

results = find_top_n_combinations(go_matrix, transferrin_index, n_results=30, max_cols=2, min_samples=100)
for r in results:
    print(r)
    for go_id in r['columns']:
        print(get_go_description(go_id))
