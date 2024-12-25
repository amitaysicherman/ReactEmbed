import torch

from contrastive_learning.model import ReactEmbedModel
from eval_tasks.trainer import main as trainer_task_main
from transferrin.utils import PreprocessManager, find_top_n_combinations

transferrin_id = "P02787"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser.add_argument("--use_fuse", type=int, default=1)
parser.add_argument("--use_model", type=int, default=1)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--drop_out", type=float, default=0.3)
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--task_name", type=str, default="BACE")
parser.add_argument("--fusion_name", type=str,
                    default="data/pathbank/model/ProtBert-MolCLR-2-64-0.3-10-0.001-8192-0.0/")
parser.add_argument("--m_model", type=str, default="ChemBERTa")
parser.add_argument("--p_model", type=str, default="ProtBert")
parser.add_argument("--max_no_improve", type=int, default=5)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--metric", type=str, default="f1_max")


def main(p_model, m_model, fuse_base, metric):
    preprocess = PreprocessManager(p_model=p_model, reactome=True)
    score, model = trainer_task_main(use_fuse=True, use_model=False, bs=16, lr=0.001, drop_out=0.3, hidden_dim=512,
                                     task_name="BBBP", fuse_base=fuse_base, mol_emd=m_model, protein_emd=p_model,
                                     n_layers=2, metric=metric, max_no_improve=5, return_model=True)
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
    results = find_top_n_combinations(go_matrix, transferrin_index, n_results=1, max_cols=2, min_samples=100)
    results = results[0][2]
    with open("transferrin/results.csv", "a") as f:
        f.write(
            f"{p_model},{m_model},{fuse_base},{metric},{results['filtered_size']},{results['new_rank']}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--p_model", type=str, default="ProtBert")
    parser.add_argument("--m_model", type=str, default="ChemBERTa")
    parser.add_argument("--fusion_name", type=str,
                        default="data/reactome/model/ProtBert-MolCLR-2-256-0.3-10-5e-05-256-0.0/")
    parser.add_argument("--metric", type=str, default="f1_max")
    args = parser.parse_args()
    torch.manual_seed(42)
    main(args.p_model, args.m_model, args.fusion_name, args.metric)
