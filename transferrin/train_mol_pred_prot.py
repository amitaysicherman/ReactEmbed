import torch

from contrastive_learning.model import ReactEmbedModel
from eval_tasks.trainer import main as trainer_task_main
from preprocessing.seq_to_vec import SeqToVec
from transferrin.utils import PreprocessManager, find_top_n_combinations

transferrin_id = "P02787"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DPPC = 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC'
cholesterol = 'CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C'


def main(p_model="esm3-medium", m_model="ChemBERTa",
         fuse_base="data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.0/", metric="f1_max",
         n_layers=2, hid_dim=512, drop_out=0.3, print_full_res=False, save_models=False):
    preprocess = PreprocessManager(p_model=p_model, reactome=True)
    score, model = trainer_task_main(use_fuse=True, use_model=False, bs=2048, lr=0.001, drop_out=drop_out,
                                     hidden_dim=hid_dim,
                                     task_name="BBBP", fuse_base=fuse_base, mol_emd=m_model, protein_emd=p_model,
                                     n_layers=n_layers, metric=metric, max_no_improve=15, return_model=True)
    vecs = preprocess.get_vecs()
    proteins = torch.tensor(vecs).to(device).float()

    fuse_model: ReactEmbedModel = model.fuse_model
    fuse_model.eval()
    proteins_fuse = fuse_model(proteins, "P")
    seq_to_vec = SeqToVec(model_name=m_model)
    dppc_vec = torch.tensor(seq_to_vec.to_vec(DPPC)).to(device).float()
    dppc_fuse = fuse_model(dppc_vec, "M")
    cholesterol_vec = torch.tensor(seq_to_vec.to_vec(cholesterol)).to(device).float()
    cholesterol_fuse = fuse_model(cholesterol_vec, "M")
    complex_fuse = 0.5 * proteins_fuse + 0.33 * dppc_fuse + 0.17 * cholesterol_fuse

    model.eval()
    pred = model.layers(complex_fuse)
    res = torch.sigmoid(pred).detach().cpu().numpy().flatten()
    go_matrix = preprocess.get_go_matrix()
    assert transferrin_id in go_matrix.index
    assert len(go_matrix) == preprocess.get_vecs().shape[0]
    go_matrix["S"] = res.flatten()
    transferrin_index = go_matrix.index.get_loc(transferrin_id)
    transferrin_score = go_matrix.iloc[transferrin_index]["S"]
    higher_score_count = (go_matrix["S"] > transferrin_score).sum()
    with open("transferrin/results.csv", "a") as f:
        txt = f"{p_model},{m_model},{fuse_base},{metric},{n_layers},{hid_dim},{drop_out},{transferrin_score},{higher_score_count}\n"
        print(txt)
        f.write(txt)

    if save_models:
        import os
        save_dir = f"transferrin/models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = f"{p_model}_{m_model}_{metric}_{n_layers}_{hid_dim}_{drop_out}"
        torch.save(model.state_dict(), f"{save_dir}/{file_name}.pt")

    if not print_full_res:
        return

    single_res = find_top_n_combinations(go_matrix, transferrin_index, n_results=10, max_cols=1, min_samples=100)
    print("Transferrin results Single")
    print(single_res)
    double_res = find_top_n_combinations(go_matrix, transferrin_index, n_results=10, max_cols=2, min_samples=100)
    print("Transferrin results Double")
    print(double_res)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--p_model", type=str, default="ProtBert")
    parser.add_argument("--m_model", type=str, default="ChemBERTa")
    parser.add_argument("--fusion_name", type=str,
                        default="data/reactome/model/ProtBert-ChemBERTa-1-512-0.0-10-0.0001-256-0.0-256")
    parser.add_argument("--metric", type=str, default="auc")
    parser.add_argument("--print_full_res", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--drop_out", type=float, default=0.0)
    args = parser.parse_args()
    torch.manual_seed(42)
    main(args.p_model, args.m_model, args.fusion_name, args.metric, args.n_layers, args.hid_dim, args.drop_out,
         args.print_full_res, args.save_models)
