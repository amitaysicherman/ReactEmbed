import os

import numpy as np
import torch

from eval_tasks.models import load_fuse_model
from preprocessing.seq_to_vec import SeqToVec
from transferrin.utils import PreprocessManager

transferrin_id = "P02787"
insulin_id = "P01308"
Leptin_id = "P41159"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DPPC = 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC'
cholesterol = 'CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C'


def get_task_data(p_model, m_model):
    from eval_tasks.dataset import load_data
    task_name = "BBBP"
    x1_train, x2_train, labels_train, x1_valid, x2_valid, labels_valid, x1_test, x2_test, labels_test = load_data(
        task_name, m_model, p_model)
    return x1_train, x2_train, labels_train, x1_valid, x2_valid, labels_valid, x1_test, x2_test, labels_test


def train_ml_model(p_model, m_model, fuse_model):
    from sklearn.svm import SVC

    x1_train, x2_train, labels_train, x1_valid, x2_valid, labels_valid, x1_test, x2_test, labels_test = get_task_data(
        p_model, m_model)
    x = np.concatenate([x1_train, x1_valid, x1_test])
    x = torch.tensor(x).to(device).float()
    x = fuse_model(x, "M").detach().cpu().numpy()
    y = np.concatenate([labels_train, labels_valid, labels_test])
    # model = KNeighborsClassifier(n_neighbors=7)
    model = SVC(probability=True)
    # model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0)
    # model = LogisticRegression(random_state=0)
    model.fit(x, y)
    print(f"Model score: {model.score(x, y)}")
    # predict probabilities
    return model


def main(p_model="esm3-medium", m_model="ChemBERTa",
         fuse_base="data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.0/", metric="f1_max",
         n_layers=2, hid_dim=512, drop_out=0.3, print_full_res=False, save_models=False):
    preprocess = PreprocessManager(p_model=p_model, reactome=True)
    fuse_model, dim = load_fuse_model(fuse_base)
    fuse_model.eval().to(device)
    print(f"fuse_model: {fuse_model}")
    print(f"dim: {dim}")
    model = train_ml_model(p_model, m_model, fuse_model)

    vecs = preprocess.get_vecs()
    protein_names = preprocess.get_proteins()
    proteins = torch.tensor(vecs).to(device).float()
    fuse_model.eval()
    proteins_fuse = fuse_model(proteins, "P")
    mol_file = f"transferrin/mols_{m_model}.npy"
    if os.path.exists(mol_file):
        molecules = torch.tensor(np.load(mol_file)).to(device).float()
    else:
        seq_to_vec = SeqToVec(model_name=m_model)
        dppc_vec = torch.tensor(seq_to_vec.to_vec(DPPC)).to(device).float()
        dppc_fuse = fuse_model(dppc_vec, "M")
        cholesterol_vec = torch.tensor(seq_to_vec.to_vec(cholesterol)).to(device).float()
        cholesterol_fuse = fuse_model(cholesterol_vec, "M")
        molecules = 0.67 * dppc_fuse + 0.33 * cholesterol_fuse
        np.save(mol_file, molecules.detach().cpu().numpy())
    complex = 0.5 * proteins_fuse + 0.5 * molecules
    # complex_fuse = proteins_fuse
    # model.eval()
    # pred = model.layers(complex_fuse)
    # res = torch.sigmoid(pred).detach().cpu().numpy().flatten()
    # res = model.predict_proba(complex_fuse.detach().cpu().numpy())[:, 1]
    mol_pred = model.predict_proba(molecules.detach().cpu().numpy().reshape(1, -1))[:, 1]
    print(f"Molecule score: {mol_pred}")
    for name, id_ in [("transferrin", transferrin_id), ("insulin", insulin_id), ("Leptin", Leptin_id)]:
        index = protein_names.index(id_)
        complex_score = model.predict_proba(complex[index].detach().cpu().numpy().reshape(1, -1))[:, 1]
        print(f"{name} score: {complex_score}")
    # go_matrix = preprocess.get_go_matrix()
    # assert transferrin_id in go_matrix.index
    # assert len(go_matrix) == preprocess.get_vecs().shape[0]
    # go_matrix["S"] = res.flatten()
    # transferrin_index = go_matrix.index.get_loc(transferrin_id)
    # transferrin_score = go_matrix.iloc[transferrin_index]["S"]
    # print(f"Transferrin score: {transferrin_score}")
    # print(f"Higher score count: {(go_matrix['S'] > transferrin_score).sum()}")
    # print(f"Higher score count: {(go_matrix['S'] >= transferrin_score).sum()}")
    # higher_score_count = (go_matrix["S"] > transferrin_score).sum()
    # with open("transferrin/results.csv", "a") as f:
    #     txt = f"{p_model},{m_model},{fuse_base},{metric},{n_layers},{hid_dim},{drop_out},{transferrin_score},{higher_score_count}\n"
    #     print(txt)
    #     f.write(txt)
    #
    # if save_models:
    #     import os
    #     save_dir = f"transferrin/models"
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     file_name = f"{p_model}_{m_model}_{metric}_{n_layers}_{hid_dim}_{drop_out}"
    #     torch.save(model.state_dict(), f"{save_dir}/{file_name}.pt")
    #
    # if not print_full_res:
    #     return
    #
    # single_res = find_top_n_combinations(go_matrix, transferrin_index, n_results=10, max_cols=1, min_samples=100)
    # print("Transferrin results Single")
    # print(single_res)
    # double_res = find_top_n_combinations(go_matrix, transferrin_index, n_results=10, max_cols=2, min_samples=100)
    # print("Transferrin results Double")
    # print(double_res)
    #

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
