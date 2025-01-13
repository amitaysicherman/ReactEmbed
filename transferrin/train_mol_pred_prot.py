import os

import numpy as np
import torch
from sklearn.svm import SVC

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


def main(p_model="esm3-medium", m_model="ChemBERTa",
         fuse_base="data/reactome/model/esm3-medium-ChemBERTa-1-256-0.3-1-5e-05-256-0.0/"):
    fuse_model, dim = load_fuse_model(fuse_base)
    fuse_model.eval().to(device)
    preprocess = PreprocessManager(p_model=p_model, reactome=True)
    vecs = preprocess.get_vecs()
    protein_names = preprocess.get_proteins()
    proteins = torch.tensor(vecs).to(device).float()
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
    x1_train, x2_train, labels_train, x1_valid, x2_valid, labels_valid, x1_test, x2_test, labels_test = get_task_data(
        p_model, m_model)
    x = np.concatenate([x1_train, x1_valid, x1_test])
    x = torch.tensor(x).to(device).float()
    x = fuse_model(x, "M").detach().cpu().numpy()
    y = np.concatenate([labels_train, labels_valid, labels_test]).flatten()
    print("=== Results ===")
    print(f"{m_model} + {p_model}")
    print("=" * 70)
    print(f"{'Model':<25} {'Molecule':>10} {'Transferrin':>12} {'Insulin':>10} {'Leptin':>10}")
    print("-" * 70)  # Add separator line
    model = SVC(probability=True, kernel='linear')
    model_name = "SVM"
    model.fit(x, y)
    mol_pred = model.predict_proba(molecules.detach().cpu().numpy().reshape(1, -1))[:, 1]
    complex_scores = []
    for name, id_ in [("transferrin", transferrin_id), ("insulin", insulin_id), ("Leptin", Leptin_id)]:
        index = protein_names.index(id_)
        complex_score = model.predict_proba(complex[index].detach().cpu().numpy().reshape(1, -1))[:, 1]
        complex_scores.append(complex_score[0])
    print(
        f"{model_name:<25} {mol_pred[0]:>10.3f} {complex_scores[0]:>12.3f} {complex_scores[1]:>10.3f} {complex_scores[2]:>10.3f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--p_model", type=str, default="esm3-medium")
    parser.add_argument("--m_model", type=str, default="MoLFormer")
    parser.add_argument("--fusion_name", type=str,
                        default="data/reactome/model/esm3-medium-MoLFormer-1-512-0.0-10-0.0001-256-0.0-256")
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
