import os

import numpy as np
import torch

from eval_tasks.models import load_fuse_model
from preprocessing.seq_to_vec import SeqToVec

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DPPC = 'CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC'
cholesterol = 'CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C'
with open("transferrin/can_seqs.txt") as f:
    lines = f.read().splitlines()
p_names = [line.split(",")[0] for line in lines]
t_index = p_names.index("TfR")


def get_sklearn_classifier(name, **kwargs):
    if name == "KNeighbors":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**kwargs)
    elif name == "SVC":
        from sklearn.svm import SVC
        return SVC(**kwargs)
    elif name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**kwargs)
    elif name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**kwargs)
    elif name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**kwargs)
    elif name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(**kwargs)
    elif name == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def get_classifiers_iter():
    for name in ["KNeighbors", "SVC", "RandomForest", "LogisticRegression", "GradientBoosting", "MLP", "AdaBoost"]:
        if name == "KNeighbors":
            yield get_sklearn_classifier(name, n_neighbors=1), "KNeighbors-1"
            yield get_sklearn_classifier(name, n_neighbors=2), "KNeighbors-2"
            yield get_sklearn_classifier(name, n_neighbors=3), "KNeighbors-4"
            yield get_sklearn_classifier(name, n_neighbors=4), "KNeighbors-4"
            yield get_sklearn_classifier(name, n_neighbors=5), "KNeighbors-5"
            yield get_sklearn_classifier(name, n_neighbors=10), "KNeighbors-10"
            yield get_sklearn_classifier(name, n_neighbors=20), "KNeighbors-20"
            yield get_sklearn_classifier(name, n_neighbors=50), "KNeighbors-50"
        elif name == "SVC":
            yield get_sklearn_classifier(name, probability=True, kernel="linear"), "SVC-linear"
            yield get_sklearn_classifier(name, probability=True, kernel="poly"), "SVC-poly"
            yield get_sklearn_classifier(name, probability=True, kernel="rbf"), "SVC-rbf"
            yield get_sklearn_classifier(name, probability=True, kernel="sigmoid"), "SVC-sigmoid"
        elif name == "RandomForest":
            yield get_sklearn_classifier(name, n_estimators=10, max_depth=2, random_state=0), "RandomForest-10-2"
            yield get_sklearn_classifier(name, n_estimators=100, max_depth=1, random_state=0), "RandomForest-100-1"
            yield get_sklearn_classifier(name, n_estimators=50, max_depth=2, random_state=0), "RandomForest-50-2"
        elif name == "LogisticRegression":
            yield get_sklearn_classifier(name, random_state=0, max_iter=1000), "LogisticRegression"
            yield get_sklearn_classifier(name, C=0.1, random_state=0, max_iter=1000), "LogisticRegression-0.1"
        elif name == "GradientBoosting":
            yield get_sklearn_classifier(name, n_estimators=10, learning_rate=0.1, max_depth=2,
                                         random_state=0), "GradientBoosting-10-0.1-2"
            yield get_sklearn_classifier(name, n_estimators=100, learning_rate=0.1, max_depth=1,
                                         random_state=0), "GradientBoosting-100-0.1-1"
        elif name == "MLP":
            yield get_sklearn_classifier(name, hidden_layer_sizes=(10,), max_iter=1000), "MLP-10"
            yield get_sklearn_classifier(name, hidden_layer_sizes=(50,), max_iter=1000), "MLP-50"
            yield get_sklearn_classifier(name, hidden_layer_sizes=(100,), max_iter=1000), "MLP-100"
            yield get_sklearn_classifier(name, hidden_layer_sizes=(100, 50), max_iter=1000), "MLP-100-50"
            yield get_sklearn_classifier(name, hidden_layer_sizes=(100, 50, 10), max_iter=1000), "MLP-100-50-10"
        elif name == "AdaBoost":
            yield get_sklearn_classifier(name, n_estimators=100, random_state=0), "AdaBoost-100"
            yield get_sklearn_classifier(name, n_estimators=50, random_state=0), "AdaBoost-50"
            yield get_sklearn_classifier(name, n_estimators=10, random_state=0), "AdaBoost-10"
        else:
            raise ValueError(f"Unknown classifier: {name}")


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
    vecs = np.load(f"transferrin/can_{p_model}.npy")
    protein_names = p_names
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
    for model, model_name in get_classifiers_iter():
        print(f"{m_model} + {p_model} + {model_name}")
        model.fit(x, y)
        mol_score = model.predict_proba(molecules.detach().cpu().numpy().reshape(1, -1))[:, 1].flatten()[0]
        proteins_scores = model.predict_proba(complex.detach().cpu().numpy())[:, 1].flatten().tolist()
        t_v = proteins_scores[t_index]
        print(mol_score, t_v, max(proteins_scores), sorted(proteins_scores, reverse=True).index(t_v))


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
    main(args.p_model, args.m_model, args.fusion_name)
