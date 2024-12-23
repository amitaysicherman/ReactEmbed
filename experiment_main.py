"""
Run the experiment end-to-end

1) preprocess the data
2) train contrastive learning model
3) evaluate the model on downstream task

If the data is already preprocessed, or the model is already trained, automatically skip the step
"""
import argparse
import os

import pandas as pd

from common.path_manager import data_path
from common.utils import model_args_to_name
from contrastive_learning.trainer import main as train_model
from eval_tasks.prep_tasks_seqs import main as task_seqs_main
from eval_tasks.prep_tasks_vecs import main as tasks_vecs_main
from eval_tasks.trainer import main as train_task
from preprocessing.biopax_parser import main as preprocess_data
from preprocessing.seq_to_vec import main as preprocess_sequences

parser = argparse.ArgumentParser()

parser.add_argument('--p_model', type=str, help='Protein model', default="ProtBert")
parser.add_argument('--m_model', type=str, help='Molecule model', default="ChemBERTa")
parser.add_argument('--data_name', type=str, help='Data name', default="reactome")
parser.add_argument('--cl_batch_size', type=int, help='Batch size', default=8192)
parser.add_argument('--cl_n_layers', type=int, help='Number of layers', default=1)
parser.add_argument('--cl_hidden_dim', type=int, help='Hidden dimension', default=64)
parser.add_argument('--cl_dropout', type=float, help='Dropout', default=0.3)
parser.add_argument('--cl_epochs', type=int, help='Number of epochs', default=5)
parser.add_argument('--cl_lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--cl_flip_prob', type=float, help='Flip Prob', default=0.0)

parser.add_argument("--task_name", type=str, default="BBBP")
parser.add_argument("--task_bs", type=int, default=8)
parser.add_argument("--task_lr", type=float, default=0.0001)
parser.add_argument("--task_drop_out", type=float, default=0.0)
parser.add_argument("--task_hidden_dim", type=int, default=-1)
parser.add_argument("--task_max_no_improve", type=int, default=25)
parser.add_argument("--task_n_layers", type=int, default=2)
parser.add_argument("--task_metric", type=str, default="auc")

args = parser.parse_args()
item_path = f"data/{args.data_name}"
os.makedirs(item_path, exist_ok=True)
reactions_file = f"{item_path}/reaction.txt"
proteins_file = f"{item_path}/proteins.txt"
molecules_file = f"{item_path}/molecules.txt"
if not os.path.exists(reactions_file):
    print(f"Start preprocess data")
    preprocess_data()
else:
    print("Skip preprocess data")

proteins_file = f'{item_path}/{args.p_model}_vectors.npy'
molecules_file = f'{item_path}/{args.m_model}_vectors.npy'
if not os.path.exists(proteins_file) or not os.path.exists(molecules_file):
    print(f"Start preprocess sequences")
    preprocess_sequences(args.p_model, args.data_name)
    preprocess_sequences(args.m_model, args.data_name)
else:
    print("Skip preprocess sequences")

print("Start train task")
p_model = args.p_model
m_model = args.m_model
n_layers = args.cl_n_layers
hidden_dim = args.cl_hidden_dim
dropout = args.cl_dropout
epochs = args.cl_epochs
lr = args.cl_lr
flip_prob = args.cl_flip_prob
batch_size = args.cl_batch_size
fuse_base = model_args_to_name(batch_size=batch_size, p_model=p_model, m_model=m_model, n_layers=n_layers,
                               hidden_dim=hidden_dim, dropout=dropout, epochs=epochs, lr=lr, flip_prob=flip_prob)
fuse_base = f"data/{args.data_name}/model/{fuse_base}"
cl_model_file = f"{fuse_base}/model.pt"
if not os.path.exists(cl_model_file):
    print("Start train contrastive learning model")
    train_model(args.cl_batch_size, args.p_model, args.m_model, args.cl_n_layers, args.cl_hidden_dim, args.cl_dropout,
                args.cl_epochs, args.cl_lr, args.cl_flip_prob)
else:
    print("Skip train contrastive learning model")

task_seq_file = f"{data_path}/torchdrug/{args.task_name}/train_labels.txt"
if not os.path.exists(task_seq_file):
    print("Start prep task sequences")
    task_seqs_main(args.task_name)
else:
    print("Skip prep task sequences")

task_prep_file = f"{data_path}/torchdrug/{args.task_name}/{args.p_model}_{args.m_model}.npz"
if not os.path.exists(task_prep_file):
    print("Start prep task vecs")
    tasks_vecs_main(args.task_name, args.p_model, args.m_model)
else:
    print("Skip prep task vecs")

print("Experiment finished")
print(f"Protein model: {args.p_model}, Molecule model: {args.m_model}")
print(f"Task name: {args.task_name}, metric: {args.task_metric}")
# write the results to file:

results_file = "results.csv"

for f in [0, 1]:

    results = train_task(f, 1 - f, args.task_bs, args.task_lr, args.task_drop_out,
                         args.task_hidden_dim, args.task_name, fuse_base, args.m_model, args.p_model,
                         args.task_n_layers, args.task_metric, args.task_max_no_improve)
    print(f"Fuse: {f}, Model: {1}, Score: {results}")
    args_dict = args.__dict__
    args_dict["use_fuse"] = f
    res_cols = sorted(list(args_dict.keys()))
    res_values = [args_dict[x] for x in res_cols] + [results]
    res_cols += ["score"]

    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            cols_str = ",".join(res_cols)
            vals_str = ",".join([str(x) for x in res_values])
            f.write(cols_str + "\n" + vals_str)

    else:
        old_res = pd.read_csv(results_file)
        old_cols = list(sorted(old_res.columns))
        if old_cols == res_cols:
            with open(results_file, "a") as f:
                f.write("\n" + ",".join([str(x) for x in res_values]))
        else:
            n = max(old_res.index) + 1
            for col in res_cols:
                if col == "score":
                    old_res.loc[n, col] = results
                    continue

                if col not in old_res.columns:
                    old_res[col] = ""
                old_res.loc[n, col] = args_dict[col]
            old_res.to_csv(results_file, index=None)
