"""
Run the experiment end-to-end

1) preprocess the data
2) train contrastive learning model
3) evaluate the model on downstream task

If the data is already preprocessed, or the model is already trained, automatically skip the step
"""
import argparse
import os

from common.path_manager import data_path, reactions_file, item_path, fuse_path
from contrastive_learning.trainer import main as train_model
from eval_tasks.prep_tasks import main as prep_tasks
from eval_tasks.trainer import main as train_task
from preprocessing.biopax_parser import main as preprocess_data
from preprocessing.seq_to_vec import main as preprocess_sequences

parser = argparse.ArgumentParser()

parser.add_argument('--p_model', type=str, help='Protein model', default="ProtBert")
parser.add_argument('--m_model', type=str, help='Molecule model', default="ChemBERTa")

parser.add_argument('--cl_batch_size', type=int, help='Batch size', default=8192)
parser.add_argument('--cl_output_dim', type=int, help='Output dimension', default=1024)
parser.add_argument('--cl_n_layers', type=int, help='Number of layers', default=2)
parser.add_argument('--cl_hidden_dim', type=int, help='Hidden dimension', default=64)
parser.add_argument('--cl_dropout', type=float, help='Dropout', default=0.3)
parser.add_argument('--cl_epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--cl_lr', type=float, help='Learning rate', default=0.001)

parser.add_argument("--task_name", type=str, default="BACE")
parser.add_argument("--task_use_fuse", type=int, default=0)
parser.add_argument("--task_use_model", type=int, default=1)
parser.add_argument("--task_bs", type=int, default=16)
parser.add_argument("--task_lr", type=float, default=0.001)
parser.add_argument("--task_drop_out", type=float, default=0.0)
parser.add_argument("--task_hidden_dim", type=int, default=64)
parser.add_argument("--task_max_no_improve", type=int, default=15)
parser.add_argument("--task_n_layers", type=int, default=1)
parser.add_argument("--task_metric", type=str, default="f1_max")

args = parser.parse_args()

if not os.path.exists(reactions_file):
    print(f"Start preprocess data")
    preprocess_data()
else:
    print("Skip preprocess data")

proteins_file = f'{item_path}/{args.p_model}_vectors.npy'
molecules_file = f'{item_path}/{args.m_model}_vectors.npy'
if not os.path.exists(proteins_file) or not os.path.exists(molecules_file):
    print(f"Start preprocess sequences")
    preprocess_sequences(args.p_model)
    preprocess_sequences(args.m_model)
else:
    print("Skip preprocess sequences")

cl_model_file = f"{fuse_path}/{args.p_model}-{args.m_model}/model.pt"
if not os.path.exists(cl_model_file):
    print("Start train contrastive learning model")
    train_model(args.cl_batch_size, args.p_model, args.m_model, args.cl_output_dim, args.cl_n_layers,
                args.cl_hidden_dim,
                args.cl_dropout, args.cl_epochs, args.cl_lr)
else:
    print("Skip train contrastive learning model")

task_prep_file = f"{data_path}/torchdrug/{args.task_name}_{args.p_model}_{args.m_model}.npz"
if not os.path.exists(task_prep_file):
    print("Start prep task")
    prep_tasks(args.task_name, args.p_model, args.m_model)
else:
    print("Skip prep task")

print("Start train task")
results = train_task(args.task_use_fuse, args.task_use_model, args.task_bs, args.task_lr, args.task_drop_out,
                     args.task_hidden_dim, args.task_name, f"{args.p_model}-{args.m_model}", args.m_model, args.p_model,
                     args.task_n_layers, args.task_metric, args.task_max_no_improve)

print("Experiment finished")
print(f"Protein model: {args.p_model}, Molecule model: {args.m_model}")
print(f"Task name: {args.task_name}, metric: {args.task_metric}")
print(f"Results: {results}")
