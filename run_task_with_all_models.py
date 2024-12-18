import argparse
import os

from common.utils import name_to_model_args
from eval_tasks.trainer import main as train_task

parser = argparse.ArgumentParser()
parser.add_argument("--use_fuse", type=int, default=1)
parser.add_argument("--use_model", type=int, default=1)
parser.add_argument("--bs", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--drop_out", type=float, default=0.3)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--task_name", type=str, default="BBBP")
parser.add_argument("--max_no_improve", type=int, default=15)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--metric", type=str, default="auc")
args = parser.parse_args()

model_names = os.listdir("data/models/fuse")
for name in model_names:
    model_args = name_to_model_args(name)
    try:
        res = train_task(use_fuse=args.use_fuse, use_model=args.use_model, bs=args.bs, lr=args.lr,
                         drop_out=args.drop_out,
                         hidden_dim=args.hidden_dim, task_name=args.task_name, fuse_base=name,
                         mol_emd=model_args['m_model'], protein_emd="ProtBert",
                         max_no_improve=args.max_no_improve,
                         n_layers=args.n_layers, metric=args.metric)
        print(f"Model: {name}, Result: {res}")
    except Exception as e:
        print(f"Model: {name}, Error: {e}")
        continue
