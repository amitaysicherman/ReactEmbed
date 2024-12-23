import argparse
from itertools import product

from contrastive_learning.trainer import get_loader
from contrastive_learning.trainer import main as train_model

parser = argparse.ArgumentParser()
parser.add_argument("--m_model", type=str, default="ChemBERTa")
parser.add_argument("--p_model", type=str, default="ProtBert")
parser.add_argument("--batch_size", type=int, default=8192)
parser.add_argument("--data_name", type=str, default="reactome")
args = parser.parse_args()

n_layers_list = [1, 3]
hidden_dim_list = [512, 64]
dropout_list = [0.3, 0.0]
epochs_list = [1, 10]
lr_list = [1e-1, 1e-3]
flip_prob = 0
batch_size = args.batch_size
m_model = args.m_model
p_model = args.p_model
product_list = list(product(n_layers_list, hidden_dim_list, dropout_list, epochs_list, lr_list))
print(
    f"Start train {m_model} and {p_model} with batch size {batch_size} - {len(product_list)} models")

train_loader = get_loader(args.data_name, "train", batch_size, p_model, m_model, flip_prob)
valid_loader = get_loader(args.data_name, "valid", batch_size, p_model, m_model, flip_prob)
test_loader = get_loader(args.data_name, "test", batch_size, p_model, m_model, flip_prob)
datasets = (train_loader, valid_loader, test_loader)
for (n_layers, hidden_dim, dropout, epochs, lr) in product_list:
    train_model(batch_size, p_model, m_model, n_layers, hidden_dim, dropout, epochs, lr,
                flip_prob, datasets=datasets)
