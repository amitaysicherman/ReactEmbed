from itertools import product

from contrastive_learning.trainer import get_loader
from contrastive_learning.trainer import main as train_model

m_model_list = ["ChemBERTa", "MoLFormer"]
p_model_list = ["ProtBert", "esm3-small", "esm3-medium"]
batch_size_list = [8192, 64]
output_dim_list = [1024, 128]
n_layers_list = [1, 3]
hidden_dim_list = [512, 64]
dropout_list = [0.3, 0.0]
epochs_list = [1, 10]
lr_list = [1e-1, 1e-3]
flip_prob = 0

for m_model in m_model_list:
    for p_model in p_model_list:
        for batch_size in batch_size_list:
            product_list = product(output_dim_list, n_layers_list, hidden_dim_list, dropout_list, epochs_list, lr_list)
            print(
                f"Start train {m_model} and {p_model} with batch size {batch_size} - {len(list(product_list))} models")

            train_loader = get_loader("train", batch_size, p_model, m_model, flip_prob)
            valid_loader = get_loader("valid", batch_size, p_model, m_model, flip_prob)
            test_loader = get_loader("test", batch_size, p_model, m_model, flip_prob)
            datasets = (train_loader, valid_loader, test_loader)
            for (output_dim, n_layers, hidden_dim, dropout, epochs, lr) in product_list:
                train_model(batch_size, p_model, m_model, output_dim, n_layers, hidden_dim, dropout, epochs, lr,
                            flip_prob, datasets=datasets)
