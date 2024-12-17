from itertools import product

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
list_product = product(m_model_list, p_model_list, batch_size_list, output_dim_list, n_layers_list, hidden_dim_list,
                       dropout_list, epochs_list, lr_list)
for (m_model, p_model, batch_size, output_dim, n_layers, hidden_dim, dropout, epochs, lr) in list_product:
    train_model(batch_size, p_model, m_model, output_dim, n_layers, hidden_dim, dropout, epochs, lr, flip_prob)
