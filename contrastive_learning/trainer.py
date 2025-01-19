import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.utils import model_args_to_name
from contrastive_learning.dataset import TripletsDataset, TripletsBatchSampler
from contrastive_learning.model import ReactEmbedConfig, ReactEmbedModel
from contrastive_learning.visualization import visualize_training

model_to_dim = {
    "ChemBERTa": 768,
    "ProtBert": 1024,
    "esm3-medium": 1152,
    "esm3-small": 960,
    "esm3-large": 2560,
    "MoLFormer": 768,
    "MolCLR": 512,
    "GearNet": 3072,
    "esm3-6b": 2560,
}

try:
    if torch.mps.device_count() > 0:
        device = "mps"
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
except:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


def run_epoch(model, optimizer, loader, contrastive_loss, is_train):
    if loader is None:
        return
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    count = 0
    for i, (t1, t2, t3, data_1, data_2, data_3) in tqdm(enumerate(loader), total=len(loader)):
        data_1, data_2, data_3 = data_1.to(device), data_2.to(device), data_3.to(device)

        shared_1 = model(data_1, t1[0])  # anchor
        shared_2 = model(data_2, t2[0])  # positive
        shared_3 = model(data_3, t3[0])  # negative
        cont_loss = contrastive_loss(shared_1, shared_2, shared_3)
        total_loss += cont_loss.mean().item()
        count += 1
        if not is_train:
            continue
        cont_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / count


def get_loader(data_name, split, batch_size, p_model, m_model, flip_prob, samples_ratio, no_pp_mm):
    dataset = TripletsDataset(data_name=data_name, p_model=p_model, m_model=m_model, split=split,
                              flip_prob=flip_prob, samples_ratio=samples_ratio, no_pp_mm=no_pp_mm)
    sampler = TripletsBatchSampler(dataset, batch_size)
    return DataLoader(dataset, batch_sampler=sampler)


def build_models(p_dim, m_dim, shared_dim, n_layers, hidden_dim, dropout, save_dir):
    model_config = ReactEmbedConfig(p_dim, m_dim, shared_dim, n_layers, hidden_dim, dropout)
    model = ReactEmbedModel(model_config).to(device)
    model_config.save_to_file(f"{save_dir}/config.txt")
    return model


def main(data_name, batch_size, p_model, m_model, shared_dim, n_layers, hidden_dim, dropout,
         epochs, lr, flip_prob=0, samples_ratio=1, no_pp_mm=0, datasets=None, override=False):
    name = model_args_to_name(p_model, m_model, n_layers, hidden_dim, dropout, epochs, lr, batch_size, flip_prob,
                              shared_dim, samples_ratio, no_pp_mm)
    save_dir = f"data/{data_name}/model/{name}/"
    model_file = f"{save_dir}/model.pt"
    if not override and os.path.exists(model_file):
        print("Model already exists")
        return

    os.makedirs(save_dir, exist_ok=True)
    if datasets is not None:
        train_loader, valid_loader, test_loader = datasets
    else:
        train_loader = get_loader(data_name, "train", batch_size, p_model, m_model, flip_prob, samples_ratio, no_pp_mm)
        valid_loader = get_loader(data_name, "valid", batch_size, p_model, m_model, flip_prob, samples_ratio, no_pp_mm)
        test_loader = get_loader(data_name, "test", batch_size, p_model, m_model, flip_prob, samples_ratio, no_pp_mm)

    p_dim = model_to_dim[p_model]
    m_dim = model_to_dim[m_model]

    model = build_models(p_dim, m_dim, shared_dim, n_layers, hidden_dim, dropout, save_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

    contrastive_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x1, x2: 1 - F.cosine_similarity(x1, x2))

    best_valid_loss = float("inf")
    for epoch in range(epochs):
        visualize_training(model, train_loader.dataset, device, epoch, save_dir)

        train_loss = run_epoch(model, optimizer, train_loader, contrastive_loss, is_train=True)
        with torch.no_grad():
            valid_loss = run_epoch(model, optimizer, valid_loader, contrastive_loss, is_train=False)
            test_loss = run_epoch(model, optimizer, test_loader, contrastive_loss, is_train=False)
        print(f"Epoch {epoch} Train Loss: {train_loss}, Valid Loss: {valid_loss}, Test Loss: {test_loss}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{save_dir}/model.pt")
            print("Model saved")

        with open(f"{save_dir}/losses.txt", "a") as f:
            f.write(f"Epoch {epoch}: Train Loss: {train_loss}, Valid Loss: {valid_loss}, Test Loss: {test_loss}\n")

    with open(f"{save_dir}/losses.txt", "a") as f:
        f.write(f"Best Valid Loss: {best_valid_loss}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Contrastive Learning')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=256)
    parser.add_argument('--p_model', type=str, help='Protein model', default="ProtBert")
    parser.add_argument('--m_model', type=str, help='Molecule model', default="ChemBERTa")
    parser.add_argument('--shared_dim', type=int, help='Shared space dimension', default=256)
    parser.add_argument('--n_layers', type=int, help='Number of layers', default=1)
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension', default=512)
    parser.add_argument('--dropout', type=float, help='Dropout', default=0.0)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=10)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--flip_prob', type=float, help='Flip Prob', default=0.0)
    parser.add_argument("--samples_ratio", type=float, default=1)
    parser.add_argument("--no_pp_mm", type=int, default=0)
    parser.add_argument('--data_name', type=str, help='Data name', default="reactome")
    parser.add_argument('--override', action='store_true', help='Override existing model')
    args = parser.parse_args()

    main(data_name=args.data_name, batch_size=args.batch_size, p_model=args.p_model, m_model=args.m_model,
         shared_dim=args.shared_dim, n_layers=args.n_layers, hidden_dim=args.hidden_dim, dropout=args.dropout,
         epochs=args.epochs, lr=args.lr, flip_prob=args.flip_prob, samples_ratio=args.samples_ratio,
         no_pp_mm=args.no_pp_mm, override=args.override)
