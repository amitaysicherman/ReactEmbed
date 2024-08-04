import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from common.data_types import PRETRAINED_EMD, PROTEIN, MOLECULE
from common.path_manager import fuse_path
from common.utils import get_type_to_vec_dim
from contrastive_learning.dataset import PairsDataset, SameNameBatchSampler, get_reactions
from contrastive_learning.index_manger import NodesIndexManager
from contrastive_learning.model import MultiModalLinearConfig, MiltyModalLinear

EMBEDDING_DATA_TYPES = [PROTEIN, MOLECULE]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def indexes_to_tensor(indexes, node_index_manager: NodesIndexManager, return_type=True):
    type_ = node_index_manager.index_to_node[indexes[0].item()].type
    array = np.stack([node_index_manager.index_to_node[i.item()].vec for i in indexes])
    if return_type:
        return torch.tensor(array), type_
    return torch.tensor(array)


def save_fuse_model(model: MiltyModalLinear, save_dir, epoch):
    cp_to_remove = []
    for file_name in os.listdir(save_dir):
        if file_name.endswith(".pt"):
            cp_to_remove.append(f"{save_dir}/{file_name}")

    output_file = f"{save_dir}/fuse_{epoch}.pt"
    if output_file in cp_to_remove:
        cp_to_remove.remove(output_file)
    torch.save(model.state_dict(), output_file)
    for cp in cp_to_remove:
        os.remove(cp)


def run_epoch(model, node_index_manager, optimizer, loader, contrastive_loss, part="train", all_to_protein=True,
              triples=False):
    if len(loader) == 0:
        return 0
    is_train = part == "train"
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    for i, (data) in enumerate(loader):
        if triples:
            idx1, idx2, idx3 = data
            data_1, type_1 = indexes_to_tensor(idx1, node_index_manager)
            data_2, type_2 = indexes_to_tensor(idx2, node_index_manager)
            data_3, type_3 = indexes_to_tensor(idx3, node_index_manager)
            data_1 = data_1.to(device).float()
            data_2 = data_2.to(device).float()
            data_3 = data_3.to(device).float()
            assert type_2 == type_3
            if all_to_protein:
                if not model.have_type((type_2, type_1)):
                    continue
                out1 = data_1
                model_type = (type_2, type_1)
                out2 = model(data_2, model_type)
                out3 = model(data_3, model_type)
            else:
                out1 = model(data_1, type_1)
                out2 = model(data_2, type_2)
                out3 = model(data_3, type_3)
            step_labels = [1] * len(out1) + [0] * len(out1)
            pos_preds = (0.5 * (1 + F.cosine_similarity(out1, out2).cpu().detach().numpy())).tolist()
            neg_preds = (0.5 * (1 + F.cosine_similarity(out1, out3).cpu().detach().numpy())).tolist()
            step_preds = pos_preds + neg_preds
            cont_loss = contrastive_loss(out1, out2, out3)
        else:
            idx1, idx2, label = data
            data_1, type_1 = indexes_to_tensor(idx1, node_index_manager)
            data_2, type_2 = indexes_to_tensor(idx2, node_index_manager)
            data_1 = data_1.to(device).float()
            data_2 = data_2.to(device).float()
            if all_to_protein:
                if not model.have_type((type_1, type_2)):
                    continue
                out2 = data_2
                model_type = (type_1, type_2)
                out1 = model(data_1, model_type)
            else:
                out1 = model(data_1, type_1)
                out2 = model(data_2, type_2)
            step_labels = (label == 1).cpu().detach().numpy().astype(int).tolist()
            step_preds = (0.5 * (1 + F.cosine_similarity(out1, out2).cpu().detach().numpy())).tolist()
            cont_loss = contrastive_loss(out1, out2, label.to(device))

        all_labels.extend(step_labels)
        all_preds.extend(step_preds)
        total_loss += cont_loss.mean().item()
        if not is_train:
            continue
        cont_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if not is_train:
        return roc_auc_score(all_labels, all_preds)
    else:
        return 0


def get_loader(reactions, node_index_manager, batch_size, triples):
    dataset = PairsDataset(reactions, node_index_manager, triples=triples)
    sampler = SameNameBatchSampler(dataset, batch_size)
    return DataLoader(dataset, batch_sampler=sampler)


def build_models(type_to_vec_dim, all_to_protein, fuse_output_dim, fuse_n_layers, fuse_hidden_dim, fuse_dropout,
                 save_dir):
    if not all_to_protein:
        names = EMBEDDING_DATA_TYPES
        src_dims = [type_to_vec_dim[x] for x in EMBEDDING_DATA_TYPES]
        dst_dim = [fuse_output_dim] * len(EMBEDDING_DATA_TYPES)
    else:
        names = []
        src_dims = []
        dst_dim = []
        for src in EMBEDDING_DATA_TYPES:
            for dst in EMBEDDING_DATA_TYPES:
                if dst == PROTEIN:
                    src_dims.append(type_to_vec_dim[src])
                    names.append((src, dst))
                    dst_dim.append(type_to_vec_dim[dst])

    model_config = MultiModalLinearConfig(embedding_dim=src_dims, n_layers=fuse_n_layers, names=names,
                                          hidden_dim=fuse_hidden_dim, output_dim=dst_dim, dropout=fuse_dropout,
                                          normalize_last=1)

    model = MiltyModalLinear(model_config).to(device)
    model_config.save_to_file(f"{save_dir}/config.txt")
    return model


def main(args):
    save_dir = f"{fuse_path}/{args.fusion_name}"
    os.makedirs(save_dir, exist_ok=True)
    node_index_manager = NodesIndexManager(PRETRAINED_EMD, prot_emd_type=args.protein_embedding,
                                           mol_emd_type=args.molecule_embedding)
    train_reactions, validation_reactions, test_reaction = get_reactions()
    if args.fusion_train_all:
        train_reactions = train_reactions + validation_reactions + test_reaction
        validation_reactions = []
        test_reaction = []
    train_loader = get_loader(train_reactions, node_index_manager, args.fusion_batch_size, args.use_triplet_loss)
    valid_loader = get_loader(validation_reactions, node_index_manager, args.fusion_batch_size, args.use_triplet_loss)
    test_loader = get_loader(test_reaction, node_index_manager, args.fusion_batch_size, args.use_triplet_loss)
    type_to_vec_dim = get_type_to_vec_dim(args.protein_embedding)

    model = build_models(type_to_vec_dim, args.fusion_all_to_protein, args.fusion_output_dim, args.fusion_num_layers,
                         args.fusion_hidden_dim, args.fusion_dropout, save_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.fusion_learning_rate)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), "parameters")

    if args.use_triplet_loss:
        contrastive_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x1, x2: 1 - F.cosine_similarity(x1, x2))
    else:
        contrastive_loss = nn.CosineEmbeddingLoss(margin=0.0)

    best_valid_auc = -1e6
    best_test_auc = -1e6
    running_args = {"model": model, "node_index_manager": node_index_manager, "optimizer": optimizer,
                    "contrastive_loss": contrastive_loss, "all_to_protein": args.fusion_all_to_protein,
                    'triples': args.use_triplet_loss}
    no_improve_count = 0
    for epoch in range(args.fusion_epochs):
        _ = run_epoch(**running_args, loader=train_loader, part="train")
        with torch.no_grad():
            valid_auc = run_epoch(**running_args, loader=valid_loader, part="valid")
            test_auc = run_epoch(**running_args, loader=test_loader, part="test")
        if args.fusion_train_all or epoch == 0:
            save_fuse_model(model, save_dir, epoch)
            continue
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_test_auc = test_auc
            save_fuse_model(model, save_dir, epoch)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= args.fusion_max_epochs_no_improve:
                break
        print(f"Epoch: {epoch}, Valid AUC: {best_valid_auc}, Test AUC: {best_test_auc}")
    return best_valid_auc


if __name__ == '__main__':
    from common.args_manager import get_args

    args = get_args()
    main(args)
