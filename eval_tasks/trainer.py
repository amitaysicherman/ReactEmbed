import torch

from common.data_types import Config
from eval_tasks.dataset import get_dataloaders
from eval_tasks.models import LinFuseModel, PairsFuseModel
from eval_tasks.scores import Scores, ScoresManager
from eval_tasks.tasks import name_to_task, Task

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_epoch(model, loader, optimizer, criterion, metric_name, part):
    if part == "train":
        model.train()
    else:
        model.eval()
    reals = []
    preds = []
    for *all_x, labels in loader:
        if len(all_x) == 1:
            x = all_x[0]
            x = x.float().to(device)
            output = model(x)

        else:
            x1, x2 = all_x
            x1 = x1.float().to(device)
            x2 = x2.float().to(device)
            output = model(x1, x2)

        optimizer.zero_grad()
        labels = labels.float().to(device)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            labels = labels.squeeze(1).long()

        loss = criterion(output, labels)
        if part == "train":
            loss.backward()
            optimizer.step()
        reals.append(labels)
        preds.append(output)
    if part != "train":
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        return Scores(metric_name, preds, reals)
    else:
        return None


def get_model_from_task(task: Task, dataset, conf, fuse_base, drop_out, n_layers, hidden_dim, fuse_model=None):
    model_class = task.model
    input_dim_1 = dataset.x1.shape[1]
    dtype_1 = task.dtype1
    if task.dtype2 is not None:
        input_dim_2 = dataset.x2.shape[1]
        dtype_2 = task.dtype2
    else:
        input_dim_2 = None
        dtype_2 = None
    output_dim = task.output_dim
    common_args_dict = {
        "input_dim_1": input_dim_1,
        "dtype_1": dtype_1,
        "output_dim": output_dim,
        "conf": conf,
        "fuse_base": fuse_base,
        "drop_out": drop_out,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "fuse_model": fuse_model
    }
    if task.model == LinFuseModel:
        return model_class(**common_args_dict)
    elif task.model == PairsFuseModel:
        common_args_dict["input_dim_2"] = input_dim_2
        common_args_dict["dtype_2"] = dtype_2
        return model_class(**common_args_dict)
    else:
        raise ValueError("Unknown model")


def train_model_with_config(config: dict, task_name: str, fuse_base: str, mol_emd: str, protein_emd: str,
                            max_no_improve=15, fuse_model=None, return_valid=False, task_suffix=""):
    use_fuse = config["use_fuse"]
    use_model = config["use_model"]
    bs = config["bs"]
    lr = config["lr"]

    drop_out = config["drop_out"]
    n_layers = config["n_layers"]
    hidden_dim = config["hidden_dim"]

    task = name_to_task[task_name]
    train_loader, valid_loader, test_loader = get_dataloaders(task_name, mol_emd, protein_emd, bs)
    if task.criterion == torch.nn.BCEWithLogitsLoss:
        train_labels = train_loader.dataset.labels
        positive_sample_weight = train_labels.sum() / len(train_labels)
        negative_sample_weight = 1 - positive_sample_weight
        pos_weight = negative_sample_weight / positive_sample_weight
        criterion = task.criterion(pos_weight=torch.tensor(pos_weight).to(device))
    else:
        criterion = task.criterion()
    if use_fuse and use_model:
        conf = Config.both
    elif use_fuse:
        conf = Config.our
    elif use_model:
        conf = Config.PRE
    else:
        return -1e6, -1e6

    model = get_model_from_task(task, train_loader.dataset, conf, fuse_base=fuse_base, drop_out=drop_out,
                                n_layers=n_layers, hidden_dim=hidden_dim, fuse_model=fuse_model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    no_improve = 0
    scores_manager = ScoresManager(config['metric'])
    for epoch in range(500):
        _ = run_epoch(model, train_loader, optimizer, criterion, config['metric'], "train")
        with torch.no_grad():
            val_score = run_epoch(model, valid_loader, optimizer, criterion, config['metric'], "val")
            test_score = run_epoch(model, test_loader, optimizer, criterion, config['metric'], "test")

        improved = scores_manager.update(val_score, test_score)
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    if return_valid:
        return scores_manager.test_scores.get_value(), scores_manager.valid_scores.get_value()
    return scores_manager.test_scores.get_value()


def main(use_fuse, use_model, bs, lr, drop_out, hidden_dim, task_name, fuse_base, mol_emd, protein_emd, n_layers,
         metric, max_no_improve, fuse_model=None, task_suffix=""):
    config = {
        "use_fuse": use_fuse,
        "use_model": use_model,
        "bs": bs,
        "lr": lr,
        'hidden_dim': hidden_dim,
        'drop_out': drop_out,
        'n_layers': n_layers,
        'metric': metric
    }
    res = train_model_with_config(config, task_name, fuse_base, mol_emd, protein_emd, max_no_improve,
                                  fuse_model=fuse_model, task_suffix=task_suffix)
    return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_fuse", type=int, default=0)
    parser.add_argument("--use_model", type=int, default=1)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--drop_out", type=float, default=0.0)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--task_name", type=str, default="BACE")
    parser.add_argument("--fusion_name", type=str, default="8192-ProtBert-ChemBERTa-2-64-0.3-1-0.001-0.0")
    parser.add_argument("--molecule_embedding", type=str, default="ChemBERTa")
    parser.add_argument("--protein_embedding", type=str, default="ProtBert")
    parser.add_argument("--max_no_improve", type=int, default=15)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--metric", type=str, default="f1_max")
    args = parser.parse_args()
    torch.manual_seed(42)

    r_fuse = main(use_fuse=1, use_model=0, bs=args.bs, lr=args.lr, drop_out=args.drop_out,
                  hidden_dim=args.hidden_dim, task_name=args.task_name, fuse_base=args.fusion_name,
                  mol_emd=args.molecule_embedding, protein_emd=args.protein_embedding,
                  max_no_improve=args.max_no_improve,
                  n_layers=args.n_layers, metric=args.metric)
    r_model = main(use_fuse=0, use_model=1, bs=args.bs, lr=args.lr, drop_out=args.drop_out,
                   hidden_dim=args.hidden_dim, task_name=args.task_name, fuse_base=args.fusion_name,
                   mol_emd=args.molecule_embedding, protein_emd=args.protein_embedding,
                   max_no_improve=args.max_no_improve,
                   n_layers=args.n_layers, metric=args.metric)

    r_both = main(use_fuse=1, use_model=1, bs=args.bs, lr=args.lr, drop_out=args.drop_out,
                  hidden_dim=args.hidden_dim, task_name=args.task_name, fuse_base=args.fusion_name,
                  mol_emd=args.molecule_embedding, protein_emd=args.protein_embedding,
                  max_no_improve=args.max_no_improve,
                  n_layers=args.n_layers, metric=args.metric)
    print(f"Fuse: {r_fuse}, Model: {r_model}, Both: {r_both}")
