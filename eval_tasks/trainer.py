import os

import torch
from torchdrug import metrics

from common.data_types import Config
from common.path_manager import scores_path
from eval_tasks.dataset import get_dataloaders
from eval_tasks.models import LinFuseModel, PairsFuseModel
from eval_tasks.tasks import name_to_task, Task

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def one_into_two(preds):
    probs_class_1 = torch.sigmoid(preds)
    probs_class_0 = 1 - probs_class_1
    return torch.cat((probs_class_0, probs_class_1), dim=1)


def metric_prep(preds, reals, metric):
    if metric.__name__ == "area_under_roc" or metric.__name__ == "area_under_prc":
        if preds.dim() > 1 and preds.shape[1] > 1:
            reals = torch.nn.functional.one_hot(reals.long().flatten(), num_classes=preds.shape[1] + 1).flatten()
        else:
            reals = reals.flatten()
        preds = torch.sigmoid(preds).flatten()


    elif metric.__name__ == "f1_max" or metric.__name__ == "accuracy":
        is_multilabel = reals.shape[1] > 1
        is_binary = preds.shape[1] <= 1

        if metric.__name__ == "accuracy":
            if is_multilabel:
                preds = preds.flatten().unsqueeze(1)
                preds = one_into_two(preds)
                reals = reals.flatten()
            elif is_binary:
                preds = preds.flatten().unsqueeze(1)
                preds = one_into_two(preds)
                reals = reals.flatten()
            else:
                preds = torch.softmax(preds, dim=1)
                reals = reals.long().flatten()
        else:  # f1_max
            if is_binary:
                preds = preds.flatten().unsqueeze(1)
                preds = one_into_two(preds)
                reals = torch.nn.functional.one_hot(reals.long().flatten(), num_classes=2)
            elif is_multilabel:
                preds = torch.sigmoid(preds)
                reals = reals
            else:
                preds = torch.softmax(preds, dim=1)
                reals = reals
    else:
        raise ValueError("Unknown metric")
    return preds, reals


def mse_metric(output, target):
    squared_diff = (output - target) ** 2
    mse = torch.mean(squared_diff)
    return mse


def mae_metric(output, target):
    abs_diff = torch.abs(output - target)
    mae = torch.mean(abs_diff)
    return mae


class Scores:

    def __init__(self, mode, preds=None, reals=None):
        self.mode = mode
        if mode == "classification":
            self.auc: float = -1e6
            self.auprc: float = -1e6
            self.acc: float = -1e6
            self.f1_max: float = -1e6

        else:
            self.mse: float = 1e6
            self.mae: float = 1e6
            self.r2: float = -1e6
            self.pearsonr: float = -1e6
            self.spearmanr: float = -1e6

        if preds is not None:
            self.calcualte(preds, reals)

    def calcualte(self, preds, reals):
        if self.mode == "classification":
            self.auc = metrics.area_under_roc(*metric_prep(preds, reals, metrics.area_under_roc)).item()
            self.auprc = metrics.area_under_prc(*metric_prep(preds, reals, metrics.area_under_prc)).item()
            self.acc = metrics.accuracy(*metric_prep(preds, reals, metrics.accuracy)).item()
            self.f1_max = metrics.f1_max(*metric_prep(preds, reals, metrics.f1_max)).item()
        else:
            self.mse = mse_metric(preds.flatten(), reals.flatten()).item()
            self.mae = mae_metric(preds.flatten(), reals.flatten()).item()
            self.r2 = metrics.r2(preds.flatten(), reals.flatten()).item()
            self.pearsonr = metrics.pearsonr(preds.flatten(), reals.flatten()).item()
            self.spearmanr = metrics.spearmanr(preds.flatten(), reals.flatten()).item()

    def __repr__(self):
        if self.mode == "regression":
            return f"MSE: {self.mse}, MAE: {self.mae}, R2: {self.r2}, Pearsonr: {self.pearsonr}, Spearmanr: {self.spearmanr}\n"
        else:
            return f"AUC: {self.auc}, AUPRC: {self.auprc}, ACC: {self.acc}, F1: {self.f1_max}\n"

    def get_regression_metrics(self):
        if self.mode == "regression":
            return [self.mse, self.mae, self.r2, self.pearsonr, self.spearmanr]
        else:
            return [0, 0, 0, 0, 0]

    def get_classification_metrics(self):
        if self.mode == "classification":
            return [self.auc, self.auprc, self.acc, self.f1_max]
        else:
            return [0, 0, 0, 0]

    def get_metrics(self):
        return self.get_regression_metrics() + self.get_classification_metrics()

    def get_metrics_names(self):
        return ["mse", "mae", "r2", "pearsonr", "spearmanr"] + ["auc", "auprc", "acc", "f1_max"]


class ScoresManager:
    def __init__(self, mode):
        self.valid_scores = Scores(mode=mode)
        self.test_scores = Scores(mode=mode)

    def update_classification(self, valid_score: Scores, test_score: Scores):
        improved = False
        if valid_score.auc > self.valid_scores.auc:
            self.valid_scores.auc = valid_score.auc
            self.test_scores.auc = test_score.auc
            improved = True
        if valid_score.auprc > self.valid_scores.auprc:
            self.valid_scores.auprc = valid_score.auprc
            self.test_scores.auprc = test_score.auprc
            improved = True
        if valid_score.acc > self.valid_scores.acc:
            self.valid_scores.acc = valid_score.acc
            self.test_scores.acc = test_score.acc
            improved = True
        if valid_score.f1_max > self.valid_scores.f1_max:
            self.valid_scores.f1_max = valid_score.f1_max
            self.test_scores.f1_max = test_score.f1_max
            improved = True
        return improved

    def update_regression(self, valid_score: Scores, test_score: Scores):
        improved = False
        if valid_score.mse < self.valid_scores.mse:
            self.valid_scores.mse = valid_score.mse
            self.test_scores.mse = test_score.mse
            improved = True
        if valid_score.mae < self.valid_scores.mae:
            self.valid_scores.mae = valid_score.mae
            self.test_scores.mae = test_score.mae
            improved = True
        if valid_score.r2 > self.valid_scores.r2:
            self.valid_scores.r2 = valid_score.r2
            self.test_scores.r2 = test_score.r2
            improved = True
        if valid_score.pearsonr > self.valid_scores.pearsonr:
            self.valid_scores.pearsonr = valid_score.pearsonr
            self.test_scores.pearsonr = test_score.pearsonr
            improved = True
        if valid_score.spearmanr > self.valid_scores.spearmanr:
            self.valid_scores.spearmanr = valid_score.spearmanr
            self.test_scores.spearmanr = test_score.spearmanr
            improved = True
        return improved

    def update(self, valid_score: Scores, test_score: Scores):
        if valid_score.mode == "classification":
            return self.update_classification(valid_score, test_score)
        else:
            return self.update_regression(valid_score, test_score)


def run_epoch(model, loader, optimizer, criterion, mode, part):
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
        return Scores(mode, preds, reals)
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
    if task.model == LinFuseModel:
        return model_class(input_dim=input_dim_1, input_type=dtype_1, output_dim=output_dim, conf=conf,
                           fuse_base=fuse_base, fuse_model=fuse_model, drop_out=drop_out, n_layers=n_layers,
                           hidden_dim=hidden_dim)
    elif task.model == PairsFuseModel:
        return model_class(input_dim_1=input_dim_1, dtpye_1=dtype_1, input_dim_2=input_dim_2, dtype_2=dtype_2,
                           output_dim=output_dim, conf=conf, fuse_base=fuse_base, fuse_model=fuse_model,
                           drop_out=drop_out, n_layers=n_layers, hidden_dim=hidden_dim)
    else:
        raise ValueError("Unknown model")


def train_model_with_config(config: dict, task_name: str, fuse_base: str, mol_emd: str, protein_emd: str,
                            print_output=False, max_no_improve=15, fuse_model=None, return_valid=False, task_suffix=""):
    use_fuse = config["use_fuse"]
    use_model = config["use_model"]
    bs = config["bs"]
    lr = config["lr"]

    drop_out = config["drop_out"]
    # n_layers = config["n_layers"]
    hidden_dim = config["hidden_dim"]

    task = name_to_task[task_name]
    n_layers = task.n_layers
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
        if print_output:
            print("No model selected")
        return -1e6, -1e6

    model = get_model_from_task(task, train_loader.dataset, conf, fuse_base=fuse_base, drop_out=drop_out,
                                n_layers=n_layers, hidden_dim=hidden_dim, fuse_model=fuse_model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if print_output:
        print(model)
    no_improve = 0
    scores_manager = ScoresManager(mode=task.metric)
    # best_valid_score = -1e6
    # best_test_score = -1e6
    for epoch in range(250):
        _ = run_epoch(model, train_loader, optimizer, criterion, task.metric, "train")
        with torch.no_grad():
            val_score = run_epoch(model, valid_loader, optimizer, criterion, task.metric, "val")
            test_score = run_epoch(model, test_loader, optimizer, criterion, task.metric, "test")

        if print_output:
            print(epoch, val_score, test_score)
        improved = scores_manager.update(val_score, test_score)
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
    if print_output:
        print("Best Test scores\n", scores_manager.test_scores)
        output_file = f"{scores_path}/torchdrug.csv"

        if not os.path.exists(output_file):
            names = ["task_name", "mol_emd", "protein_emd", "conf"] + scores_manager.test_scores.get_metrics_names()
            with open(output_file, "w") as f:
                f.write(",".join(names) + "\n")
        values = [task_name + task_suffix, mol_emd, protein_emd, conf.value] + scores_manager.test_scores.get_metrics()
        with open(output_file, "a") as f:
            f.write(",".join(map(str, values)) + "\n")
    if return_valid:
        return scores_manager.test_scores.get_metrics(), scores_manager.valid_scores.get_metrics()
    return scores_manager.test_scores.get_metrics()


def main(use_fuse, use_model, bs, lr, drop_out, hidden_dim, task_name, fuse_base, mol_emd, protein_emd, print_output,
         max_no_improve, fuse_model=None, task_suffix=""):
    config = {
        "use_fuse": use_fuse,
        "use_model": use_model,
        "bs": bs,
        "lr": lr,
        'hidden_dim': hidden_dim,
        'drop_out': drop_out
    }
    train_model_with_config(config, task_name, fuse_base, mol_emd, protein_emd, print_output, max_no_improve,
                            fuse_model=fuse_model, task_suffix=task_suffix)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_fuse", type=int, default=0)
    parser.add_argument("--use_model", type=int, default=0)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--drop_out", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--task_name", type=str, default="SIDER")
    parser.add_argument("--fusion_name", type=str, default="ProtBert-ChemBERTa")
    parser.add_argument("--molecule_embedding", type=str, default="ChemBERTa")
    parser.add_argument("--protein_embedding", type=str, default="ProtBert")
    parser.add_argument("--print_downstream_results", type=int, default=1)
    parser.add_argument("--max_no_improve", type=int, default=15)
    args = parser.parse_args()
    main(use_fuse=args.use_fuse, use_model=args.use_model, bs=args.bs, lr=args.lr, drop_out=args.drop_out,
         hidden_dim=args.hidden_dim, task_name=args.task_name, fuse_base=args.fusion_name,
         mol_emd=args.molecule_embedding, protein_emd=args.protein_embedding,
         print_output=args.print_downstream_results, max_no_improve=args.max_no_improve)
