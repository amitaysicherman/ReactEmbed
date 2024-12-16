import torch
from torchdrug import metrics


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

    def __init__(self, metric_name, preds=None, reals=None):
        self.metric_name = metric_name
        if self.metric_name in ['mse', 'mae', 'r2']:
            self.value = 1e6
        else:
            self.value = -1e6

        if preds is not None:
            self.calcualte(preds, reals)

    def calcualte(self, preds, reals):
        if self.metric_name == "auc":
            self.value = metrics.area_under_roc(*metric_prep(preds, reals, metrics.area_under_roc)).item()
        elif self.metric_name == "auprc":
            self.value = metrics.area_under_prc(*metric_prep(preds, reals, metrics.area_under_prc)).item()
        elif self.metric_name == "acc":
            self.value = metrics.accuracy(*metric_prep(preds, reals, metrics.accuracy)).item()
        elif self.metric_name == "f1_max":
            self.value = metrics.f1_max(*metric_prep(preds, reals, metrics.f1_max)).item()
        elif self.metric_name == "mse":
            self.value = mse_metric(preds.flatten(), reals.flatten()).item()
        elif self.metric_name == "mae":
            self.value = mae_metric(preds.flatten(), reals.flatten()).item()
        elif self.metric_name == "r2":
            self.value = metrics.r2(preds.flatten(), reals.flatten()).item()
        elif self.metric_name == "pearsonr":
            self.value = metrics.pearsonr(preds.flatten(), reals.flatten()).item()
        elif self.metric_name == "spearmanr":
            self.value = metrics.spearmanr(preds.flatten(), reals.flatten()).item()
        else:
            raise ValueError("Unknown metric")

    def __repr__(self):
        return f"{self.metric_name}: {self.value:.4f}"

    def get_value(self):
        return self.value

    def get_name(self):
        return self.metric_name


class ScoresManager:
    def __init__(self, name):
        self.valid_scores = Scores(name)
        self.test_scores = Scores(name)
        self.name = name

    def update(self, valid_score: Scores, test_score: Scores):

        if self.name in ['mse', 'mae']:
            if valid_score.value < self.valid_scores.value:
                self.valid_scores.value = valid_score.value
                self.test_scores.value = test_score.value
                return True
        else:
            if valid_score.value > self.valid_scores.value:
                self.valid_scores.value = valid_score.value
                self.test_scores.value = test_score.value
                return True
        return False
