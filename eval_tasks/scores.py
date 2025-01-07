import torch


# from torch.nn import functional as F
# from torch_scatter import scatter_mean


# # ALL the metrics copy from torchdrug.metrics
def auc(pred, target):
    pred = pred.flatten()
    target = target.flatten()
    order = pred.argsort(descending=True)
    target = target[order]
    hit = target.cumsum(0)
    all = (target == 0).sum() * (target == 1).sum()
    auroc = hit[target == 0].sum() / (all + 1e-10)
    return auroc


def multilabel_auc(predictions, targets):
    num_labels = predictions.size(1)
    aurocs = []
    for i in range(num_labels):
        auroc = auc(predictions[:, i], targets[:, i])
        aurocs.append(auroc)
    return torch.stack(aurocs).mean()  # macro average


# def area_under_prc(pred, target):
#     """
#     Area under precision-recall curve (PRC).
#
#     Parameters:
#         pred (Tensor): predictions of shape :math:`(n,)`
#         target (Tensor): binary targets of shape :math:`(n,)`
#     """
#     order = pred.argsort(descending=True)
#     target = target[order]
#     precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
#     auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
#     return auprc
#
#
# def accuracy(pred, target):
#     """
#     Classification accuracy.
#
#     Suppose there are :math:`N` sets and :math:`C` categories.
#
#     Parameters:
#         pred (Tensor): prediction of shape :math:`(N, C)`
#         target (Tensor): target of shape :math:`(N,)`
#     """
#     return (pred.argmax(dim=-1) == target).float().mean()
#

# def f1_max(pred, target):
#     """
#     F1 score with the optimal threshold.
#
#     This function first enumerates all possible thresholds for deciding positive and negative
#     samples, and then pick the threshold with the maximal F1 score.
#
#     Parameters:
#         pred (Tensor): predictions of shape :math:`(B, N)`
#         target (Tensor): binary targets of shape :math:`(B, N)`
#     """
#     order = pred.argsort(descending=True, dim=1)
#     target = target.gather(1, order)
#     precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
#     recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
#     is_start = torch.zeros_like(target).bool()
#     is_start[:, 0] = 1
#     is_start = torch.scatter(is_start, 1, order, is_start)
#
#     all_order = pred.flatten().argsort(descending=True)
#     order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
#     order = order.flatten()
#     inv_order = torch.zeros_like(order)
#     inv_order[order] = torch.arange(order.shape[0], device=order.device)
#     is_start = is_start.flatten()[all_order]
#     all_order = inv_order[all_order]
#     precision = precision.flatten()
#     recall = recall.flatten()
#     all_precision = precision[all_order] - \
#                     torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
#     all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
#     all_recall = recall[all_order] - \
#                  torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
#     all_recall = all_recall.cumsum(0) / pred.shape[0]
#     all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
#     return all_f1.max()
#

# def r2(pred, target):
#     """
#     :math:`R^2` regression score.
#
#     Parameters:
#         pred (Tensor): predictions of shape :math:`(n,)`
#         target (Tensor): targets of shape :math:`(n,)`
#     """
#     total = torch.var(target, unbiased=False)
#     residual = F.mse_loss(pred, target)
#     return 1 - residual / total


# def pearsonr(pred, target):
#     """
#     Pearson correlation between prediction and target.
#
#     Parameters:
#         pred (Tensor): prediction of shape :math: `(N,)`
#         target (Tensor): target of shape :math: `(N,)`
#     """
#     pred_mean = pred.float().mean()
#     target_mean = target.float().mean()
#     pred_centered = pred - pred_mean
#     target_centered = target - target_mean
#     pred_normalized = pred_centered / pred_centered.norm(2)
#     target_normalized = target_centered / target_centered.norm(2)
#     pearsonr = pred_normalized @ target_normalized
#     return pearsonr


# def spearmanr(pred, target):
#     """
#     Spearman correlation between prediction and target.
#
#     Parameters:
#         pred (Tensor): prediction of shape :math: `(N,)`
#         target (Tensor): target of shape :math: `(N,)`
#     """
#
#     def get_ranking(input):
#         input_set, input_inverse = input.unique(return_inverse=True)
#         order = input_inverse.argsort()
#         ranking = torch.zeros(len(input_inverse), device=input.device)
#         ranking[order] = torch.arange(1, len(input) + 1, dtype=torch.float, device=input.device)
#
#         # for elements that have the same value, replace their rankings with the mean of their rankings
#         mean_ranking = scatter_mean(ranking, input_inverse, dim=0, dim_size=len(input_set))
#         ranking = mean_ranking[input_inverse]
#         return ranking
#
#     pred = get_ranking(pred)
#     target = get_ranking(target)
#     covariance = (pred * target).mean() - pred.mean() * target.mean()
#     pred_std = pred.std(unbiased=False)
#     target_std = target.std(unbiased=False)
#     spearmanr = covariance / (pred_std * target_std + 1e-10)
#     return spearmanr
#

# def one_into_two(preds):
#     preds = preds.flatten()
#     probs_class_1 = torch.sigmoid(preds)
#     probs_class_0 = 1 - probs_class_1
#     return torch.cat((probs_class_0, probs_class_1), dim=1)


# def metric_prep(preds, reals, metric):
#     if metric.__name__ == "area_under_roc" or metric.__name__ == "area_under_prc":
#         if preds.dim() > 1 and preds.shape[1] > 1:
#             reals = torch.nn.functional.one_hot(reals.long().flatten(), num_classes=preds.shape[1] + 1).flatten()
#         else:
#             reals = reals.flatten()
#         preds = torch.sigmoid(preds).flatten()
#
#
#     elif metric.__name__ == "f1_max" or metric.__name__ == "accuracy":
#         is_multilabel = reals.shape[1] > 1
#         is_binary = preds.shape[1] <= 1
#
#         if metric.__name__ == "accuracy":
#             if is_multilabel:
#                 preds = preds.flatten().unsqueeze(1)
#                 preds = one_into_two(preds)
#                 reals = reals.flatten()
#             elif is_binary:
#                 preds = preds.flatten().unsqueeze(1)
#                 preds = one_into_two(preds)
#                 reals = reals.flatten()
#             else:
#                 preds = torch.softmax(preds, dim=1)
#                 reals = reals.long().flatten()
#         else:  # f1_max
#             if is_binary:
#                 preds = preds.flatten().unsqueeze(1)
#                 preds = one_into_two(preds)
#                 reals = torch.nn.functional.one_hot(reals.long().flatten(), num_classes=2)
#             elif is_multilabel:
#                 preds = torch.sigmoid(preds)
#                 reals = reals
#             else:
#                 preds = torch.softmax(preds, dim=1)
#                 reals = reals
#     else:
#         raise ValueError("Unknown metric")
#     return preds, reals
#
#
# def mse_metric(output, target):
#     squared_diff = (output - target) ** 2
#     mse = torch.mean(squared_diff)
#     return mse


def rmse_metric(output, target):
    squared_diff = (output - target) ** 2
    mse = torch.mean(squared_diff)
    return -1 * torch.sqrt(mse)


#
#
# def mae_metric(output, target):
#     abs_diff = torch.abs(output - target)
#     mae = torch.mean(abs_diff)
#     return mae
#

class Scores:

    def __init__(self, metric_name, preds=None, reals=None):
        self.metric_name = metric_name
        assert self.metric_name in ["auc", "rmse"]
        # if self.metric_name in ['mse', 'mae', 'r2']:
        #     self.value = 1e6
        # else:
        #     self.value = -1e6
        self.value = -1e6
        if preds is not None:
            self.calcualte(preds, reals)

    # def binary_classification_f1_max(self, preds, reals):
    #     preds = torch.sigmoid(preds)
    #     if preds.dim() == 1:
    #         preds = preds.unsqueeze(1)
    #     if reals.dim() == 1:
    #         reals = reals.unsqueeze(1)
    #     return f1_max(preds, reals).item()

    def calcualte(self, preds, reals):
        if self.metric_name == "rmse":
            self.value = rmse_metric(preds, reals).item()
        elif self.metric_name == "auc":
            self.value = multilabel_auc(preds, reals).item()
            # if preds.shape[1] == 1:
            #     # binary classification
            #     self.value = self.binary_classification_f1_max(preds, reals)
            # else:
            #     # multiclass classification
            #     values = []
            #     for i in range(preds.shape[1]):
            #         values.append(self.binary_classification_f1_max(preds[:, i], reals[:, i]))
            #     self.value = sum(values) / len(values)

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
        # if self.name in ['mse', 'mae']:
        #     if valid_score.value < self.valid_scores.value:
        #         self.valid_scores.value = valid_score.value
        #         self.test_scores.value = test_score.value
        #         return True
        # else:
        if valid_score.value > self.valid_scores.value:
            self.valid_scores.value = valid_score.value
            self.test_scores.value = test_score.value
            return True
        return False
