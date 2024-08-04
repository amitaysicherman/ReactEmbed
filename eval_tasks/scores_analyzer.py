import argparse

import pandas as pd
from scipy.stats import ttest_ind

from common.data_types import NAME_TO_UI, ESM_1B, PEBCHEM10M

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", type=int, default=1)
parser.add_argument("--print_csv", type=int, default=1)

args = parser.parse_args()

no = 'trip' if args.ablation == 0 else 'no'
ablations = ['trip', no, 'noport', 'long']

# Configuration Constants
our = "our"
both = "both"
pre = "pre"
SELECTED_METRIC = "selected_metric"
ablation_col = "ablation"
ablation_config_prot = ESM_1B
ablation_config_mol = PEBCHEM10M

# Define columns and mappings
index_cols = ['task_name', 'protein_emd', 'mol_emd']
conf = ['conf']
metrics_cols = ['mse', 'mae', 'r2', 'pearsonr', 'spearmanr', 'auc', 'auprc', 'acc', 'f1_max']
all_cols = index_cols + conf + metrics_cols
conf_cols = ['pre', 'our', 'both']
type_to_metric = {'M': 'acc', 'P': "pearsonr", 'PD': "auprc", 'PDA': "pearsonr", 'PPI': "auprc", 'PPIA': "pearsonr"}

# Mapping from task names to types
name_to_type_dict = {
    'BetaLactamase': "P", 'Fluorescence': "P", 'Stability': "P", 'Solubility': "P",
    'HumanPPI': "PPI", 'YeastPPI': "PPI", 'PPIAffinity': "PPIA", 'BindingDB': "PDA",
    'PDBBind': "PDA", 'BACE': "M", 'BBBP': "M", 'ClinTox': "M", 'HIV': "M",
    'SIDER': "M", 'Tox21': "M", 'DrugBank': "PD", 'Davis': "PD", 'KIBA': "PD"
}

TYPE_TO_NAME = {
    'P': 'Protein Function', 'PPI': 'Protein-Protein Interaction',
    "PPIA": 'Protein-Protein Interaction Affinity',
    'M': 'Molecule Property', 'PD': 'Protein-Drug Interaction',
    "PDA": 'Protein-Drug Interaction Affinity'
}
# METRIC_TO_NAME = {
#     'mse': 'Mean Squared Error', 'mae': 'Mean Absolute Error', 'r2': 'R2', 'pearsonr': 'Pearson Correlation',
#     'spearmanr': 'Spearman Correlation', 'auc': 'Area Under the ROC Curve (AUC)',
#     'auprc': 'Area Under the PR Curve (AUPRC)',
#     'acc': 'Accuracy', 'f1_max': 'F1 Max Score'
# }
METRIC_TO_NAME = {
    'mse': 'MSE', 'mae': 'MAE', 'r2': 'R2', 'pearsonr': 'Pearson', 'spearmanr': 'Spearman',
    'auc': 'AUC', 'auprc': 'AUPRC', 'acc': 'Accuracy', 'f1_max': 'F1 Max'
}
COLS_TO_NAME = {
    'task_name': 'Task', 'protein_emd': 'Protein Embedding', 'mol_emd': 'Molecule Embedding', 'task_type': 'Task Type'
}
lower_is_better_metrics = {'mse', 'mae'}


def name_to_type(x):
    return name_to_type_dict[x.split("_")[0]]


def task_to_selected_matic(task):
    if task in name_to_type_dict:
        return type_to_metric[name_to_type_dict[task]]
    elif task.split("_")[0] in name_to_type_dict:
        return type_to_metric[name_to_type(task.split("_")[0])]
    else:
        return None


def df_to_selected_matic(df):
    bool_filter = []
    for i, row in df.iterrows():
        metric = task_to_selected_matic(row['task_name'])
        if metric is None:
            bool_filter.append(False)
            continue
        else:
            bool_filter.append(True)
        if metric in ['mse', 'mae']:
            df.loc[i, SELECTED_METRIC] = -1 * df.loc[i, metric]
        else:
            df.loc[i, SELECTED_METRIC] = df.loc[i, metric]
    df = df[bool_filter]
    return df


def round_num(x):
    if pd.isna(x):
        return 0

    return abs(round(x * 100, 2))  # for mse and mae


def get_format_results_agg_ablations(group):
    group_our = group[group['conf'] == our]

    ablations_data_our = {
        ablation: group_our[group_our[ablation_col] == ablation][SELECTED_METRIC] for ablation in ablations
    }

    group_both = group[group['conf'] == both]
    ablations_data_both = {
        ablation: group_both[group_both[ablation_col] == ablation][SELECTED_METRIC] for ablation in ablations
    }
    ablations_data = {
        ablation: ablations_data_our[ablation] if ablations_data_our[ablation].mean() > ablations_data_both[
            ablation].mean() else ablations_data_both[ablation] for ablation in ablations
    }

    ablations_means = {ablation: data.mean() for ablation, data in ablations_data.items()}
    ablations_stds = {ablation: data.std() for ablation, data in ablations_data.items()}
    ablations_is_max = {ablation: ablations_means[ablation] == max(ablations_means.values()) for ablation in ablations}
    format_results = []
    for ablation in ablations:
        format_value = f"{round_num(ablations_means[ablation])}({round_num(ablations_stds[ablation])})"
        if ablations_is_max[ablation]:
            format_value = f"\\textbf{{{format_value}}}"
        format_results.append(format_value)
    return format_results


def get_format_results_agg_no_ablations(group):
    group = group[group[ablation_col] == no]
    pre_values = group[group['conf'] == pre][SELECTED_METRIC]
    our_values = group[group['conf'] == our][SELECTED_METRIC]
    both_values = group[group['conf'] == both][SELECTED_METRIC]
    # Calculate mean and standard deviation for each configuration
    pre_mean, pre_std = pre_values.mean(), pre_values.std()
    our_mean, our_std = our_values.mean(), our_values.std()
    both_mean, both_std = both_values.mean(), both_values.std()
    # Determine the best configuration based on the metric
    if SELECTED_METRIC in lower_is_better_metrics:
        best_mean, best_std = (our_mean, our_std) if our_mean < both_mean else (both_mean, both_std)
        best_values = our_values if our_mean < both_mean else both_values
    else:
        best_mean, best_std = (our_mean, our_std) if our_mean > both_mean else (both_mean, both_std)
        best_values = our_values if our_mean > both_mean else both_values

    # Perform t-test between 'pre' and 'best'
    t_stat, p_value = ttest_ind(pre_values, best_values, equal_var=False, nan_policy='omit')

    # Check if the difference is statistically significant
    significant = p_value < 0.05

    # Format the results with potential bolding for statistical significance
    if significant:
        if (SELECTED_METRIC in lower_is_better_metrics and pre_mean > best_mean) or \
                (SELECTED_METRIC not in lower_is_better_metrics and pre_mean < best_mean):
            # If the best is significantly better, bold the best
            best_result = f"\\textbf{{{round_num(best_mean)}}}({round_num(best_std)})"
            pre_result = f"{round_num(pre_mean)}({round_num(pre_std)})"
        else:
            # If pre is significantly better, bold pre
            best_result = f"{round_num(best_mean)}({round_num(best_std)})"
            pre_result = f"\\textbf{{{round_num(pre_mean)}}}({round_num(pre_std)})"
    else:
        # No significant difference, just format normally
        pre_result = f"{round_num(pre_mean)}({round_num(pre_std)})"
        best_result = f"{round_num(best_mean)}({round_num(best_std)})"

    return pre_result, best_result


def get_format_results_agg(group):
    if args.ablation == 0:
        return get_format_results_agg_no_ablations(group)
    else:
        return get_format_results_agg_ablations(group)


def add_ablation_col(data):
    data[ablation_col] = data['task_name'].apply(lambda x: x.split("_")[1] if len(x.split("_")) > 1 else no)
    data['task_name'] = data['task_name'].apply(lambda x: x.split("_")[0])
    return data


# Load data
data = pd.read_csv("data/scores/torchdrug.csv", on_bad_lines='warn')
data = data.dropna()
data = add_ablation_col(data)
data = data[data['task_name'] != 'BACE']

data = df_to_selected_matic(data)

# Group by and apply aggregation
print(index_cols)
format_results = data.groupby(index_cols).apply(get_format_results_agg)

# Convert the results to a DataFrame for easy handling
columns_names = ['Pretrained Models', 'Our'] if args.ablation == 0 else ablations
format_results_df = pd.DataFrame(format_results.tolist(), columns=columns_names,
                                 index=format_results.index)
format_results_df = format_results_df.reset_index()
format_results_df['task_type'] = format_results_df['task_name'].apply(name_to_type)

if args.ablation == 1:
    filter_bool = []
    # for i, row in format_results_df.iterrows():
    #     if row['task_type'] in ["P", "PPI", "PPIA"]:
    #         filter_bool.append(row['protein_emd'] == ablation_config_prot)
    #     elif row['task_type'] in ["M"]:
    #         filter_bool.append(row['mol_emd'] == ablation_config_mol)
    #     else:
    #         filter_bool.append(row['protein_emd'] == ablation_config_prot and row['mol_emd'] == ablation_config_mol)
    # format_results_df = format_results_df[filter_bool]
    # format_results_df = format_results_df.drop(columns=['protein_emd', 'mol_emd'])
    format_results_df = format_results_df.sort_values(by=['task_type', 'task_name'])
    format_results_df['task_type'] = format_results_df['task_type'].apply(lambda x: TYPE_TO_NAME[x])
    print(format_results_df)
    format_results_df = format_results_df.rename(columns={
        "trip": "Our",
        'task_type': 'Task Type',
        'task_name': 'Task',
        'noport': 'No Proteins Anchors',
        'long': 'Long Traning',
        # 'comp': 'Complex Model',
        'no': 'No Triplet',
    })
    format_results_df.set_index(['Task Type', "Task"], inplace=True)

    print(format_results_df.to_latex(index=True, escape=False, caption="Ablation Study Results",
                                     label="tab:ablation_results", column_format="ll|llll").replace("begin{table}",
                                                                                                    "begin{table}\n\centering"))
    exit(0)

format_results_df['protein_emd'] = format_results_df['protein_emd'].apply(lambda x: NAME_TO_UI[x])
format_results_df['mol_emd'] = format_results_df['mol_emd'].apply(lambda x: NAME_TO_UI[x])
format_results_df['Metric'] = format_results_df['task_name'].apply(lambda x: METRIC_TO_NAME[task_to_selected_matic(x)])
format_results_df['task_type'] = format_results_df['task_type'].apply(lambda x: TYPE_TO_NAME[x])
format_results_df = format_results_df.sort_values(by=['task_type', 'task_name', 'protein_emd', 'mol_emd'])
for i, row in format_results_df.iterrows():
    if row['task_type'] in [TYPE_TO_NAME[x] for x in ["P", "PPI", "PPIA"]]:
        format_results_df.loc[i, 'mol_emd'] = "-"
    elif row['task_type'] == TYPE_TO_NAME["M"]:
        format_results_df.loc[i, 'protein_emd'] = "-"

format_results_df.rename(columns=COLS_TO_NAME, inplace=True)

index_cols_print = [COLS_TO_NAME[x] for x in ['task_type', 'task_name']]
format_results_df.set_index(index_cols_print, inplace=True)
format_results_df = format_results_df[
    [COLS_TO_NAME['protein_emd'], COLS_TO_NAME['mol_emd'], 'Metric', 'Pretrained Models', 'Our']]
print(format_results_df)
print(format_results_df.to_latex(index=True, escape=False, caption="Results", label="tab:results",
                                 column_format="llll|c|cc").replace("begin{table}", "begin{table}\n\centering"))

# def print_format_latex(data: pd.DataFrame):
#     tast_type = data['task_type'].iloc[0]
#     caption = f'{METRIC_TO_NAME[type_to_metric[tast_type]]},{TYPE_TO_NAME[tast_type]}'
#     label = f'tab:{tast_type}_results'
#     index_cols_print = index_cols[:]
#     data = data.drop(columns=['task_type'])
#     if data['task_name'].nunique() == 1:
#         data = data.drop(columns=['task_name'])
#         index_cols_print.remove('task_name')
#     if data['protein_emd'].nunique() == 1:
#         data = data.drop(columns=['protein_emd'])
#         index_cols_print.remove('protein_emd')
#     if data['mol_emd'].nunique() == 1:
#         data = data.drop(columns=['mol_emd'])
#         index_cols_print.remove('mol_emd')
#     data.rename(columns=COLS_TO_NAME, inplace=True)
#     index_cols_print = [COLS_TO_NAME[x] for x in index_cols_print]
#     data = data.set_index(index_cols_print)
#     len_index = len(index_cols_print)
#     col_format = 'l' * len_index + "|" + 'l' * len(data.columns)
#
#     print("----------------\n" * 5)
#     print(caption)
#     print("----------------\n" * 5)
#
#     if args.print_csv:
#         print(data)
#     print(data.to_latex(index=True, escape=False, caption=caption, label=label, column_format=col_format).replace(
#         "begin{table}", "begin{table}\n\centering"))
#
#
# format_results_df.groupby('task_type').apply(print_format_latex)
