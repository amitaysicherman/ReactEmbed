from tqdm import tqdm

td_task_1 = ["BetaLactamase", "BinaryLocalization", "BindingDB", "Davis", "DrugBank", "Fluorescence", "PDBBind"]
td_task_2 = ["HumanPPI", "PPIAffinity", "YeastPPI"]
proteins_seq_files = ["data/reactome/proteins.txt",
                      "data/pathbank/proteins.txt"]
for task_name in td_task_1:
    for split in ['train', 'valid', 'test']:
        proteins_seq_files.append(f"data/torchdrug/{task_name}/{split}_1.txt")
for task_name in td_task_2:
    for split in ['train', 'valid', 'test']:
        proteins_seq_files.append(f"data/torchdrug/{task_name}/{split}_1.txt")
        proteins_seq_files.append(f"data/torchdrug/{task_name}/{split}_2.txt")
all_lines = []
for proteins_seq_file in tqdm(proteins_seq_files):
    with open(proteins_seq_file) as f:
        lines = f.read().replace(".", "").splitlines()
    all_lines.extend(lines)

print(len(all_lines))
all_lines = list(set(all_lines))
print(len(all_lines))
