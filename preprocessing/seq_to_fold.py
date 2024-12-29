import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils import atom14_to_atom37, to_pdb
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def fold_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


class ESMFold:
    def __init__(self, ):

        self.fold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device).eval()

    def fold_seq(self, seq: str, output_file):
        if len(seq) > 550:
            seq = seq[:550]
        tokenized_input = self.fold_tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            output = self.fold_model(tokenized_input)
        pdbs = fold_to_pdb(output)
        with open(output_file, "w") as output_io:
            output_io.write(pdbs[0])

    def fold_save_lines(self, lines, output_dir, start_index=0):
        for i, line in tqdm(enumerate(lines)):
            output_file = os.path.join(output_dir, f"{start_index + i}.pdb")
            self.fold_seq(line, output_file)


def main(data_name, start_index, end_index):
    proteins_file = f'data/{data_name}/proteins.txt'
    fold_model = ESMFold()
    file = proteins_file.replace(".txt", "_sequences.txt")
    with open(file, "r") as f:
        lines = f.readlines()
    if start_index != 0 and end_index != -1:
        end_index = min(end_index, len(lines))
        lines = lines[start_index:end_index]
    output_dir = f"data/{data_name}/fold/"
    os.makedirs(output_dir, exist_ok=True)
    fold_model.fold_save_lines(lines, output_dir, start_index=start_index)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert sequence to vector')
    parser.add_argument('--data_name', type=str, help='Data name', default="reactome")
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=-1)

    args = parser.parse_args()
    main(args.data_name, args.start_index, args.end_index)
