import os

import numpy as np
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.sdk.forge import ESM3ForgeInferenceClient


class ESM3FEmbedding:
    def __init__(self, token: str, retry: int = 5, sleep_time: int = 60):
        self.emb_model = ESM3ForgeInferenceClient(model="esmc-6b-2024-12", url="https://forge.evolutionaryscale.ai",
                                                  token=token)
        self.token = token
        self.retry = retry
        self.sleep_time = sleep_time

    def to_vec(self, protein_seq: str):
        for i in range(self.retry):
            try:
                protein = ESMProtein(sequence=protein_seq)
                protein_tensor = self.emb_model.encode(protein)
                logits_output = self.emb_model.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                embeddings = logits_output.embeddings[0].numpy().flatten()
                if embeddings is not None:
                    return embeddings
            except Exception as e:
                pass
        return np.zeros(2560)

    def lines_to_vecs(self, lines):
        all_vecs = []
        for line in tqdm(lines):
            if len(line.strip()) == 0:
                all_vecs.append(None)
                continue
            seq = line.strip()
            vec = self.to_vec(seq)
            all_vecs.append(vec)
        all_vecs = np.array(all_vecs)
        return all_vecs


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="reactome")
    parser.add_argument("--start_index", type=int, default=-1)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()
    esm3_embed = ESM3FEmbedding(token=args.token)
    data_name = args.data_name
    start_index = args.start_index
    end_index = args.end_index
    model = "esm3-6b"
    proteins_file = f'data/{data_name}/proteins.txt'
    file = proteins_file.replace(".txt", "_sequences.txt")
    with open(file, "r") as f:
        lines = f.readlines()
    if start_index != -1 and end_index != -1:
        end_index = min(end_index, len(lines))
        lines = lines[start_index:end_index]
        output_file = f"data/{data_name}/{model}_vectors_{start_index}_{end_index}.npy"
    else:
        output_file = f"data/{data_name}/{model}_vectors.npy"
    if os.path.exists(output_file):
        print(f"{output_file} already exists")
        exit(0)
    all_vecs = esm3_embed.lines_to_vecs(lines)
    np.save(output_file, all_vecs)
