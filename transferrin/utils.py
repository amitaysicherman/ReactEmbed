import os
from heapq import heappush, heappushpop
from itertools import combinations

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from tqdm import tqdm

from preprocessing.biopax_parser import get_req, from_second_line
from preprocessing.seq_to_vec import SeqToVec
from transferrin.go_terms import create_and_save_go_matrix

ENZ_FILE = "transferrin/enz.txt"
ENZ_SEQ_FILE = "transferrin/enz_seq.txt"


def save_human_enzyme_binding_proteins(output_file, proteins_to_add=["P02787"]):
    url = f"https://rest.uniprot.org/uniprotkb/stream?fields=accession&format=tsv&query=%28%2A%29%20AND%20%28organism_id%3A9606%29%20AND%20%28go%3A0019899%29"

    response = requests.get(url)

    if response.status_code == 200:
        protein_ids = response.text.strip().split("\n")[1:]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    if proteins_to_add:
        protein_ids.extend(proteins_to_add)
    with open(output_file, "w") as f:
        f.write("\n".join(protein_ids))


def save_all_sequences(ids, output_file):
    def fetch_sequence(protein_id):
        try:
            return get_req(f"https://www.uniprot.org/uniprot/{protein_id}.fasta", ret=1)
        except Exception as e:
            print(f"Error fetching sequence for {protein_id}: {str(e)}")
            return ""

    all_seq = Parallel(n_jobs=-1)(
        delayed(fetch_sequence)(protein_id) for protein_id in tqdm(ids))
    all_fasta = [from_second_line(seq) for seq in all_seq]
    with open(output_file, "w") as f:
        f.write("\n".join(all_fasta))


class Preprocess:
    def __init__(self, p_model="esm3-medium", reactome=False):
        self.reactome = reactome
        self.add_transferrin = not reactome
        os.makedirs(f"transferrin/{p_model}", exist_ok=True)
        if reactome:
            self.proteins_ids_file = "data/reactome/proteins.txt"
            self.proteins_seq_file = None
            self.vec_file = f"data/reactome/{p_model}_vectors.npy"
            self.go_file = f"transferrin/{p_model}/reactome_go.txt"
        else:
            self.proteins_ids_file = ENZ_FILE
            self.proteins_seq_file = ENZ_SEQ_FILE
            self.vec_file = f"transferrin/{p_model}/vecs.npy"
            self.go_file = f"transferrin/{p_model}/go_terms.csv"
        if not reactome:
            if not os.path.exists(self.proteins_ids_file):
                save_human_enzyme_binding_proteins(self.proteins_ids_file)
            if not os.path.exists(self.proteins_seq_file):
                save_all_sequences(self.get_proteins(), self.proteins_seq_file)
            if os.path.exists(self.vec_file):
                with open(self.proteins_seq_file) as f:
                    all_seq = f.read().splitlines()
                seq_to_vec = SeqToVec(model_name=p_model)
                vecs = seq_to_vec.lines_to_vecs(all_seq)
                np.save(self.vec_file, vecs)
        if not os.path.exists(self.go_file):
            create_and_save_go_matrix(self.get_proteins(), self.go_file)

    def get_proteins(self):
        with open(self.proteins_ids_file) as f:
            proteins = f.read().splitlines()
        if self.reactome:
            proteins = [p.split(",")[1] for p in proteins]

        return proteins

    def get_vecs(self):
        return np.load(self.vec_file)

    def get_go_matrix(self):
        return pd.read_csv(self.go_file, index_col=0)


def find_top_n_combinations(df, index, n_results=5, max_cols=3, min_samples=10, binary_cols=None):
    SELECTED_SCORE = float(df.iloc[index]["S"])
    if binary_cols is None:
        binary_cols = [col for col in df.columns if col != 'S']
    target_row = df.iloc[index]
    candidate_cols = [col for col in binary_cols if target_row[col] == 1]
    relevant_cols = ['S'] + candidate_cols
    filtered_df = df[relevant_cols]
    data_array = filtered_df.values
    scores = data_array[:, 0]  # First column is 'S'
    binary_data = data_array[:, 1:]  # Rest are binary columns
    col_to_idx = {col: idx for idx, col in enumerate(candidate_cols)}
    top_results = []
    sequence_num = 0  # Add sequence number for stable comparison
    for length in range(1, max_cols + 1):
        for cols in combinations(candidate_cols, length):
            col_indices = [col_to_idx[col] for col in cols]
            mask = np.all(binary_data[:, col_indices] == 1, axis=1)
            filtered_size = np.sum(mask)
            if filtered_size >= min_samples:
                filtered_scores = scores[mask]
                new_rank = np.sum(filtered_scores >= SELECTED_SCORE) - 1
                rank_ratio = float(new_rank / filtered_size)
                result = {
                    'rank_ratio': rank_ratio,
                    'columns': list(cols),
                    'filtered_size': int(filtered_size),
                    'new_rank': int(new_rank),
                    'lower_rank_indices': np.where(mask & (scores >= SELECTED_SCORE))[0].tolist()
                }
                if len(top_results) < n_results:
                    heappush(top_results, (-rank_ratio, sequence_num, result))
                elif -rank_ratio > top_results[0][0]:  # Compare with negative values
                    heappushpop(top_results, (-rank_ratio, sequence_num, result))
                    print([(-t[0], t[2]['columns']) for t in top_results])  # Print actual rank ratios
                sequence_num += 1
    return top_results


if __name__ == "__main__":
    for p_model in ["ProtBert", "esm3-small", "esm3-medium", "GearNet"]:
        print(f"Processing {p_model}")
        for reactome in [True]:
            print(f"Reactome: {reactome}")
            preprocess = Preprocess(p_model=p_model, reactome=reactome)
