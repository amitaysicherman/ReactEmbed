import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from functools import partial
from itertools import combinations
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed
from tqdm import tqdm

from preprocessing.biopax_parser import get_req, from_second_line
from preprocessing.seq_to_vec import SeqToVec

IDS_FILE = "transferrin/human_enzyme_binding_proteins.txt"
SEQ_FILE = "transferrin/all_sequences.txt"
VEC_FILE = "transferrin/esm3-medium_vecs.npy"
GO_FILE = "transferrin/go.csv"
GO_ANCESTORS_FIRE = "transferrin/go_ancestors.txt"
TRANSFERRIN_FILE = "transferrin/transferrin.txt"
GO_REACTOME = "transferrin/go_reactome.txt"
GO_REACTOME_ANCESTORS = "transferrin/go_reactome_ancestors.txt"


def get_reactome_vecs():
    return np.load("data/reactome/esm3-medium_vectors.npy")


def save_reactome_go_terms():
    with open("data/reactome/proteins.txt") as f:
        proteins = f.read().splitlines()
    proteins = [protein.split(",")[1] for protein in proteins]
    n_cores = max(1, os.cpu_count() - 1)
    n_cores = min(n_cores, 32)
    all_goes = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Map the process_protein function across all proteins
        results = executor.map(process_protein, proteins)
        for protein, go_terms in results:
            go_terms = " ".join(go_terms)
            all_goes.append(go_terms)
    with open(GO_REACTOME, "w") as f:
        f.write("\n".join(all_goes))


def save_reactome_go_ancestors():
    with open(GO_REACTOME) as f:
        go_terms = f.read().splitlines()
    go_terms = [go_term.split() for go_term in go_terms]
    all_goes = set()
    for go_term in go_terms:
        all_goes.update(go_term)
    all_goes = list(all_goes)
    n_cores = max(1, os.cpu_count() - 1)
    n_cores = min(n_cores, 32)
    mapping_lines = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Map the process_protein function across all proteins
        results = executor.map(get_go_ancestors_cached, all_goes)
        for go_term, ancestors in zip(all_goes, results):
            print(f"Processed {go_term}")
            line = f"{go_term}|{' '.join(ancestors)}\n"
            mapping_lines.append(line)
    with open(GO_REACTOME_ANCESTORS, "w") as f:
        f.writelines(mapping_lines)


def process_combination(cols, df, SELECTED_R, min_samples):
    """Helper function to process a single combination of columns"""
    # Create filter mask - all selected columns must be 1
    mask = pd.Series(True, index=df.index)
    for col in cols:
        mask &= (df[col] == 1)

    filtered_df = df[mask]
    filtered_size = len(filtered_df)

    # If we don't meet minimum samples, return None
    if filtered_size < min_samples:
        print(cols, "filtered_size", filtered_size)
        return None

    # Calculate new rank (0-based) in filtered dataset
    new_rank = (filtered_df['R'] >= SELECTED_R).sum() - 1

    # Calculate rank ratio (lower is better)
    rank_ratio = new_rank / filtered_size
    print(cols, rank_ratio, new_rank, filtered_size)
    return {
        'columns': list(cols),
        'rank_ratio': rank_ratio,
        'filtered_size': filtered_size,
        'new_rank': new_rank
    }


def find_optimal_filter_columns(df, index=0, min_samples=500, binary_cols=None, n=3):
    """Find optimal filter columns using parallel processing"""
    SELECTED_R = float(df.iloc[index]["R"])
    if binary_cols is None:
        binary_cols = [col for col in df.columns if col != 'R']

    target_row = df.iloc[index]

    # Only consider columns where target row has value 1
    candidate_cols = [col for col in binary_cols if target_row[col] == 1]

    # Generate all possible combinations
    all_combinations = []
    for length in range(1, n + 1):
        all_combinations.extend(combinations(candidate_cols, length))

    # Create partial function with fixed parameters
    process_func = partial(process_combination,
                           df=df,
                           SELECTED_R=SELECTED_R,
                           min_samples=min_samples)

    # Use all available CPUs except one
    num_cpus = max(1, cpu_count() - 1)

    # Process combinations in parallel
    with Pool(num_cpus) as pool:
        results = pool.map(process_func, all_combinations)

    # Filter out None results and find the best combination
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        return [], 0, 0

    # Sort by rank_ratio (primary) and number of columns (secondary)
    best_result = min(valid_results,
                      key=lambda x: (x['rank_ratio'], len(x['columns'])))

    return (
        best_result['columns'],
        best_result['new_rank'],
        best_result['filtered_size']
    )


def save_human_enzyme_binding_proteins():
    url = f"https://rest.uniprot.org/uniprotkb/stream?fields=accession&format=tsv&query=%28%2A%29%20AND%20%28organism_id%3A9606%29%20AND%20%28go%3A0019899%29"

    response = requests.get(url)

    if response.status_code == 200:
        protein_ids = response.text.strip().split("\n")[1:]
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    with open(IDS_FILE, "w") as f:
        f.write("\n".join(protein_ids))


def get_human_enzyme_binding_proteins():
    with open(IDS_FILE) as f:
        proteins = f.read().splitlines()
    return proteins


def save_all_sequences(human_enzyme_binding_proteins):
    def fetch_sequence(protein_id):
        try:
            return get_req(f"https://www.uniprot.org/uniprot/{protein_id}.fasta", ret=1)
        except Exception as e:
            print(f"Error fetching sequence for {protein_id}: {str(e)}")
            return ""

    all_seq = Parallel(n_jobs=-1)(
        delayed(fetch_sequence)(protein_id) for protein_id in tqdm(human_enzyme_binding_proteins))

    all_fasta = [from_second_line(seq) for seq in all_seq]
    with open(SEQ_FILE, "w") as f:
        f.write("\n".join(all_fasta))


def get_all_sequences():
    with open(SEQ_FILE) as f:
        all_seq = f.read().splitlines()
    return all_seq


def save_vecs():
    seq_to_vec = SeqToVec(model_name="esm3-medium")
    all_seq = get_all_sequences()
    print("Converting sequences to vectors")
    print(f"Total sequences: {len(all_seq)}")
    vecs = seq_to_vec.lines_to_vecs(all_seq)
    print("Saving vectors")
    print(f"Total vectors: {len(vecs)}")
    np.save(VEC_FILE, vecs)


def get_vecs():
    return np.load(VEC_FILE)


@lru_cache(maxsize=10000)
def get_go_ancestors_cached(term):
    """Cached function to get GO term ancestors"""
    try:
        url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{term}/ancestors"
        try:
            response = requests.get(url, headers={"Accept": "application/json"})
        except Exception as e:
            print(f"Error fetching ancestors for {term}: {str(e)}")
            return set()

        if response.status_code == 200:
            ancestors = response.json()
            return ancestors['results'][0]['ancestors']
    except Exception as e:
        print(f"Error fetching ancestors for {term}: {str(e)}")

    return set()


def get_go_terms(uniprot_id):
    """Get GO terms for a UniProt ID with cached ancestry lookup"""
    print(f"Processing {uniprot_id}")
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json"
    try:
        response = requests.get(uniprot_url)
    except Exception as e:
        print(f"Error fetching UniProt data: {str(e)}")
        return set()
    if response.status_code != 200:
        print(f"Error fetching UniProt data: {response.status_code}")
        return set()

    data = response.json()
    direct_terms = set()

    if 'uniProtKBCrossReferences' in data:
        for ref in data['uniProtKBCrossReferences']:
            if ref['database'] == 'GO':
                go_id = ref['id']
                direct_terms.add(go_id)
    return direct_terms


def process_protein(protein):
    """Process a single protein and return its GO terms"""
    return (protein, get_go_terms(protein))


def save_all_go_terms():
    proteins = get_human_enzyme_binding_proteins()
    n_cores = max(1, os.cpu_count() - 1)
    n_cores = min(n_cores, 32)
    all_goes = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Map the process_protein function across all proteins
        results = executor.map(process_protein, proteins)
        for protein, go_terms in results:
            go_terms = " ".join(go_terms)
            all_goes.append(go_terms)
    with open(GO_FILE, "w") as f:
        f.write("\n".join(all_goes))


def save_go_ancestors():
    with open(GO_FILE) as f:
        go_terms = f.read().splitlines()
    go_terms = [go_term.split() for go_term in go_terms]
    all_goes = set()
    for go_term in go_terms:
        all_goes.update(go_term)
    all_goes = list(all_goes)
    n_cores = max(1, os.cpu_count() - 1)
    n_cores = min(n_cores, 32)
    mapping_lines = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Map the process_protein function across all proteins
        results = executor.map(get_go_ancestors_cached, all_goes)
        for go_term, ancestors in zip(all_goes, results):
            print(f"Processed {go_term}")
            line = f"{go_term}|{' '.join(ancestors)}\n"
            mapping_lines.append(line)
    with open(GO_ANCESTORS_FIRE, "w") as f:
        f.writelines(mapping_lines)


def get_go_matrix():
    proteins = get_human_enzyme_binding_proteins()
    with open(GO_FILE) as f:
        go_terms = f.read().splitlines()
    go_terms = [go_term.split() for go_term in go_terms]

    mapping = {}
    with open(GO_ANCESTORS_FIRE) as f:
        for line in f:
            go_term, ancestors = line.strip().split("|")
            mapping[go_term] = ancestors.split()
    full_go_terms = []
    for go_term in go_terms:
        full_list = set()
        for term in go_term:
            full_list.add(term)
            full_list.update(mapping[term])
        full_go_terms.append(full_list)
    all_goes = set()
    for go_term in full_go_terms:
        all_goes.update(go_term)
    all_goes = list(all_goes)
    go_matrix = pd.DataFrame(index=proteins, columns=all_goes)
    for protein, go_term in zip(proteins, full_go_terms):
        go_matrix.loc[protein, list(go_term)] = 1
    go_matrix.fillna(0, inplace=True)
    return go_matrix


def get_reactome_go_matrix():
    with open("data/reactome/proteins.txt") as f:
        proteins = f.read().splitlines()
    proteins = [protein.split(",")[1] for protein in proteins]
    with open(GO_REACTOME) as f:
        go_terms = f.read().splitlines()
    go_terms = [go_term.split() for go_term in go_terms]

    mapping = {}
    with open(GO_REACTOME_ANCESTORS) as f:
        for line in f:
            go_term, ancestors = line.strip().split("|")
            mapping[go_term] = ancestors.split()
    full_go_terms = []
    for go_term in go_terms:
        full_list = set()
        for term in go_term:
            full_list.add(term)
            full_list.update(mapping[term])
        full_go_terms.append(full_list)
    all_goes = set()
    for go_term in full_go_terms:
        all_goes.update(go_term)
    all_goes = list(all_goes)
    go_matrix = pd.DataFrame(index=proteins, columns=all_goes)
    for protein, go_term in zip(proteins, full_go_terms):
        go_matrix.loc[protein, list(go_term)] = 1
    go_matrix.fillna(0, inplace=True)
    return go_matrix


def save_transferrin():
    transferrin_id = "P02787"
    transferrin_seq = "MRLAVGALLVCAVLGLCLAVPDKTVRWCAVSEHEATKCQSFRDHMKSVIPSDGPSVACVKKASYLDCIRAIAANEADAVTLDAGLVYDAYLAPNNLKPVVAEFYGSKEDPQTFYYAVAVVKKDSGFQMNQLRGKKSCHTGLGRSAGWNIPIGLLYCDLPEPRKPLEKAVANFFSGSCAPCADGTDFPQLCQLCPGCGCSTLNQYFGYSGAFKCLKDGAGDVAFVKHSTIFENLANKADRDQYELLCLDNTRKPVDEYKDCHLAQVPSHTVVARSMGGKEDLIWELLNQAQEHFGKDKSKEFQLFSSPHGKDLLFKDSAHGFLKVPPRMDAKMYLGYEYVTAIRNLREGTCPEAPTDECKPVKWCALSHHERLKCDEWSVNSVGKIECVSAETTEDCIAKIMNGEADAMSLDGGFVYIAGKCGLVPVLAENYNKSDNCEDTPEAGYFAIAVVKKSASDLTWDNLKGKKSCHTAVGRTAGWNIPMGLLYNKINHCRFDEFFSEGCAPGSKKDSSLCKLCMGSGLNLCEPNNKEGYYGYTGAFRCLVEKGDVAFVKHQTVPQNTGGKNPDPWAKNLNEKDYELLCLDGTRKPVEEYANCHLARAPNHAVVTRKDKEACVHKILRQQQHLFGSNVTDCSGNFCLFRSETKDLLFRDDTVCLAKLHDRNTYEKYLGEEYVKAVGNLRKCSTSSLLEACTFRRP"
    seq_to_vec = SeqToVec("esm3-medium")
    transferrin_vec = seq_to_vec.to_vec(transferrin_seq)
    transferrin_vec_as_str = " ".join(map(str, transferrin_vec))
    t_go_terms = list(get_go_terms(transferrin_id))
    t_go_terms_with_ans = sum([get_go_ancestors_cached(go_term) for go_term in t_go_terms], [])
    t_go_terms.extend(t_go_terms_with_ans)
    t_go_terms = list(set(t_go_terms))
    with open(TRANSFERRIN_FILE, "w") as f:
        f.write(transferrin_vec_as_str + "\n")
        f.write(" ".join(t_go_terms))


def load_transferrin():
    with open(TRANSFERRIN_FILE) as f:
        transferrin_vec = list(map(float, f.readline().strip().split()))
        transferrin_vec = np.array(transferrin_vec)
        t_go_terms = f.readline().split()
    return transferrin_vec, t_go_terms


def prep_all():
    if not os.path.exists(IDS_FILE):
        save_human_enzyme_binding_proteins()
    human_enzyme_binding_proteins = get_human_enzyme_binding_proteins()
    if not os.path.exists(SEQ_FILE):
        save_all_sequences(human_enzyme_binding_proteins)
    if not os.path.exists(VEC_FILE):
        save_vecs()
    if not os.path.exists(GO_FILE):
        save_all_go_terms()
    if not os.path.exists(GO_ANCESTORS_FIRE):
        save_go_ancestors()
    if not os.path.exists(TRANSFERRIN_FILE):
        save_transferrin()
    if not os.path.exists(GO_REACTOME):
        save_reactome_go_terms()
    if not os.path.exists(GO_REACTOME_ANCESTORS):
        save_reactome_go_ancestors()


if __name__ == "__main__":
    prep_all()
