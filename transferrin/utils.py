import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import pandas as pd
import requests


def get_human_enzyme_binding_proteins():
    with open("transferrin/human_enzyme_binding_proteins.txt") as f:
        proteins = f.read().splitlines()
    return proteins


@lru_cache(maxsize=10000)
def get_go_ancestors_cached(term):
    """Cached function to get GO term ancestors"""
    try:
        url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{term}/ancestors"
        response = requests.get(url, headers={"Accept": "application/json"})

        if response.status_code == 200:
            ancestors = response.json()
            return {ancestor['id'] for ancestor in ancestors['results']}
    except Exception as e:
        print(f"Error fetching ancestors for {term}: {str(e)}")

    return set()


def get_go_terms(uniprot_id):
    """Get GO terms for a UniProt ID with cached ancestry lookup"""
    print(f"Processing {uniprot_id}")
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json"
    response = requests.get(uniprot_url)
    if response.status_code != 200:
        print(f"Error fetching UniProt data: {response.status_code}")
        return set()

    data = response.json()
    go_terms = set()
    direct_terms = set()

    if 'uniProtKBCrossReferences' in data:
        for ref in data['uniProtKBCrossReferences']:
            if ref['database'] == 'GO':
                go_id = ref['id']
                direct_terms.add(go_id)

    go_terms.update(direct_terms)

    # Use cached ancestor lookup
    for term in direct_terms:
        ancestors = get_go_ancestors_cached(term)
        go_terms.update(ancestors)

    return go_terms


def process_protein(protein):
    """Process a single protein and return its GO terms"""
    return (protein, get_go_terms(protein))


def build_go_matrix():
    # Get the list of proteins first
    proteins = get_human_enzyme_binding_proteins()

    # Determine number of CPU cores to use (leave one core free for system)
    n_cores = max(1, os.cpu_count() - 1)

    # Process proteins in parallel
    go_matrix = {}
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Map the process_protein function across all proteins
        results = executor.map(process_protein, proteins)

        # Collect results
        for protein, go_terms in results:
            go_matrix[protein] = go_terms

    # Collect all unique GO terms
    all_go_terms = set()
    for go_terms in go_matrix.values():
        all_go_terms.update(go_terms)

    # Create and fill the DataFrame
    go_df = pd.DataFrame(index=proteins, columns=list(all_go_terms))

    # Fill the matrix
    for protein, go_terms in go_matrix.items():
        go_df.loc[protein, go_terms] = 1

    # Fill missing values with 0
    go_df.fillna(0, inplace=True)

    # Save to CSV
    go_df.to_csv("transferrin/go_human_enzyme_binding_proteins_matrix.csv")


def get_go_matrix():
    return pd.read_csv("transferrin/go_human_enzyme_binding_proteins_matrix.csv", index_col=0)


if __name__ == "__main__":
    build_go_matrix()
