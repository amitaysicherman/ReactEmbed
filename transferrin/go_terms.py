import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import pandas as pd
import requests


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
    p_go_terms = get_go_terms(protein)
    ancestors = set()
    for term in p_go_terms:
        ancestors.update(get_go_ancestors_cached(term))
    ancestors.update(p_go_terms)
    return list(ancestors)


def create_and_save_go_matrix(proteins, output_file_name):
    n_cores = min(max(1, os.cpu_count() - 1), 32)
    all_goes = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = executor.map(process_protein, proteins)
        for go_terms in results:
            all_goes.append(go_terms)
    all_unique_goes = set()
    for terms in all_goes:
        all_unique_goes.update(terms)
    all_unique_goes = list(all_unique_goes)
    go_matrix = pd.DataFrame(0, index=proteins, columns=all_unique_goes)
    for i, terms in enumerate(all_goes):
        go_matrix.loc[proteins[i], terms] = 1
    go_matrix.to_csv(output_file_name)
