import pandas as pd
import requests


def get_human_enzyme_binding_proteins():
    with open("transferrin/human_enzyme_binding_proteins.txt") as f:
        proteins = f.read().splitlines()
    return proteins


def get_go_terms(uniprot_id):
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=json"
    response = requests.get(uniprot_url)
    if response.status_code != 200:
        print(f"Error fetching UniProt data: {response.status_code}")
    data = response.json()
    go_terms = set()
    direct_terms = set()
    if 'uniProtKBCrossReferences' in data:
        for ref in data['uniProtKBCrossReferences']:
            if ref['database'] == 'GO':
                go_id = ref['id']
                direct_terms.add(go_id)

    go_terms.update(direct_terms)
    for term in direct_terms:
        try:
            print(term)
            # QuickGO API endpoint for ancestry
            url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{term}/ancestors"
            response = requests.get(url, headers={"Accept": "application/json"})

            if response.status_code == 200:
                ancestors = response.json()
                for ancestor in ancestors['results']:
                    go_terms.add(ancestor['id'])
        except Exception as e:
            print(f"Error fetching ancestors for {term}: {str(e)}")
            continue
    return go_terms


def build_go_matrix():
    proteins = get_human_enzyme_binding_proteins()
    go_matrix = {}
    for protein in proteins:
        go_matrix[protein] = get_go_terms(protein)
    all_go_terms = set()
    for go_terms in go_matrix.values():
        all_go_terms.update(go_terms)
    go_df = pd.DataFrame(index=proteins, columns=list(all_go_terms))
    for protein, go_terms in go_matrix.items():
        go_df.loc[protein, go_terms] = 1
    go_df.fillna(0, inplace=True)
    go_df.to_csv("transferrin/go_human_enzyme_binding_proteins_matrix.csv")


def get_go_matrix():
    return pd.read_csv("transferrin/go_human_enzyme_binding_proteins_matrix.csv", index_col=0)


if __name__ == "__main__":
    build_go_matrix()
