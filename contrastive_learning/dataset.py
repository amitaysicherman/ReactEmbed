import random
from os.path import join as pjoin

import torch
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

# TYPES = ["P-P", "P-M", "M-P", "M-M"]

TYPES = ["M", "P"]

PAIR_TYPES = ["P-P", "P-M", "M-P", "M-M"]
TRIPLET_TYPES = ["P-P-M", "P-M-P", "M-P-P", "M-M-P", "P-P-P", "M-M-M", "M-P-M", "P-M-M"]


def print_hist_as_csv(counts):
    counts = list(counts)
    print(np.mean(counts))
    print(np.median(counts))
    print(np.std(counts))
    print(np.quantile(counts, 0.25), np.quantile(counts, 0.75))
    counts = np.clip(counts, np.quantile(counts, 0.5), np.quantile(counts, 0.95))
    hist, bins = np.histogram(counts, bins=10)
    print("bin, count")
    for i, c in enumerate(hist):
        print(bins[i], bins[i + 1], c, sep=",")


def prep_entity(entities, empty_list):
    if entities == "" or entities == " ":
        return []
    else:
        return [int(x) for x in entities.split(",") if int(x) not in empty_list]


class TripletsDataset(Dataset):
    def __init__(self, data_name, split, p_model="ProtBert", m_model="ChemBERTa", n_duplicates=10, flip_prob=0,
                 samples_ratio=1, no_pp_mm=0):
        self.split = split

        self.flip_prob = flip_prob
        self.item_path = f"data/{data_name}"
        reactions_file = pjoin(self.item_path, "reaction.txt")
        self.proteins = np.load(pjoin(self.item_path, f"{p_model}_vectors.npy"))
        self.molecules = np.load(pjoin(self.item_path, f"{m_model}_vectors.npy"))
        self.empty_protein_index = set(np.where(np.all(self.proteins == 0, axis=1))[0].tolist())
        self.empty_molecule_index = set(np.where(np.all(self.molecules == 0, axis=1))[0].tolist())
        self.proteins_non_empty = [i for i in range(len(self.proteins)) if i not in self.empty_protein_index]
        self.molecules_non_empty = [i for i in range(len(self.molecules)) if i not in self.empty_molecule_index]
        print(f"Empty proteins: {len(self.empty_protein_index)}")
        print(f"Empty molecules: {len(self.empty_molecule_index)}")
        print("Not empty proteins:", len(self.proteins_non_empty))
        print("Not empty molecules:", len(self.molecules_non_empty))
        self.types = PAIR_TYPES
        self.pair_counts = {t: Counter() for t in self.types}
        # Count pair frequencies
        with open(reactions_file) as f:
            lines = f.read().splitlines()
        for line in tqdm(lines):
            if line.startswith(" "):
                proteins, molecules = "", line
            elif line.endswith(" "):
                proteins, molecules = line, ""
            else:
                proteins, molecules = line.split()
            proteins = prep_entity(proteins, self.empty_protein_index)
            molecules = prep_entity(molecules, self.empty_molecule_index)

            types = ["P"] * len(proteins) + ["M"] * len(molecules)
            elements = proteins + molecules
            for i, e1 in enumerate(elements):
                for j, e2 in enumerate(elements[i + 1:], start=i + 1):
                    if no_pp_mm == 1 and types[i] == types[j]:
                        continue
                    self.pair_counts[f"{types[i]}-{types[j]}"][(e1, e2)] += 1
                    self.pair_counts[f"{types[j]}-{types[i]}"][(e2, e1)] += 1


        # Split the valid pairs
        self.split_pair = {}
        for t in self.types:
            t_pairs = list(self.pair_counts[t].keys())
            t_pairs.sort()
            random.seed(42)
            random.shuffle(t_pairs)
            if self.split == "all":
                self.split_pair[t] = t_pairs
            elif self.split == "train":
                self.split_pair[t] = t_pairs[:int(len(t_pairs) * 0.8)]
            elif self.split == "valid":
                self.split_pair[t] = t_pairs[int(len(t_pairs) * 0.8):int(len(t_pairs) * 0.9)]
            elif self.split == "test":
                self.split_pair[t] = t_pairs[int(len(t_pairs) * 0.9):]
            else:
                raise ValueError("Unknown split")

        self.triples = {t: set() for t in TRIPLET_TYPES}
        for t in self.types:
            t1, t2 = t.split("-")
            ttag = "P" if t2 == "M" else "M"
            for e1, e2 in tqdm(self.split_pair[t], desc=f"Generating {t} triplets"):
                if samples_ratio < 1 and random.random() > samples_ratio:
                    continue
                for _ in range(n_duplicates):
                    pair_count = self.pair_counts[t][(e1, e2)]
                    pair_count = min(pair_count, 10)
                    for _ in range(pair_count):
                        e3_a = self.sample_neg_element(e1, t1, t2)
                        e3_b = self.sample_neg_element(e1, t1, ttag)

                        if self.flip_prob > 0 and random.random() < self.flip_prob:
                            self.triples[f"{t1}-{t2}-{t2}"].add((e1, e3_a, e2))
                            self.triples[f"{t1}-{ttag}-{t2}"].add((e1, e3_b, e2))
                        else:
                            self.triples[f"{t1}-{t2}-{t2}"].add((e1, e2, e3_a))
                            self.triples[f"{t1}-{t2}-{ttag}"].add((e1, e2, e3_b))
        self.triples = {t: list(self.triples[t]) for t in TRIPLET_TYPES}
        # shuffle the triples
        for t in self.triples:
            random.seed(42)
            random.shuffle(self.triples[t])
            print(f"Number of {t} triples: {len(self.triples[t])}")

    def sample_neg_element(self, e1, e1_type, e2_type):
        """Sample negative element that has never appeared with e1"""
        pair_type = f"{e1_type}-{e2_type}"
        while True:
            if e2_type == "P":
                e3 = random.choice(self.proteins_non_empty)
            else:
                e3 = random.choice(self.molecules_non_empty)
            if (e1, e3) not in self.pair_counts[pair_type]:
                return e3

    def __len__(self):
        return sum(len(self.triples[t]) for t in self.triples)

    def type_to_start_index(self, t):
        return sum(len(self.triples[x]) for x in TRIPLET_TYPES[:self.types.index(t)])

    def idx_type_to_vec(self, idx, t):
        if t == "P":
            return torch.tensor(self.proteins[idx]).float()
        return torch.tensor(self.molecules[idx]).float()

    def __getitem__(self, t_idx):
        t, idx = t_idx
        e1, e2, e3 = self.triples[t][idx]
        t1, t2, t3 = t.split("-")
        v1, v2, v3 = self.idx_type_to_vec(e1, t1), self.idx_type_to_vec(e2, t2), self.idx_type_to_vec(e3, t3)
        return t1, t2, t3, v1, v2, v3


class TripletsBatchSampler(Sampler):
    def __init__(self, dataset: TripletsDataset, batch_size, max_num_steps=5_000):
        self.dataset = dataset
        self.batch_size = batch_size
        max_len = max(len(self.dataset.triples[t]) for t in self.dataset.triples)
        print(f"Max length: {max_len}")
        # self.types_upsample = {t: max_len // len(self.dataset.triples[t]) for t in self.dataset.triples}
        self.max_num_steps = max_num_steps

    def __iter__(self):
        for _ in range(self.max_num_steps):
            t = random.choice(TRIPLET_TYPES)
            if len(self.dataset.triples[t]) == 0:
                continue
            if len(self.dataset.triples[t]) < self.batch_size:
                yield [(t, i) for i in range(len(self.dataset.triples[t]))]
            else:
                idx = random.choice(range(0, len(self.dataset.triples[t]) - self.batch_size - 1))
                yield [(t, idx + i) for i in range(self.batch_size)]

    def __len__(self):
        return self.max_num_steps


import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, Tuple


def build_heterogeneous_graph(pair_counts):
    """
    Build a heterogeneous graph from pair counts

    Args:
        pair_counts (dict): Dictionary of Counters containing edge information
            Format: {edge_type: Counter((node1, node2): count)}

    Returns:
        nx.MultiGraph: Heterogeneous graph with protein and molecule nodes
    """
    G = nx.MultiGraph()

    for edge_type, pairs in pair_counts.items():
        source_type, target_type = edge_type.split("-")

        for (source, target), weight in pairs.items():
            if not G.has_node(source):
                G.add_node(source, node_type=source_type)
            if not G.has_node(target):
                G.add_node(target, node_type=target_type)

            G.add_edge(source,
                       target,
                       weight=weight,
                       edge_type=edge_type)

    return G


def analyze_edge_weights(G) -> Dict[str, Dict[str, float]]:
    """Analyze edge weights by type"""
    weight_stats = defaultdict(lambda: {"count": 0, "total": 0, "min": float('inf'),
                                        "max": 0, "mean": 0, "median": 0, "std": 0})

    for _, _, attrs in G.edges(data=True):
        edge_type = attrs['edge_type']
        weight = attrs['weight']
        stats = weight_stats[edge_type]
        stats["count"] += 1
        stats["total"] += weight
        stats["min"] = min(stats["min"], weight)
        stats["max"] = max(stats["max"], weight)

        # Collect weights for statistical analysis
        if "weights" not in stats:
            stats["weights"] = []
        stats["weights"].append(weight)

    # Calculate statistics
    for edge_type, stats in weight_stats.items():
        weights = stats.pop("weights")  # Remove raw weights after calculation
        stats["mean"] = np.mean(weights)
        stats["median"] = np.median(weights)
        stats["std"] = np.std(weights)
        stats["quartiles"] = np.percentile(weights, [25, 75]).tolist()

    return dict(weight_stats)


def analyze_node_connectivity(G) -> Dict[str, Dict[str, float]]:
    """Analyze node connectivity by type"""
    node_stats = defaultdict(lambda: {"count": 0, "avg_degree": 0, "max_degree": 0,
                                      "min_degree": float('inf'), "density": 0})

    # Calculate degree statistics by node type
    for node, attrs in G.nodes(data=True):
        node_type = attrs['node_type']
        degree = G.degree(node)
        stats = node_stats[node_type]
        stats["count"] += 1
        stats["max_degree"] = max(stats["max_degree"], degree)
        stats["min_degree"] = min(stats["min_degree"], degree)
        if "degrees" not in stats:
            stats["degrees"] = []
        stats["degrees"].append(degree)

    # Calculate averages and clean up
    for node_type, stats in node_stats.items():
        degrees = stats.pop("degrees")  # Remove raw degrees after calculation
        stats["avg_degree"] = np.mean(degrees)
        stats["degree_std"] = np.std(degrees)
        stats["degree_quartiles"] = np.percentile(degrees, [25, 75]).tolist()

    return dict(node_stats)


def calculate_centrality_metrics(G) -> Tuple[Dict, Dict, Dict]:
    """Calculate various centrality metrics"""
    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    # Betweenness centrality (can be slow for large graphs)
    between_cent = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    # Eigenvector centrality
    eigen_cent = nx.eigenvector_centrality_numpy(G, weight='weight')

    return degree_cent, between_cent, eigen_cent


def analyze_clustering(G) -> Dict[str, float]:
    """Analyze clustering coefficients"""
    clustering_stats = {
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
    }

    # Calculate clustering by node type
    clustering_by_type = defaultdict(list)
    node_clustering = nx.clustering(G)

    for node, coef in node_clustering.items():
        node_type = G.nodes[node]['node_type']
        clustering_by_type[node_type].append(coef)

    for node_type, coeffs in clustering_by_type.items():
        clustering_stats[f"avg_clustering_{node_type}"] = np.mean(coeffs)

    return clustering_stats


def analyze_graph(G):
    """
    Comprehensive graph analysis including edge weights, connectivity, centrality, and clustering

    Args:
        G (nx.MultiGraph): The heterogeneous graph
    """
    print("=== Basic Graph Statistics ===")
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    # Node type distribution
    node_types = Counter(nx.get_node_attributes(G, 'node_type').values())
    print("\n=== Node Type Distribution ===")
    for ntype, count in node_types.items():
        print(f"{ntype}: {count}")

    # Edge weight analysis
    print("\n=== Edge Weight Analysis ===")
    weight_stats = analyze_edge_weights(G)
    for edge_type, stats in weight_stats.items():
        print(f"\n{edge_type} edges:")
        print(f"Count: {stats['count']}")
        print(f"Total weight: {stats['total']}")
        print(f"Weight range: {stats['min']} - {stats['max']}")
        print(f"Mean weight: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"Median weight: {stats['median']:.2f}")
        print(f"Weight quartiles (25%, 75%): {stats['quartiles']}")

    # Connectivity analysis
    print("\n=== Node Connectivity Analysis ===")
    node_stats = analyze_node_connectivity(G)
    for node_type, stats in node_stats.items():
        print(f"\n{node_type} nodes:")
        print(f"Count: {stats['count']}")
        print(f"Degree range: {stats['min_degree']} - {stats['max_degree']}")
        print(f"Average degree: {stats['avg_degree']:.2f} ± {stats['degree_std']:.2f}")
        print(f"Degree quartiles (25%, 75%): {stats['degree_quartiles']}")

    # Component analysis
    components = list(nx.connected_components(G))
    print(f"\n=== Component Analysis ===")
    print(f"Number of connected components: {len(components)}")
    print(f"Largest component size: {len(max(components, key=len))}")

    # Clustering analysis
    print("\n=== Clustering Analysis ===")
    try:
        clustering_stats = analyze_clustering(G)
        print(f"Average clustering coefficient: {clustering_stats['avg_clustering']:.4f}")
        print(f"Transitivity: {clustering_stats['transitivity']:.4f}")
        for metric, value in clustering_stats.items():
            if metric.startswith('avg_clustering_'):
                node_type = metric.split('_')[-1]
                print(f"Average clustering for {node_type} nodes: {value:.4f}")
    except Exception as e:
        print(f"Error calculating clustering metrics: {e}")


class HeterogeneousGraphBuilder:
    def __init__(self, dataset):
        """
        Initialize graph builder with a TripletsDataset

        Args:
            dataset (TripletsDataset): Dataset containing pair counts
        """
        self.dataset = dataset
        self.graph = None

    def build(self):
        """Build the heterogeneous graph from the dataset's pair counts"""
        self.graph = build_heterogeneous_graph(self.dataset.pair_counts)
        return self.graph

    def analyze(self):
        """Analyze the built graph"""
        if self.graph is None:
            raise ValueError("Graph hasn't been built yet. Call build() first.")
        analyze_graph(self.graph)

    def get_node_vectors(self, node):
        """Get the feature vector for a node based on its type"""
        node_type = self.graph.nodes[node]['node_type']
        if node_type == 'P':
            return self.dataset.proteins[node]
        else:  # node_type == 'M'
            return self.dataset.molecules[node]

    def get_edge_subgraph(self, edge_type):
        """Extract a subgraph containing only edges of a specific type"""
        edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                 if d['edge_type'] == edge_type]
        return nx.Graph(self.graph.edge_subgraph(edges))

if __name__ == "__main__":
    dataset = TripletsDataset("reactome", "all", p_model="ProtBert", m_model="ChemBERTa", n_duplicates=1)
    builder = HeterogeneousGraphBuilder(dataset)
    G = builder.build()
    builder.analyze()
