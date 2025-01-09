import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
# import tsne
import torch
from sklearn.decomposition import PCA


class ReactionVisualizer:
    def __init__(self, max_triples=50):
        self.max_triples = max_triples
        self.sampled_data = None

    def collect_entity_triples(self, dataset):
        """
        Collect all triples where a randomly selected entity appears as anchor
        """
        # Collect all unique entities that appear as anchors
        random.seed(42)
        random_type = random.choice(list(dataset.triples.keys()))
        anchor_entity = random.choice(dataset.triples[random_type])[0]
        entity_triples = defaultdict(list)

        for triplet_type in dataset.triples:
            t1, _, _ = triplet_type.split("-")
            for e1, e2, e3 in dataset.triples[triplet_type]:
                if e1 != anchor_entity:
                    continue
                entity_triples[triplet_type].append((e2, e3))
        for triplet_type in entity_triples:
            entity_triples[triplet_type] = random.sample(entity_triples[triplet_type],
                                                         min(len(entity_triples[triplet_type]), self.max_triples))
        entity_triples_list = []
        for e in entity_triples:
            for i in range(len(entity_triples[e])):
                entity_triples_list.append((e, *entity_triples[e][i]))
        return anchor_entity, entity_triples_list

    def sample_data(self, dataset):
        """Sample data for visualization - collect triples for both a protein and molecule anchor"""
        self.dataset = dataset
        self.sampled_data = self.collect_entity_triples(dataset)

    def get_embeddings(self, model, device):
        """Get embeddings for all sampled entities"""
        embeddings = []
        labels = []
        markers = []
        colors = []
        anchor_entity, triples = self.sampled_data
        anchor_type = triples[0][0].split("-")[0]
        # Get anchor embedding
        anchor_vec = self.dataset.idx_type_to_vec(anchor_entity, anchor_type).to(device)
        with torch.no_grad():
            anchor_emb = model(anchor_vec.unsqueeze(0), anchor_type)[0].cpu().numpy()

        # Add anchor to plots
        embeddings.append(anchor_emb)
        labels.append(f"Anchor({anchor_type})")
        markers.append('X' if anchor_type == 'P' else 'X')
        colors.append('blue')
        marker_to_name = {"X": "Protein", "s": "Molecule"}
        color_to_name = {"blue": "Anchor", "green": "Positive", "red": "Negative"}
        # Process each triple
        for triplet_type, e2, e3 in triples:
            t1, t2, t3 = triplet_type.split("-")

            # Get positive and negative embeddings
            v2 = self.dataset.idx_type_to_vec(e2, t2).to(device)
            v3 = self.dataset.idx_type_to_vec(e3, t3).to(device)

            with torch.no_grad():
                e2_emb = model(v2.unsqueeze(0), t2)[0].cpu().numpy()
                e3_emb = model(v3.unsqueeze(0), t3)[0].cpu().numpy()

            # Add positive sample
            embeddings.append(e2_emb)
            markers.append('X' if t2 == 'P' else 's')
            colors.append('green')
            labels.append(f"{marker_to_name[markers[-1]]}({color_to_name[colors[-1]]})")

            # Add negative sample
            embeddings.append(e3_emb)
            markers.append('X' if t3 == 'P' else 's')
            colors.append('red')
            labels.append(f"{marker_to_name[markers[-1]]}({color_to_name[colors[-1]]})")

        return np.array(embeddings), labels, markers, colors

    def visualize_epoch(self, model, device, epoch, save_dir):
        """Create visualization for current epoch"""
        embeddings, labels, markers, colors = self.get_embeddings(model, device)

        if len(embeddings) == 0:
            print("No valid embeddings to visualize")
            return

        # Use PCA for dimensionality reduction
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        # Use t-SNE for dimensionality reduction
        # tsne = TSNE(n_components=2, perplexity=10, n_iter=300)
        # embeddings_2d = tsne.fit_transform(embeddings)
        # Create separate plots for protein anchor and molecule anchor
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        seen_labels = set()
        for i in range(len(embeddings)):
            if labels[i] in seen_labels:
                label_args = {}
            else:
                seen_labels.add(labels[i])
                label_args = {"label": labels[i]}

            ax1.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                        c=colors[i], marker=markers[i], s=200 if labels[i].startswith("Anchor") else 50,
                        alpha=1 if labels[i].startswith("Anchor") else 0.5,
                        **label_args)

        ax1.set_title(f'Anchor Triples\nEpoch {epoch}')
        ax1.legend()
        ax1.grid(True)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/epoch_{epoch}_embeddings.png",
                    bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


_visualizer = None  # Global instance to maintain same samples across epochs


def visualize_training(model, dataset, device, epoch, save_dir):
    """Main function to call for visualization"""
    global _visualizer

    if _visualizer is None:
        _visualizer = ReactionVisualizer(max_triples=20)
        _visualizer.sample_data(dataset)

    # Create visualization for current epoch
    _visualizer.visualize_epoch(model, device, epoch, save_dir)
