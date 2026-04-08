"""
OSmAN-Net: Step 02 - Graph Construction
=========================================
Build temporal window mini-graphs with:
  1. Protocol homophily edges (same protocol type → edge)
  2. Feature similarity edges (top-k cosine similarity)
  3. Prepare for adaptive adjacency learning (in model)

Each window = one graph → graph-level classification
"""

import numpy as np
import os
import pickle
import torch
from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config import (
    PROCESSED_DIR, WINDOW_SIZE, WINDOW_STRIDE,
    K_NEIGHBORS, SEED, CLASSIFICATION_MODE
)

np.random.seed(SEED)
torch.manual_seed(SEED)


def build_protocol_edges(features, protocol_col_idx):
    """
    Build edges between nodes sharing the same protocol type.
    Protocol homophily: same protocol → connected.
    Returns edge_index (2, E) tensor.
    """
    n = len(features)
    protocols = features[:, protocol_col_idx]

    src, dst = [], []
    # Group nodes by protocol
    proto_groups = {}
    for i in range(n):
        p = int(protocols[i]) if not np.isnan(protocols[i]) else -1
        if p not in proto_groups:
            proto_groups[p] = []
        proto_groups[p].append(i)

    # Connect nodes within same protocol group
    for p, nodes in proto_groups.items():
        if len(nodes) > 1:
            # For large groups, limit connections to avoid O(n^2)
            if len(nodes) > 50:
                # Random subset of edges within group
                for node in nodes:
                    neighbors = np.random.choice(
                        nodes, size=min(K_NEIGHBORS, len(nodes)-1), replace=False
                    )
                    for nb in neighbors:
                        if nb != node:
                            src.append(node)
                            dst.append(nb)
            else:
                # Full connections for small groups
                for i, n1 in enumerate(nodes):
                    for n2 in nodes[i+1:]:
                        src.extend([n1, n2])
                        dst.extend([n2, n1])

    if len(src) == 0:
        # Fallback: at least self-loops
        src = list(range(n))
        dst = list(range(n))

    return src, dst


def build_similarity_edges(features, k=K_NEIGHBORS):
    """
    Build top-k cosine similarity edges.
    Each node connects to its k most similar neighbors.
    Returns src, dst lists.
    """
    n = len(features)
    k = min(k, n - 1)

    if n <= 1:
        return [0], [0]

    # Cosine similarity matrix
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, -1)  # exclude self

    src, dst = [], []
    for i in range(n):
        top_k_idx = np.argsort(sim_matrix[i])[-k:]
        for j in top_k_idx:
            if sim_matrix[i, j] > 0:  # only positive similarity
                src.append(i)
                dst.append(j)

    if len(src) == 0:
        src = list(range(n))
        dst = list(range(n))

    return src, dst


def create_window_graph(X_window, y_window, feature_cols, protocol_col_idx):
    """
    Create a PyG Data object from a temporal window.

    Node features: flow features (normalized)
    Edges: protocol homophily + cosine similarity (merged, deduplicated)
    Graph label: majority vote of node labels in window
    """
    n = len(X_window)

    # --- Edge Construction ---
    # Layer 1: Protocol homophily
    src_proto, dst_proto = build_protocol_edges(X_window, protocol_col_idx)

    # Layer 2: Feature similarity (exclude protocol col for similarity)
    feature_mask = np.ones(X_window.shape[1], dtype=bool)
    # Don't use protocol type for similarity (it's categorical)
    feature_mask[protocol_col_idx] = False
    src_sim, dst_sim = build_similarity_edges(X_window[:, feature_mask], k=K_NEIGHBORS)

    # Merge edges (union, deduplicate)
    edge_set = set()
    for s, d in zip(src_proto, dst_proto):
        edge_set.add((s, d))
    for s, d in zip(src_sim, dst_sim):
        edge_set.add((s, d))

    if len(edge_set) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edges = list(edge_set)
        src_all = [e[0] for e in edges]
        dst_all = [e[1] for e in edges]
        edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)

    # --- Node features ---
    x = torch.tensor(X_window, dtype=torch.float)

    # --- Graph-level label ---
    # Majority vote
    label_counts = Counter(y_window)
    graph_label = label_counts.most_common(1)[0][0]

    # Also store per-node labels for node-level analysis
    node_labels = torch.tensor(y_window, dtype=torch.long)

    # Attack ratio in window (useful for analysis)
    if CLASSIFICATION_MODE == "binary":
        attack_ratio = np.mean(y_window > 0)
    else:
        attack_ratio = np.mean(y_window != 0)  # non-benign ratio

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([graph_label], dtype=torch.long),
        node_labels=node_labels,
        num_nodes=n,
        attack_ratio=attack_ratio,
    )

    return data


def build_graphs(X, y, feature_cols, split_name="train"):
    """
    Slide temporal window over data, create mini-graphs.
    """
    n_samples = len(X)
    n_windows = (n_samples - WINDOW_SIZE) // WINDOW_STRIDE + 1

    # Find protocol type column index
    if "Protocol Type" in feature_cols:
        protocol_col_idx = feature_cols.index("Protocol Type")
    else:
        protocol_col_idx = 1  # fallback

    graphs = []
    print(f"  Building {n_windows} graphs from {n_samples:,} flows "
          f"(window={WINDOW_SIZE}, stride={WINDOW_STRIDE})...")

    for i in range(n_windows):
        start = i * WINDOW_STRIDE
        end = start + WINDOW_SIZE

        if end > n_samples:
            break

        X_window = X[start:end]
        y_window = y[start:end]

        graph = create_window_graph(X_window, y_window, feature_cols, protocol_col_idx)
        graphs.append(graph)

        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{n_windows} graphs built...")

    print(f"  Total graphs: {len(graphs)}")
    return graphs


def main():
    print("=" * 60)
    print("OSmAN-Net: Step 02 - Graph Construction")
    print("=" * 60)

    # Load preprocessed data
    print("\n[1/3] Loading preprocessed data...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))

    with open(os.path.join(PROCESSED_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    feature_cols = metadata["feature_cols"]

    # Select label type based on classification mode
    label_suffix = {
        "binary": "binary",
        "coarse": "coarse",
        "fine": "fine"
    }[CLASSIFICATION_MODE]

    y_train = np.load(os.path.join(PROCESSED_DIR, f"y_train_{label_suffix}.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, f"y_test_{label_suffix}.npy"))

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Classification mode: {CLASSIFICATION_MODE}")
    print(f"  Features: {len(feature_cols)}")

    # Build graphs
    print("\n[2/3] Building train graphs...")
    train_graphs = build_graphs(X_train, y_train, feature_cols, "train")

    print("\n[3/3] Building test graphs...")
    test_graphs = build_graphs(X_test, y_test, feature_cols, "test")

    # Graph statistics
    print("\n" + "=" * 60)
    print("Graph Statistics")
    print("=" * 60)

    # Label distribution in graphs
    train_labels = [g.y.item() for g in train_graphs]
    test_labels = [g.y.item() for g in test_graphs]

    print(f"\n  Train graph label distribution:")
    for label, count in sorted(Counter(train_labels).items()):
        print(f"    Class {label}: {count} graphs")

    print(f"\n  Test graph label distribution:")
    for label, count in sorted(Counter(test_labels).items()):
        print(f"    Class {label}: {count} graphs")

    # Edge statistics
    avg_edges_train = np.mean([g.edge_index.shape[1] for g in train_graphs])
    avg_edges_test = np.mean([g.edge_index.shape[1] for g in test_graphs])
    print(f"\n  Avg edges per graph - Train: {avg_edges_train:.1f}, Test: {avg_edges_test:.1f}")

    # Save graphs
    print("\n  Saving graphs...")
    save_path_train = os.path.join(PROCESSED_DIR, "train_graphs.pt")
    save_path_test = os.path.join(PROCESSED_DIR, "test_graphs.pt")

    torch.save(train_graphs, save_path_train)
    torch.save(test_graphs, save_path_test)

    print(f"  Train graphs saved: {save_path_train} ({len(train_graphs)} graphs)")
    print(f"  Test graphs saved: {save_path_test} ({len(test_graphs)} graphs)")
    print("\nDONE!")


if __name__ == "__main__":
    main()
