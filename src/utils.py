"""
Shared utilities for iomt-graph-representation.
"""

import numpy as np
import torch
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler

KNOWN_GATEWAYS = {'10.0.0.254'}
KNOWN_INTERNAL = ('192.168.137.', '10.0.0.')


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_node_features(n_nodes, src_idx, dst_idx, edge_features, ips):
    """
    Compute 12 node features from local topology.
    
    Features: in_degree, out_degree, in_packets, out_packets,
              in_rate, out_rate, degree_ratio, traffic_asymmetry,
              unique_peers, is_internal, in_burstiness, out_burstiness
    """
    nf = np.zeros((n_nodes, 12), dtype=np.float32)
    for i in range(n_nodes):
        out_mask = [j for j, s in enumerate(src_idx) if s == i]
        in_mask = [j for j, d in enumerate(dst_idx) if d == i]
        od, id_ = len(out_mask), len(in_mask)
        op = sum(edge_features[j][0] for j in out_mask) if out_mask else 0
        ip_ = sum(edge_features[j][0] for j in in_mask) if in_mask else 0
        or_ = np.mean([edge_features[j][4] for j in out_mask]) if out_mask else 0
        ir = np.mean([edge_features[j][4] for j in in_mask]) if in_mask else 0
        ob = np.mean([edge_features[j][13] for j in out_mask]) if out_mask else 0
        ib = np.mean([edge_features[j][13] for j in in_mask]) if in_mask else 0
        dr = od / (id_ + 1e-6)
        ta = op / (ip_ + 1e-6)
        pe = od + id_
        ip_str = ips[i] if i < len(ips) else ""
        ii = 1.0 if any(ip_str.startswith(p) for p in KNOWN_INTERNAL) else 0.0
        nf[i] = [id_, od, ip_, op, ir, or_, dr, ta, pe, ii, ib, ob]
    return nf


def pcap_stratified_split(graphs, val_ratio=0.15, seed=42):
    """
    PCAP-level stratified train/val split.
    
    All graphs from the same source PCAP stay in the same split.
    Classes with <=2 PCAPs go entirely to train (no val for them).
    """
    rng = np.random.RandomState(seed)
    pcap_groups = defaultdict(list)
    for i, g in enumerate(graphs):
        pcap_groups[g.source_file].append(i)

    pcap_class = {}
    for pf, indices in pcap_groups.items():
        pcap_class[pf] = Counter([graphs[i].y.item() for i in indices]).most_common(1)[0][0]

    class_pcaps = defaultdict(list)
    for pf, cls in pcap_class.items():
        class_pcaps[cls].append(pf)

    train_idx, val_idx = [], []
    skipped = []

    for cls, pcaps in class_pcaps.items():
        rng.shuffle(pcaps)
        if len(pcaps) <= 2:
            skipped.append(cls)
            for p in pcaps:
                train_idx.extend(pcap_groups[p])
        else:
            nv = max(1, int(len(pcaps) * val_ratio))
            for p in pcaps[:nv]:
                val_idx.extend(pcap_groups[p])
            for p in pcaps[nv:]:
                train_idx.extend(pcap_groups[p])

    if skipped:
        print(f"  Note: classes {skipped} have <=2 PCAPs, no val graphs for them")

    return train_idx, val_idx


def normalize_features(train_g, val_g, test_g):
    """Fit StandardScaler on train, transform all splits."""
    all_x = torch.cat([g.x for g in train_g], 0).numpy()
    all_ea = torch.cat([g.edge_attr for g in train_g], 0).numpy()
    sx = StandardScaler().fit(all_x)
    se = StandardScaler().fit(all_ea)
    for g in train_g + val_g + test_g:
        g.x = torch.nan_to_num(torch.tensor(sx.transform(g.x.numpy()), dtype=torch.float))
        g.edge_attr = torch.nan_to_num(torch.tensor(se.transform(g.edge_attr.numpy()), dtype=torch.float))


def rf_graph_features(graphs):
    """Aggregate node/edge features to graph-level for RF baseline."""
    feats = []
    for g in graphs:
        xn, en = g.x.numpy(), g.edge_attr.numpy()
        n, e = g.num_nodes, g.edge_index.shape[1]
        density = e / (n * (n - 1) + 1e-6)
        feats.append(np.concatenate([
            xn.mean(0), xn.std(0), xn.max(0),
            en.mean(0), en.std(0), en.max(0),
            [n, e, density]
        ]))
    return np.nan_to_num(np.array(feats))
