"""
Domain-typed edge augmentation for communication-topology graphs.

Edge types:
  0 = communication (original IP-to-IP flow)
  1 = same-subnet (internal IPs sharing /24, no existing direct edge)
  2 = gateway-linked (one endpoint is known gateway)

Non-communication edges receive scaled proxy features (10% of mean
communication edge features) to keep them informative but weaker
than observed traffic edges.
"""

import numpy as np
import torch
from collections import defaultdict
from torch_geometric.data import Data
from .utils import build_node_features, KNOWN_GATEWAYS, KNOWN_INTERNAL


def get_subnet(ip):
    parts = ip.split('.')
    return '.'.join(parts[:3]) if len(parts) == 4 else ''


def build_graphs_with_typed_edges(raw_graphs, target_classes=None, proxy_scale=0.1):
    """
    Build PyG graphs with domain-typed edges from raw graph dicts.
    
    Args:
        raw_graphs: list of dicts with keys n_nodes, src_idx, dst_idx,
                    edge_features, ips, label, source_file
        target_classes: optional list of class names to include (None = all)
        proxy_scale: scale factor for structural edge features
    
    Returns:
        list of PyG Data objects
    """
    pyg = []
    for g in raw_graphs:
        label = g['label']
        if target_classes and label not in target_classes:
            continue

        n = g['n_nodes']
        si, di = g['src_idx'], g['dst_idx']
        ef = np.array(g['edge_features'], dtype=np.float32)
        ips = g['ips']

        if len(si) < 2 or n < 2:
            continue

        nf = build_node_features(n, si, di, g['edge_features'], ips)

        # Original communication edges (type 0)
        all_si = list(si)
        all_di = list(di)
        all_types = [0] * len(si)
        existing = set(zip(si, di))

        # Proxy features for structural edges
        mean_ef = ef.mean(axis=0) if len(ef) > 0 else np.zeros(14, dtype=np.float32)
        proxy_ef = mean_ef * proxy_scale

        # Same-subnet edges (type 1)
        subnet_groups = defaultdict(list)
        for i, ip in enumerate(ips):
            if any(ip.startswith(p) for p in KNOWN_INTERNAL):
                subnet_groups[get_subnet(ip)].append(i)

        for subnet, nodes in subnet_groups.items():
            if len(nodes) < 2:
                continue
            added = 0
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if added >= len(nodes) * 2:
                        break
                    ni, nj = nodes[i], nodes[j]
                    if (ni, nj) not in existing and (nj, ni) not in existing:
                        all_si.append(ni); all_di.append(nj); all_types.append(1)
                        all_si.append(nj); all_di.append(ni); all_types.append(1)
                        existing.add((ni, nj)); existing.add((nj, ni))
                        added += 2

        # Gateway-linked edges (type 2)
        gw_indices = [i for i, ip in enumerate(ips) if ip in KNOWN_GATEWAYS]
        for gw in gw_indices:
            for i in range(n):
                if i == gw:
                    continue
                ip = ips[i] if i < len(ips) else ""
                if any(ip.startswith(p) for p in KNOWN_INTERNAL):
                    if (gw, i) not in existing:
                        all_si.append(gw); all_di.append(i); all_types.append(2)
                        all_si.append(i); all_di.append(gw); all_types.append(2)
                        existing.add((gw, i)); existing.add((i, gw))

        # Build edge features: original 14 + 3-dim type one-hot = 17
        n_edges = len(all_si)
        edge_feats = np.zeros((n_edges, 14), dtype=np.float32)
        for j in range(len(si)):
            edge_feats[j] = ef[j]
        for j in range(len(si), n_edges):
            edge_feats[j] = proxy_ef

        type_oh = np.zeros((n_edges, 3), dtype=np.float32)
        for j, t in enumerate(all_types):
            type_oh[j, t] = 1.0

        final_ef = np.concatenate([edge_feats, type_oh], axis=1)

        data = Data(
            x=torch.tensor(nf, dtype=torch.float),
            edge_index=torch.tensor([all_si, all_di], dtype=torch.long),
            edge_attr=torch.tensor(final_ef, dtype=torch.float),
            y=torch.tensor([0], dtype=torch.long),
            num_nodes=n,
            label_str=label,
            source_file=g['source_file'],
        )
        pyg.append(data)
    return pyg
