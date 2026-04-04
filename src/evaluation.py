"""
Evaluation module: training with proper PCAP-level validation protocol.

Supports three experiment modes:
  - final_clean: 6-class, proper protocol, 5 seeds, all models
  - diagnostic: domain edges (A), binary (B), node-role (C)
  - motif_4class: 4-class topology-heavy subset
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import pickle
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .utils import (set_seed, build_node_features, pcap_stratified_split,
                    normalize_features, rf_graph_features)
from .models import AdaptiveGAT, PureGAT, GraphSAGEModel
from .domain_edges import build_graphs_with_typed_edges


def build_standard_graphs(raw_graphs, target_classes=None):
    """Build standard PyG graphs (communication edges only, 14 features)."""
    pyg = []
    for g in raw_graphs:
        label = g['label']
        if target_classes and label not in target_classes:
            continue
        n = g['n_nodes']
        si, di = g['src_idx'], g['dst_idx']
        ef = g['edge_features']
        if len(si) < 2 or n < 2:
            continue
        nf = build_node_features(n, si, di, ef, g['ips'])
        data = Data(
            x=torch.tensor(nf, dtype=torch.float),
            edge_index=torch.tensor([si, di], dtype=torch.long),
            edge_attr=torch.tensor(ef, dtype=torch.float),
            y=torch.tensor([0], dtype=torch.long),
            num_nodes=n,
            label_str=label,
            source_file=g['source_file'],
        )
        pyg.append(data)
    return pyg


def train_with_val(model, train_g, val_g, test_g, num_classes, device,
                   epochs=300, patience=40, lr=5e-4):
    """
    Train GNN with validation-based early stopping.
    Returns test metrics from best validation checkpoint.
    """
    model = model.to(device)
    labels = [g.y.item() for g in train_g]
    counts = Counter(labels)
    total = len(labels)
    cw = torch.tensor([total / (num_classes * counts.get(i, 1))
                       for i in range(num_classes)], dtype=torch.float).to(device)
    ce = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    trl = DataLoader(train_g, batch_size=32, shuffle=True)
    vll = DataLoader(val_g, batch_size=32, shuffle=False)
    tel = DataLoader(test_g, batch_size=32, shuffle=False)

    best_vf1, best_state, pc = 0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        for b in trl:
            b = b.to(device); opt.zero_grad()
            logits = model(b)
            ce(logits, b.y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for b in vll:
                b = b.to(device)
                vp.extend(model(b).argmax(-1).cpu().numpy())
                vl.extend(b.y.cpu().numpy())
        vf1 = f1_score(vl, vp, average='macro', zero_division=0)
        if vf1 > best_vf1:
            best_vf1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
        if pc >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    tp, tl = [], []
    with torch.no_grad():
        for b in tel:
            b = b.to(device)
            tp.extend(model(b).argmax(-1).cpu().numpy())
            tl.extend(b.y.cpu().numpy())
    tp, tl = np.array(tp), np.array(tl)

    return {
        'accuracy': float(accuracy_score(tl, tp)),
        'f1_macro': float(f1_score(tl, tp, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(tl, tp, average='weighted', zero_division=0)),
        'precision': float(precision_score(tl, tp, average='macro', zero_division=0)),
        'recall': float(recall_score(tl, tp, average='macro', zero_division=0)),
        'per_class_f1': f1_score(tl, tp, average=None, zero_division=0).tolist(),
        'best_val_f1': float(best_vf1),
    }


def run_final_clean(graph_dir, results_dir, seeds, device):
    """Main 6-class experiment: RF, GraphSAGE, PureGAT, AdaptiveGAT."""
    print("=" * 60)
    print("Final Clean: 6-Class, Proper Protocol")
    print("=" * 60)

    with open(os.path.join(graph_dir, "train_graphs.pkl"), 'rb') as f:
        raw_train = pickle.load(f)
    with open(os.path.join(graph_dir, "test_graphs.pkl"), 'rb') as f:
        raw_test = pickle.load(f)

    all_results = {'RF': [], 'GraphSAGE': [], 'PureGAT': [], 'AdaptiveGAT': []}

    for si, seed in enumerate(seeds):
        set_seed(seed)
        print(f"\nSeed {seed} ({si+1}/{len(seeds)})")

        all_train = build_standard_graphs(raw_train)
        test_g = build_standard_graphs(raw_test)

        le = LabelEncoder()
        le.fit([g.label_str for g in all_train + test_g])
        for g in all_train + test_g:
            g.y = torch.tensor([le.transform([g.label_str])[0]], dtype=torch.long)

        tri, vai = pcap_stratified_split(all_train, seed=seed)
        tr = [all_train[i] for i in tri]
        va = [all_train[i] for i in vai]
        normalize_features(tr, va, test_g)

        nd, ed = tr[0].x.shape[1], tr[0].edge_attr.shape[1]
        nc = len(le.classes_)

        # RF
        Xtr = rf_graph_features(tr)
        ytr = np.array([g.y.item() for g in tr])
        Xte = rf_graph_features(test_g)
        yte = np.array([g.y.item() for g in test_g])
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    n_jobs=16, random_state=seed)
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xte)
        all_results['RF'].append({
            'accuracy': float(accuracy_score(yte, yp)),
            'f1_macro': float(f1_score(yte, yp, average='macro', zero_division=0)),
            'per_class_f1': f1_score(yte, yp, average=None, zero_division=0).tolist(),
        })

        # GNN models
        for name, ModelClass in [('GraphSAGE', GraphSAGEModel),
                                  ('PureGAT', PureGAT),
                                  ('AdaptiveGAT', AdaptiveGAT)]:
            set_seed(seed)
            model = ModelClass(nd, ed, 128, nc)
            res = train_with_val(model, tr, va, test_g, nc, device)
            all_results[name].append(res)
            print(f"  {name}: F1m={res['f1_macro']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    for m in all_results:
        vals = [r['f1_macro'] for r in all_results[m]]
        print(f"{m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    with open(os.path.join(results_dir, 'final_clean_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"Saved to {results_dir}/final_clean_results.json")


def run_4class_topology(graph_dir, results_dir, seeds, device):
    """4-class topology-heavy subset with domain-typed edges."""
    print("=" * 60)
    print("4-Class Topology-Heavy: Domain-Typed Edges")
    print("=" * 60)

    target = ['Benign', 'DDoS', 'DoS', 'Recon']

    with open(os.path.join(graph_dir, "train_graphs.pkl"), 'rb') as f:
        raw_train = pickle.load(f)
    with open(os.path.join(graph_dir, "test_graphs.pkl"), 'rb') as f:
        raw_test = pickle.load(f)

    all_results = {'RF': [], 'AdaptiveGAT': []}

    for si, seed in enumerate(seeds):
        set_seed(seed)
        print(f"\nSeed {seed} ({si+1}/{len(seeds)})")

        all_train = build_graphs_with_typed_edges(raw_train, target)
        test_g = build_graphs_with_typed_edges(raw_test, target)

        le = LabelEncoder()
        le.fit(target)
        for g in all_train + test_g:
            g.y = torch.tensor([le.transform([g.label_str])[0]], dtype=torch.long)

        tri, vai = pcap_stratified_split(all_train, seed=seed)
        tr = [all_train[i] for i in tri]
        va = [all_train[i] for i in vai]
        normalize_features(tr, va, test_g)

        nd, ed = tr[0].x.shape[1], tr[0].edge_attr.shape[1]
        nc = len(target)

        # RF
        Xtr = rf_graph_features(tr)
        ytr = np.array([g.y.item() for g in tr])
        Xte = rf_graph_features(test_g)
        yte = np.array([g.y.item() for g in test_g])
        rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                    n_jobs=16, random_state=seed)
        rf.fit(Xtr, ytr)
        yp = rf.predict(Xte)
        all_results['RF'].append({
            'f1_macro': float(f1_score(yte, yp, average='macro', zero_division=0)),
            'per_class_f1': f1_score(yte, yp, average=None, zero_division=0).tolist(),
        })

        # AdaptiveGAT
        set_seed(seed)
        model = AdaptiveGAT(nd, ed, 128, nc)
        res = train_with_val(model, tr, va, test_g, nc, device)
        all_results['AdaptiveGAT'].append(res)
        print(f"  RF: {all_results['RF'][-1]['f1_macro']:.4f}  GAT: {res['f1_macro']:.4f}")

    print("\n" + "=" * 60)
    for m in all_results:
        vals = [r['f1_macro'] for r in all_results[m]]
        print(f"{m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    with open(os.path.join(results_dir, 'topology_4class_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=float)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='final_clean',
                        choices=['final_clean', 'motif_4class'])
    parser.add_argument('--graph_dir', required=True)
    parser.add_argument('--results_dir', required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [42, 123, 456, 789, 2026]
    os.makedirs(args.results_dir, exist_ok=True)

    if args.experiment == 'final_clean':
        run_final_clean(args.graph_dir, args.results_dir, seeds, device)
    elif args.experiment == 'motif_4class':
        run_4class_topology(args.graph_dir, args.results_dir, seeds, device)
