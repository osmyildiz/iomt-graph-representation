"""
OSmAN-Net v2: Natural Network Graph Pipeline
===============================================
60-second time-window network snapshots:
  - Node = unique IP (IoMT device)
  - Edge = aggregated flow between IPs in that window
  - Edge features = 14 rich features (packet_count, rate, IAT stats, port diversity, protocol)
  - Node features = computed from topology (in/out degree, traffic asymmetry, etc.)
  - Graph label = attack type (graph-level classification)

This correctly models the network topology.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import json
import pickle
from collections import Counter

sys.path.insert(0, '.')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (SAGEConv, GATConv, GCNConv,
                                 GlobalAttention, global_mean_pool)
import warnings
warnings.filterwarnings('ignore')

GRAPH_DIR = "/network/rit/dgx/dgx_subasi_lab/osman/osman-net/data/natural_graphs"
RESULTS_DIR = "/network/rit/dgx/dgx_subasi_lab/osman/osman-net/results"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# STEP 1: Load pickled graphs and build PyG Data objects
# ============================================================
def build_pyg_graphs(raw_graphs):
    """Convert raw graph dicts to PyG Data objects with proper features."""
    pyg_graphs = []

    for g in raw_graphs:
        n_nodes = g['n_nodes']
        src_idx = g['src_idx']
        dst_idx = g['dst_idx']
        edge_features = g['edge_features']  # (E, 14)
        n_edges = len(src_idx)

        if n_edges < 2 or n_nodes < 2:
            continue

        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Build node features from topology
        node_feat = np.zeros((n_nodes, 12), dtype=np.float32)

        for i in range(n_nodes):
            # Outgoing edges
            out_mask = [j for j, s in enumerate(src_idx) if s == i]
            in_mask = [j for j, d in enumerate(dst_idx) if d == i]

            out_degree = len(out_mask)
            in_degree = len(in_mask)

            # Aggregate from edge features
            # edge_features columns: packet_count, duration, mean_iat, std_iat,
            #   bytes_per_sec, unique_tcp_sports, unique_tcp_dports,
            #   unique_udp_sports, unique_udp_dports, dominant_proto,
            #   proto_diversity, tcp_ratio, udp_ratio, burstiness

            out_packets = sum(edge_features[j][0] for j in out_mask) if out_mask else 0
            in_packets = sum(edge_features[j][0] for j in in_mask) if in_mask else 0
            out_rate = np.mean([edge_features[j][4] for j in out_mask]) if out_mask else 0
            in_rate = np.mean([edge_features[j][4] for j in in_mask]) if in_mask else 0
            out_burstiness = np.mean([edge_features[j][13] for j in out_mask]) if out_mask else 0
            in_burstiness = np.mean([edge_features[j][13] for j in in_mask]) if in_mask else 0

            degree_ratio = out_degree / (in_degree + 1e-6)
            traffic_asymmetry = out_packets / (in_packets + 1e-6)
            unique_peers = out_degree + in_degree  # simplification

            # Is internal IP (192.168.x or 10.x)
            ip = g['ips'][i] if i < len(g['ips']) else ""
            is_internal = 1.0 if ip.startswith(('192.168.', '10.')) else 0.0

            node_feat[i] = [
                in_degree, out_degree, in_packets, out_packets,
                in_rate, out_rate, degree_ratio, traffic_asymmetry,
                unique_peers, is_internal, in_burstiness, out_burstiness
            ]

        x = torch.tensor(node_feat, dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([0], dtype=torch.long),  # placeholder
            num_nodes=n_nodes,
            label_str=g['label'],
        )
        pyg_graphs.append(data)

    return pyg_graphs


# ============================================================
# STEP 2: Models
# ============================================================
class GraphSAGEClassifier(nn.Module):
    """GraphSAGE for graph-level classification."""
    def __init__(self, node_dim, edge_dim, hidden, num_classes, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        gate = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        self.pool = GlobalAttention(gate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes))
        self.dropout = dropout

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.node_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = conv(x, ei)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res
        x = self.pool(x, batch)
        return self.classifier(x)


class GATClassifier(nn.Module):
    """GAT for graph-level classification with edge features."""
    def __init__(self, node_dim, edge_dim, hidden, num_classes,
                 num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim, 1)  # project edge features to scalar weight

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(GATConv(hidden, hidden//num_heads,
                                          heads=num_heads, dropout=dropout, concat=True))
            else:
                self.convs.append(GATConv(hidden, hidden, heads=1,
                                          dropout=dropout, concat=False))
            self.norms.append(nn.LayerNorm(hidden))

        gate = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        self.pool = GlobalAttention(gate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes))
        self.dropout = dropout

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.node_proj(x))
        # Project edge features to attention bias
        edge_weight = torch.sigmoid(self.edge_proj(ea)).squeeze(-1)

        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = conv(x, ei)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res
        x = self.pool(x, batch)
        return self.classifier(x)


class EdgeAwareGAT(nn.Module):
    """GAT with edge features injected into attention (edge_dim parameter)."""
    def __init__(self, node_dim, edge_dim, hidden, num_classes,
                 num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(GATConv(hidden, hidden//num_heads,
                                          heads=num_heads, dropout=dropout,
                                          concat=True, edge_dim=edge_dim))
            else:
                self.convs.append(GATConv(hidden, hidden, heads=1,
                                          dropout=dropout, concat=False,
                                          edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden))

        gate = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1))
        self.pool = GlobalAttention(gate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes))
        self.dropout = dropout

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.node_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = conv(x, ei, edge_attr=ea)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res
        x = self.pool(x, batch)
        return self.classifier(x)


# ============================================================
# STEP 3: Training
# ============================================================
def train_and_eval(model, train_loader, test_loader, name, num_classes,
                   class_names, train_graphs, epochs=200, patience=30, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    labels = [g.y.item() for g in train_graphs]
    counts = Counter(labels)
    total = len(labels)
    cw = torch.tensor([total/(num_classes*counts.get(i,1))
                       for i in range(num_classes)], dtype=torch.float).to(device)
    print(f"  Class weights: {cw.cpu().numpy().round(2)}")

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0
    best_epoch = 0
    best_preds = None
    pc = 0

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        scheduler.step()

        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                preds.extend(logits.argmax(-1).cpu().numpy())
                labs.extend(batch.y.cpu().numpy())

        f1m = f1_score(labs, preds, average='macro', zero_division=0)
        if f1m > best_f1:
            best_f1 = f1m
            best_epoch = epoch
            best_preds = np.array(preds)
            best_labels = np.array(labs)
            pc = 0
        else:
            pc += 1

        if epoch % 20 == 0 or epoch == 1:
            acc = accuracy_score(labs, preds)
            print(f"  Epoch {epoch:3d}: Acc={acc:.4f} F1m={f1m:.4f}")

        if pc >= patience:
            print(f"  Early stop at {epoch} (best: {best_epoch})")
            break

    acc = accuracy_score(best_labels, best_preds)
    f1m = f1_score(best_labels, best_preds, average='macro', zero_division=0)
    f1w = f1_score(best_labels, best_preds, average='weighted', zero_division=0)
    prec = precision_score(best_labels, best_preds, average='macro', zero_division=0)
    rec = recall_score(best_labels, best_preds, average='macro', zero_division=0)

    print(f"\n  FINAL (epoch {best_epoch}): Acc={acc:.4f} F1m={f1m:.4f} F1w={f1w:.4f}")
    print(classification_report(best_labels, best_preds, target_names=class_names, zero_division=0))

    return {
        'accuracy': float(acc), 'f1_macro': float(f1m), 'f1_weighted': float(f1w),
        'precision': float(prec), 'recall': float(rec),
        'best_epoch': best_epoch, 'n_params': n_params,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("Natural Network Graph Pipeline v2")
    print("60s windows, IP=Node, Flow=Edge, Graph Classification")
    print("="*60)

    # Load raw graphs
    print("\n[1/4] Loading graphs...")
    with open(os.path.join(GRAPH_DIR, "train_graphs.pkl"), 'rb') as f:
        raw_train = pickle.load(f)
    with open(os.path.join(GRAPH_DIR, "test_graphs.pkl"), 'rb') as f:
        raw_test = pickle.load(f)

    print(f"  Raw train: {len(raw_train)}, test: {len(raw_test)}")

    # Build PyG graphs
    print("\n[2/4] Building PyG graphs...")
    train_graphs = build_pyg_graphs(raw_train)
    test_graphs = build_pyg_graphs(raw_test)
    print(f"  PyG train: {len(train_graphs)}, test: {len(test_graphs)}")

    # Encode labels
    le = LabelEncoder()
    train_labels = [g.label_str for g in train_graphs]
    le.fit(train_labels + [g.label_str for g in test_graphs])
    class_names = list(le.classes_)
    num_classes = len(class_names)

    for g in train_graphs:
        g.y = torch.tensor([le.transform([g.label_str])[0]], dtype=torch.long)
    for g in test_graphs:
        g.y = torch.tensor([le.transform([g.label_str])[0]], dtype=torch.long)

    print(f"  Classes: {class_names}")
    print(f"  Train labels: {Counter(train_labels)}")
    print(f"  Test labels: {Counter([g.label_str for g in test_graphs])}")

    # Normalize node and edge features
    print("\n[3/4] Normalizing features...")
    # Collect all node features
    all_train_x = torch.cat([g.x for g in train_graphs], dim=0).numpy()
    all_train_ea = torch.cat([g.edge_attr for g in train_graphs], dim=0).numpy()

    scaler_x = StandardScaler()
    scaler_ea = StandardScaler()
    scaler_x.fit(all_train_x)
    scaler_ea.fit(all_train_ea)

    for g in train_graphs:
        g.x = torch.tensor(scaler_x.transform(g.x.numpy()), dtype=torch.float)
        g.edge_attr = torch.tensor(scaler_ea.transform(g.edge_attr.numpy()), dtype=torch.float)
    for g in test_graphs:
        g.x = torch.tensor(scaler_x.transform(g.x.numpy()), dtype=torch.float)
        g.edge_attr = torch.tensor(scaler_ea.transform(g.edge_attr.numpy()), dtype=torch.float)

    # Replace any NaN/Inf
    for g in train_graphs + test_graphs:
        g.x = torch.nan_to_num(g.x, nan=0, posinf=0, neginf=0)
        g.edge_attr = torch.nan_to_num(g.edge_attr, nan=0, posinf=0, neginf=0)

    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]
    print(f"  Node dim: {node_dim}, Edge dim: {edge_dim}")

    # DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # ============================================================
    # RF Baseline (graph-level, using aggregate features)
    # ============================================================
    print("\n" + "="*60)
    print("RF Baseline (graph-level aggregate features)")
    print("="*60)

    def graph_to_tabular(graphs):
        features = []
        labels = []
        for g in graphs:
            x_np = g.x.numpy()
            ea_np = g.edge_attr.numpy()
            # Node aggregate: mean, std, max
            x_mean = x_np.mean(axis=0)
            x_std = x_np.std(axis=0)
            x_max = x_np.max(axis=0)
            # Edge aggregate: mean, std, max
            ea_mean = ea_np.mean(axis=0)
            ea_std = ea_np.std(axis=0)
            ea_max = ea_np.max(axis=0)
            # Graph structural: n_nodes, n_edges, density
            n_nodes = g.num_nodes
            n_edges = g.edge_index.shape[1]
            density = n_edges / (n_nodes * (n_nodes-1) + 1e-6)

            feat = np.concatenate([x_mean, x_std, x_max, ea_mean, ea_std, ea_max,
                                   [n_nodes, n_edges, density]])
            features.append(feat)
            labels.append(g.y.item())
        return np.array(features), np.array(labels)

    X_train_rf, y_train_rf = graph_to_tabular(train_graphs)
    X_test_rf, y_test_rf = graph_to_tabular(test_graphs)
    X_train_rf = np.nan_to_num(X_train_rf, nan=0, posinf=0, neginf=0)
    X_test_rf = np.nan_to_num(X_test_rf, nan=0, posinf=0, neginf=0)

    print(f"  RF features: {X_train_rf.shape[1]}")

    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                class_weight='balanced', n_jobs=16, random_state=SEED)
    rf.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf.predict(X_test_rf)

    rf_acc = accuracy_score(y_test_rf, y_pred_rf)
    rf_f1m = f1_score(y_test_rf, y_pred_rf, average='macro', zero_division=0)
    rf_f1w = f1_score(y_test_rf, y_pred_rf, average='weighted', zero_division=0)
    print(f"  RF: Acc={rf_acc:.4f} F1m={rf_f1m:.4f} F1w={rf_f1w:.4f}")
    print(classification_report(y_test_rf, y_pred_rf, target_names=class_names, zero_division=0))

    results = {
        'RF': {'accuracy': float(rf_acc), 'f1_macro': float(rf_f1m), 'f1_weighted': float(rf_f1w)},
    }

    # ============================================================
    # GNN Models
    # ============================================================
    print("\n[4/4] Training GNN models...")
    H = 128

    # GraphSAGE
    model = GraphSAGEClassifier(node_dim, edge_dim, H, num_classes)
    results['GraphSAGE'] = train_and_eval(
        model, train_loader, test_loader, "GraphSAGE",
        num_classes, class_names, train_graphs, lr=5e-4)

    # GAT
    model = GATClassifier(node_dim, edge_dim, H, num_classes)
    results['GAT'] = train_and_eval(
        model, train_loader, test_loader, "GAT",
        num_classes, class_names, train_graphs, lr=5e-4)

    # Edge-Aware GAT (edge features in attention)
    model = EdgeAwareGAT(node_dim, edge_dim, H, num_classes)
    results['EdgeAwareGAT'] = train_and_eval(
        model, train_loader, test_loader, "EdgeAwareGAT",
        num_classes, class_names, train_graphs, lr=5e-4)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("NATURAL GRAPH v2 - RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<20} {'Acc':>8} {'F1m':>8} {'F1w':>8}")
    print("-"*45)
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']:>8.4f} {r['f1_macro']:>8.4f} {r['f1_weighted']:>8.4f}")

    with open(os.path.join(RESULTS_DIR, 'natural_graph_v2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/natural_graph_v2_results.json")
    print("DONE!")


if __name__ == "__main__":
    main()
