"""
GNN models for graph-level classification.

AdaptiveGAT: GAT with edge feature injection + learnable edge weight fusion.
PureGAT: GAT with edge features, no adaptive weighting.
GraphSAGEModel: SAGE convolution baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GlobalAttention


class AdaptiveGAT(nn.Module):
    """
    Adaptive Edge-Weighted Graph Attention Network.
    
    Fuses uniform prior edge weights with learned weights:
        w_fused = α * w_prior + (1 - α) * w_learned
    where α is a learnable scalar.
    """
    def __init__(self, node_dim, edge_dim, hidden=128, num_classes=6,
                 num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.edge_wt = nn.Sequential(
            nn.Linear(edge_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid())
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(GATConv(hidden, hidden // num_heads,
                                          heads=num_heads, dropout=dropout,
                                          concat=True, edge_dim=1))
            else:
                self.convs.append(GATConv(hidden, hidden, heads=1,
                                          dropout=dropout, concat=False, edge_dim=1))
            self.norms.append(nn.LayerNorm(hidden))

        self.pool = GlobalAttention(nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        self.cls = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes))
        self.dropout = dropout

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.node_proj(x))
        alpha = torch.sigmoid(self.alpha_logit)
        lw = self.edge_wt(ea).squeeze(-1)
        w = alpha * torch.ones_like(lw) + (1 - alpha) * lw
        for conv, norm in zip(self.convs, self.norms):
            r = x
            x = conv(x, ei, edge_attr=w.unsqueeze(-1))
            x = norm(x); x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + r
        return self.cls(self.pool(x, batch))


class PureGAT(nn.Module):
    """GAT with edge feature injection, no adaptive weighting."""
    def __init__(self, node_dim, edge_dim, hidden=128, num_classes=6,
                 num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(GATConv(hidden, hidden // num_heads, heads=num_heads,
                                          dropout=dropout, concat=True, edge_dim=edge_dim))
            else:
                self.convs.append(GATConv(hidden, hidden, heads=1,
                                          dropout=dropout, concat=False, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden))
        self.pool = GlobalAttention(nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        self.cls = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes))
        self.dropout = dropout

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.node_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            r = x
            x = conv(x, ei, edge_attr=ea)
            x = norm(x); x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + r
        return self.cls(self.pool(x, batch))


class GraphSAGEModel(nn.Module):
    """GraphSAGE baseline (no edge features)."""
    def __init__(self, node_dim, edge_dim, hidden=128, num_classes=6,
                 num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))
        self.pool = GlobalAttention(nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)))
        self.cls = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes))
        self.dropout = dropout

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.node_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            r = x
            x = conv(x, ei)
            x = norm(x); x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + r
        return self.cls(self.pool(x, batch))
