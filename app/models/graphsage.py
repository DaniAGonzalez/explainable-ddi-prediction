"""GraphSAGE model matching trained checkpoints.
Baseline uses nn.Embedding (key: 'embedding'), molecular uses nn.Linear (key: 'input_proj')."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE without features. Uses nn.Embedding for node representations."""
    def __init__(self, num_nodes=4267, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def encode(self, x_or_index, edge_index):
        h = self.embedding.weight
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GraphSAGEMolecular(nn.Module):
    """GraphSAGE with molecular features. Uses nn.Linear (key: 'input_proj')."""
    def __init__(self, input_dim=2048, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def encode(self, x_or_index, edge_index):
        h = self.input_proj(x_or_index)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
