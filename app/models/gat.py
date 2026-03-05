"""GAT model variants matching trained checkpoints.
Architecture: hidden_dim=64, heads=2, num_layers=3."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATBase(nn.Module):
    def __init__(self, num_nodes=4267, hidden_dim=64, num_layers=3,
                 heads=2, dropout=0.1, use_features=False, input_dim=2048):
        super().__init__()
        self.use_features = use_features
        self.dropout = dropout
        self.num_layers = num_layers

        if use_features:
            self.encoder = nn.Linear(input_dim, hidden_dim)
        else:
            self.encoder = nn.Embedding(num_nodes, hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim, heads=heads,
                            concat=False, dropout=dropout))
            else:
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                            concat=True, dropout=dropout))

    def encode(self, x_or_index, edge_index):
        if self.use_features:
            h = self.encoder(x_or_index)
        else:
            h = self.encoder.weight
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def encode_with_attention(self, x_or_index, edge_index):
        if self.use_features:
            h = self.encoder(x_or_index)
        else:
            h = self.encoder.weight
        attention_weights = []
        for i, conv in enumerate(self.convs):
            h, (edge_idx, alpha) = conv(h, edge_index, return_attention_weights=True)
            attention_weights.append({"edge_index": edge_idx.cpu(), "alpha": alpha.cpu()})
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h, attention_weights


class GATSkip(nn.Module):
    def __init__(self, num_nodes=4267, hidden_dim=64, num_layers=3,
                 heads=2, dropout=0.1, use_features=False, input_dim=2048):
        super().__init__()
        self.use_features = use_features
        self.dropout = dropout
        self.num_layers = num_layers

        if use_features:
            self.encoder = nn.Linear(input_dim, hidden_dim)
        else:
            self.encoder = nn.Embedding(num_nodes, hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim, heads=heads,
                            concat=False, dropout=dropout))
            else:
                self.convs.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                            concat=True, dropout=dropout))

    def encode(self, x_or_index, edge_index):
        if self.use_features:
            h = self.encoder(x_or_index)
        else:
            h = self.encoder.weight
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index)
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = h + h_in
            else:
                h = h + h_in
        return h

    def encode_with_attention(self, x_or_index, edge_index):
        if self.use_features:
            h = self.encoder(x_or_index)
        else:
            h = self.encoder.weight
        attention_weights = []
        for i, conv in enumerate(self.convs):
            h_in = h
            h, (edge_idx, alpha) = conv(h, edge_index, return_attention_weights=True)
            attention_weights.append({"edge_index": edge_idx.cpu(), "alpha": alpha.cpu()})
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = h + h_in
            else:
                h = h + h_in
        return h, attention_weights
