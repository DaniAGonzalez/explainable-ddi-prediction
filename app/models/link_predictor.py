"""LinkPredictor variants matching trained checkpoints.
- GAT models: hidden_dim=64, keys = mlp.0.weight, mlp.3.weight, mlp.6.weight
- GraphSAGE models: hidden_dim=128, keys = layers.0.weight, layers.1.weight, layers.2.weight
"""
import torch.nn as nn


class GATLinkPredictor(nn.Module):
    """Predictor for GAT models. hidden_dim=64, uses 'mlp' key."""
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.5):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, emb_a, emb_b):
        return self.mlp(emb_a * emb_b).squeeze(-1)


class GraphSAGELinkPredictor(nn.Module):
    """Predictor for GraphSAGE models. hidden_dim=128, uses 'layers' key."""
    def __init__(self, hidden_dim=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, emb_a, emb_b):
        h = emb_a * emb_b
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                import torch.nn.functional as F
                h = F.relu(h)
        return h.squeeze(-1)
