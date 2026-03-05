"""MLP baseline matching trained checkpoint.
Checkpoint saves entire model (encoder + predictor) under 'model' key.
Keys: model.encoder.encoder.weight, model.predictor.mlp.*.weight"""
import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)

    def encode(self, fingerprints):
        return self.encoder(fingerprints)


class MLPPredictor(nn.Module):
    """MLP predictor matching checkpoint structure."""
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.5):
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


class MLPFullModel(nn.Module):
    """Combined model matching checkpoint key structure:
    model.encoder.encoder.* and model.predictor.mlp.*"""
    def __init__(self, input_dim=2048, hidden_dim=128):
        super().__init__()
        self.encoder = MLPEncoder(input_dim, hidden_dim)
        self.predictor = MLPPredictor(hidden_dim)
