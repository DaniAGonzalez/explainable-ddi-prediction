"""Configuration for all DDI models."""
import os
import torch

DEVICE = torch.device(os.getenv("DEVICE", "cpu"))

NUM_NODES = 4267
FINGERPRINT_DIM = 2048

CHECKPOINTS_DIR = "checkpoints"
DATA_DIR = "data"
FINGERPRINTS_PATH = os.path.join(DATA_DIR, "fingerprints.pt")
DRUG_MAPPING_PATH = os.path.join(DATA_DIR, "drug_mapping_smiles.csv")
DRUGBANK_LOOKUP_PATH = os.path.join(DATA_DIR, "ogb_drug_lookup.json")

MODELS = {
    "graphsage_baseline": {
        "type": "graphsage",
        "use_features": False,
        "hidden_dim": 128,
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "graphsage_baseline.pt"),
        "description": "GraphSAGE topology only",
    },
    "graphsage_molecular": {
        "type": "graphsage",
        "use_features": True,
        "hidden_dim": 128,
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "graphsage_molecular.pt"),
        "description": "GraphSAGE with Morgan fingerprints",
    },
    "gat_base": {
        "type": "gat",
        "variant": "base",
        "use_features": False,
        "hidden_dim": 128,
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "gat_base.pt"),
        "description": "GAT base, no features (93.5% AUC)",
    },
    "gat_skip": {
        "type": "gat",
        "variant": "skip",
        "use_features": False,
        "hidden_dim": 128,
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "gat_skip.pt"),
        "description": "GAT + skip connections (94.8% AUC)",
    },
    "gat_base_feat": {
        "type": "gat",
        "variant": "base",
        "use_features": True,
        "hidden_dim": 128,
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "gat_base_feat.pt"),
        "description": "GAT base + fingerprints (91.9% AUC)",
    },
    "gat_skip_feat": {
        "type": "gat",
        "variant": "skip",
        "use_features": True,
        "hidden_dim": 128,
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "gat_skip_feat.pt"),
        "description": "GAT Skip + Features — best model (97.5% AUC)",
    },
    "mlp_baseline": {
        "type": "mlp",
        "use_features": True,
        "hidden_dim": 128,
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "mlp_baseline.pt"),
        "description": "MLP baseline, fingerprints only (96.9% AUC)",
    },
}
