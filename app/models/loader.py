"""Unified model loader with GATConv key remapping.

Handles version mismatch: checkpoints trained with older PyG (using 'lin')
loaded into PyG 2.4+ (expecting 'lin_src' and 'lin_dst').
"""
import torch
import logging
from .gat import GATBase, GATSkip
from .graphsage import GraphSAGEBaseline, GraphSAGEMolecular
from .mlp import MLPFullModel
from .link_predictor import GATLinkPredictor, GraphSAGELinkPredictor

logger = logging.getLogger(__name__)


def remap_gat_keys(state_dict: dict) -> dict:
    """Remap old GATConv 'lin.weight' keys to new 'lin_src.weight' + 'lin_dst.weight'.

    Old PyG GATConv had a single 'lin' for both src and dst.
    New PyG 2.4+ splits it into 'lin_src' and 'lin_dst'.
    We duplicate the weight to both since old version shared them.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if ".lin.weight" in key:
            # e.g. "convs.0.lin.weight" -> "convs.0.lin_src.weight" + "convs.0.lin_dst.weight"
            base = key.replace(".lin.weight", "")
            new_state_dict[f"{base}.lin_src.weight"] = value.clone()
            new_state_dict[f"{base}.lin_dst.weight"] = value.clone()
            logger.debug(f"Remapped {key} -> lin_src + lin_dst")
        elif ".lin.bias" in key:
            base = key.replace(".lin.bias", "")
            new_state_dict[f"{base}.lin_src.bias"] = value.clone()
            new_state_dict[f"{base}.lin_dst.bias"] = value.clone()
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_model(model_config: dict, device: torch.device,
               num_nodes: int = 4267, fingerprints: torch.Tensor = None,
               edge_index: torch.Tensor = None):
    """Load a trained model + predictor from checkpoint."""

    model_type = model_config["type"]
    checkpoint_path = model_config["checkpoint"]

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if model_type == "gat":
        variant = model_config.get("variant", "base")
        use_features = model_config.get("use_features", False)
        ModelClass = GATSkip if variant == "skip" else GATBase
        model = ModelClass(num_nodes=num_nodes, hidden_dim=128, heads=4,
                           use_features=use_features)
        predictor = GATLinkPredictor(hidden_dim=128)

        # Remap old GATConv keys to new PyG 2.4+ format
        model_state = remap_gat_keys(checkpoint["model"])
        model.load_state_dict(model_state)
        predictor.load_state_dict(checkpoint["predictor"])
        logger.info(f"Loaded GAT {variant} ({'feat' if use_features else 'no feat'}) with key remapping")

    elif model_type == "graphsage":
        use_features = model_config.get("use_features", False)
        if use_features:
            model = GraphSAGEMolecular(input_dim=2048, hidden_dim=128)
        else:
            model = GraphSAGEBaseline(num_nodes=num_nodes, hidden_dim=128)
        predictor = GraphSAGELinkPredictor(hidden_dim=128)

        model.load_state_dict(checkpoint["model"])
        predictor.load_state_dict(checkpoint["predictor"])
        logger.info(f"Loaded GraphSAGE {'molecular' if use_features else 'baseline'}")

    elif model_type == "mlp":
        full_model = MLPFullModel(input_dim=2048, hidden_dim=128)
        full_model.load_state_dict(checkpoint["model"])
        model = full_model.encoder
        predictor = full_model.predictor
        logger.info("Loaded MLP baseline")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device).eval()
    predictor.to(device).eval()

    return {
        "model": model,
        "predictor": predictor,
        "fingerprints": fingerprints,
        "edge_index": edge_index,
        "config": model_config,
    }
