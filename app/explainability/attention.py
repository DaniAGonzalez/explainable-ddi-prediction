"""Attention-based explainability for GAT models.
Extracts attention weights to show which neighbors the model attends to most."""
import torch
import numpy as np


@torch.no_grad()
def attention_analysis(model, drug_a: int, drug_b: int,
                       edge_index: torch.Tensor, device: torch.device,
                       fingerprints: torch.Tensor = None,
                       top_k: int = 10) -> dict:
    """
    Extract attention weights for a drug pair from all GAT layers.

    Returns dict with per-layer attention weights for neighbors of drug_a and drug_b.
    """
    model.eval()

    # Forward pass with attention
    use_features = hasattr(model, 'use_features') and model.use_features
    if use_features and fingerprints is not None:
        emb, attn_weights = model.encode_with_attention(
            fingerprints.to(device), edge_index.to(device)
        )
    else:
        emb, attn_weights = model.encode_with_attention(
            None, edge_index.to(device)
        )

    results = {"drug_a": drug_a, "drug_b": drug_b, "layers": []}

    for layer_idx, layer_attn in enumerate(attn_weights):
        edge_idx = layer_attn["edge_index"]  # (2, num_edges)
        alpha = layer_attn["alpha"]  # (num_edges, num_heads) or (num_edges,)

        # Average attention across heads if multi-head
        if alpha.dim() > 1:
            alpha_mean = alpha.mean(dim=1)
        else:
            alpha_mean = alpha

        layer_result = {"layer": layer_idx}

        # Get attention for drug_a's neighbors
        for drug_name, drug_idx in [("drug_a", drug_a), ("drug_b", drug_b)]:
            mask = edge_idx[1] == drug_idx  # edges pointing TO this drug
            if mask.sum() > 0:
                neighbor_indices = edge_idx[0][mask].numpy()
                attention_scores = alpha_mean[mask].numpy()

                # Sort by attention score
                sorted_idx = np.argsort(attention_scores)[::-1]
                top = sorted_idx[:top_k]

                neighbors = [
                    {
                        "node_id": int(neighbor_indices[i]),
                        "attention": round(float(attention_scores[i]), 6),
                    }
                    for i in top
                ]
                layer_result[drug_name] = {
                    "num_neighbors": int(mask.sum()),
                    "top_neighbors": neighbors,
                }
            else:
                layer_result[drug_name] = {
                    "num_neighbors": 0,
                    "top_neighbors": [],
                }

        results["layers"].append(layer_result)

    return results
