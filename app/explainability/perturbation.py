"""Perturbation-based explainability for GNN models.
Removes common neighbors one at a time and measures prediction change (delta).
Higher delta = more influential neighbor."""
import torch
import numpy as np


def get_common_neighbors(drug_a: int, drug_b: int, edge_index: torch.Tensor) -> list:
    """Find nodes that are neighbors of both drug_a and drug_b."""
    # Get neighbors of each drug
    neighbors_a = set(edge_index[1][edge_index[0] == drug_a].cpu().numpy())
    neighbors_b = set(edge_index[1][edge_index[0] == drug_b].cpu().numpy())
    common = neighbors_a.intersection(neighbors_b)
    # Exclude the drugs themselves
    common.discard(drug_a)
    common.discard(drug_b)
    return sorted(list(common))


@torch.no_grad()
def perturbation_analysis(model, predictor, drug_a: int, drug_b: int,
                          edge_index: torch.Tensor, device: torch.device,
                          fingerprints: torch.Tensor = None,
                          top_k: int = 10) -> dict:
    """
    Remove common neighbors one at a time, measure prediction delta.

    Returns dict with:
        - base_score: original prediction
        - neighbors: list of {node_id, delta, new_score}
        - top_neighbors: top-k most influential
    """
    model.eval()
    predictor.eval()

    # Get base prediction
    use_features = hasattr(model, 'use_features') and model.use_features
    if use_features and fingerprints is not None:
        emb = model.encode(fingerprints.to(device), edge_index.to(device))
    else:
        emb = model.encode(None, edge_index.to(device))

    base_score = torch.sigmoid(
        predictor(emb[drug_a].unsqueeze(0), emb[drug_b].unsqueeze(0))
    ).item()

    # Find common neighbors
    common_neighbors = get_common_neighbors(drug_a, drug_b, edge_index)

    if not common_neighbors:
        return {
            "base_score": base_score,
            "neighbors": [],
            "top_neighbors": [],
            "message": "No common neighbors found for this drug pair",
        }

    # Perturb each neighbor
    results = []
    for neighbor in common_neighbors:
        # Remove all edges involving this neighbor
        mask = (edge_index[0] != neighbor) & (edge_index[1] != neighbor)
        perturbed_edge_index = edge_index[:, mask]

        # Re-encode with perturbed graph
        if use_features and fingerprints is not None:
            perturbed_emb = model.encode(
                fingerprints.to(device), perturbed_edge_index.to(device)
            )
        else:
            perturbed_emb = model.encode(None, perturbed_edge_index.to(device))

        new_score = torch.sigmoid(
            predictor(
                perturbed_emb[drug_a].unsqueeze(0),
                perturbed_emb[drug_b].unsqueeze(0),
            )
        ).item()

        delta = abs(base_score - new_score)
        results.append({
            "node_id": int(neighbor),
            "delta": round(delta, 6),
            "new_score": round(new_score, 6),
        })

    # Sort by delta (most influential first)
    results.sort(key=lambda x: x["delta"], reverse=True)

    return {
        "base_score": round(base_score, 6),
        "num_common_neighbors": len(common_neighbors),
        "neighbors": results,
        "top_neighbors": results[:top_k],
    }
