"""Integrated Gradients for models that use molecular fingerprints.
Maps attributions back to molecular substructures via RDKit."""
import torch
import numpy as np
import io
import base64

# RDKit imports for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def integrated_gradients(model, predictor, drug_a: int, drug_b: int,
                         fingerprints: torch.Tensor, device: torch.device,
                         edge_index: torch.Tensor = None,
                         n_steps: int = 50, top_k: int = 10) -> dict:
    """
    Compute Integrated Gradients attributions over fingerprint features.

    Works for both GNN models (with edge_index) and MLP (without).
    Returns top-k most attributed fingerprint bits for each drug.
    """
    model.eval()
    fp = fingerprints.clone().to(device)

    # Baseline: zero fingerprints
    baseline = torch.zeros_like(fp)

    # Accumulate gradients along interpolation path
    accumulated_grads = torch.zeros_like(fp)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interpolated = baseline + alpha * (fp - baseline)
        interpolated.requires_grad_(True)

        # Forward pass
        is_mlp = not hasattr(model, 'convs')
        if is_mlp:
            emb = model.encode(interpolated)
        else:
            emb = model.encode(interpolated, edge_index.to(device))

        # Predict for this specific pair
        score = predictor(emb[drug_a].unsqueeze(0), emb[drug_b].unsqueeze(0))

        # Backward
        model.zero_grad()
        if interpolated.grad is not None:
            interpolated.grad.zero_()
        score.backward()

        if interpolated.grad is not None:
            accumulated_grads += interpolated.grad.detach()

    # IG = (input - baseline) * average_gradient
    ig_attributions = (fp - baseline) * accumulated_grads / (n_steps + 1)

    # Get attributions for drug_a and drug_b
    result = {}
    for drug_name, drug_idx in [("drug_a", drug_a), ("drug_b", drug_b)]:
        attr = ig_attributions[drug_idx].cpu().numpy()
        top_bits_pos = np.argsort(attr)[::-1][:top_k]
        top_bits_neg = np.argsort(attr)[:top_k]

        result[drug_name] = {
            "node_id": drug_idx,
            "top_positive_bits": [
                {"bit": int(b), "attribution": round(float(attr[b]), 6)}
                for b in top_bits_pos
            ],
            "top_negative_bits": [
                {"bit": int(b), "attribution": round(float(attr[b]), 6)}
                for b in top_bits_neg
            ],
            "total_attribution": round(float(np.sum(np.abs(attr))), 4),
        }

    return result


def visualize_ig_on_molecule(smiles: str, attributions: list, drug_name: str = "") -> str:
    """
    Highlight attributed substructures on the molecular structure.
    Returns base64-encoded PNG image.

    attributions: list of {'bit': int, 'attribution': float}
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Generate Morgan fingerprint info to map bits to atoms
    bi = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=bi)

    # Collect atoms that correspond to top attributed bits
    highlight_atoms = set()
    atom_colors = {}
    for attr_info in attributions:
        bit_idx = attr_info["bit"]
        attr_val = attr_info["attribution"]

        if bit_idx in bi:
            for center_atom, radius in bi[bit_idx]:
                # Get atoms in this substructure
                if radius == 0:
                    atoms = {center_atom}
                else:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_atom)
                    atoms = set()
                    for bond_idx in env:
                        bond = mol.GetBondWithIdx(bond_idx)
                        atoms.add(bond.GetBeginAtomIdx())
                        atoms.add(bond.GetEndAtomIdx())

                highlight_atoms.update(atoms)
                # Color: red for positive (promotes interaction), blue for negative
                color = (1.0, 0.4, 0.4) if attr_val > 0 else (0.4, 0.4, 1.0)
                for a in atoms:
                    atom_colors[a] = color

 # Draw molecule with highlights
    try:
        img = Draw.MolToImage(
            mol,
            size=(400, 300),
            highlightAtoms=list(highlight_atoms),
            highlightColor=(1.0, 0.4, 0.4),
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        return img_b64
    except Exception as e:
        print(f"Visualization error: {e}")
        return None
