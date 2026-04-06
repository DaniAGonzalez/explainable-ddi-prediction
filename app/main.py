"""
DDI Prediction API — Explainable Drug-Drug Interaction Prediction

FastAPI backend serving 7 trained GNN variants with three explainability
methods (perturbation analysis, attention weights, Integrated Gradients)
and a RAG pipeline for natural language explanations.

Endpoints:
    GET  /          -> API info and loaded model list
    GET  /health    -> Health check
    GET  /models    -> List available models and their explainability methods
    GET  /drugs     -> Drug index-to-name mapping for autocomplete
    POST /predict   -> Predict interaction probability for a drug pair
    POST /explain   -> Run explainability analysis on a drug pair
    POST /rag       -> Generate natural language explanation via RAG pipeline
"""

import csv
import json
import logging
import os

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import MODELS, DEVICE, FINGERPRINTS_PATH, DRUG_MAPPING_PATH, NUM_NODES
from .middleware import RequestLoggingMiddleware
from .rag import init_rag, generate_explanation, get_drug_name
from .schemas import PredictRequest, PredictResponse, ExplainRequest, ModelInfo
from .models.loader import load_model
from .explainability.perturbation import perturbation_analysis
from .explainability.attention import attention_analysis
from .explainability.integrated_gradients import (
    integrated_gradients,
    visualize_ig_on_molecule,
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — populated at startup, cleared at shutdown
# ---------------------------------------------------------------------------
loaded_models: dict = {}
fingerprints: torch.Tensor | None = None
edge_index: torch.Tensor | None = None
drug_smiles: dict = {}


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def load_graph_data() -> torch.Tensor:
    """Load OGB-DDI graph and return a symmetrized edge index.

    Edges are stored once per direction in OGB; we concatenate the flipped
    edges to ensure bidirectional message passing in all GNN variants.

    Returns:
        Tensor of shape (2, 2 * num_train_edges) — symmetrized edge index.

    Raises:
        RuntimeError: If the OGB dataset cannot be downloaded or loaded.
    """
    from ogb.linkproppred import PygLinkPropPredDataset

    logger.info("Loading OGB-DDI dataset...")
    dataset = PygLinkPropPredDataset(name="ogbl-ddi")
    split = dataset.get_edge_split()
    train_edges = split["train"]["edge"]

    if not isinstance(train_edges, torch.Tensor):
        train_edges = torch.from_numpy(train_edges)

    # Symmetrize: concatenate original + flipped edges
    ei = torch.cat([train_edges.t(), train_edges.flip(1).t()], dim=1)
    logger.info(f"Graph loaded: {ei.shape[1]} edges")
    return ei


def load_smiles_mapping(path: str) -> dict:
    """Load drug index → SMILES mapping from a CSV file.

    Handles both 'node idx' and 'node_idx' column name variants produced
    by different versions of the preprocessing pipeline.

    Args:
        path (str): Path to the CSV file with columns node_idx and smiles.

    Returns:
        dict mapping int(node_idx) -> smiles string.
        Returns empty dict if the file does not exist.
    """
    mapping = {}
    if not os.path.exists(path):
        logger.warning(f"SMILES mapping not found at {path}")
        return mapping

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Column name varies across preprocessing versions
            idx_key = "node idx" if "node idx" in row else "node_idx"
            try:
                mapping[int(row[idx_key])] = row["smiles"]
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed SMILES row: {e}")

    logger.info(f"Loaded SMILES for {len(mapping)} drugs")
    return mapping


# ---------------------------------------------------------------------------
# Application lifespan — startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown.

    Startup sequence:
        1. Load Morgan fingerprints from disk.
        2. Load and symmetrize the OGB-DDI graph.
        3. Load drug SMILES mapping for IG visualization.
        4. Load all 7 model checkpoints.
        5. Initialize the RAG pipeline (ChromaDB + Claude client).

    Shutdown: clears loaded_models to free GPU memory.
    """
    global loaded_models, fingerprints, edge_index, drug_smiles

    logger.info("=" * 60)
    logger.info("Starting DDI Prediction API")
    logger.info("=" * 60)

    # 1. Fingerprints
    if os.path.exists(FINGERPRINTS_PATH):
        fingerprints = torch.load(
            FINGERPRINTS_PATH, map_location=DEVICE, weights_only=False
        )
        logger.info(f"Fingerprints loaded: {fingerprints.shape}")
    else:
        logger.warning(f"Fingerprints not found at {FINGERPRINTS_PATH} — "
                       "feature-based models will be unavailable.")

    # 2. Graph
    edge_index = load_graph_data()

    # 3. SMILES
    drug_smiles = load_smiles_mapping(DRUG_MAPPING_PATH)

    # 4. Models
    for name, config in MODELS.items():
        if not os.path.exists(config["checkpoint"]):
            logger.warning(f"Checkpoint not found for {name} — SKIPPING")
            continue
        try:
            loaded_models[name] = load_model(
                config, DEVICE, NUM_NODES, fingerprints, edge_index
            )
            logger.info(f"✓ Loaded {name}")
        except Exception as e:
            logger.error(f"✗ Failed to load {name}: {e}")

    logger.info(f"Models loaded: {len(loaded_models)} / {len(MODELS)}")

    # 5. RAG pipeline
    init_rag()

    logger.info("API ready!")
    logger.info("=" * 60)

    yield  # Application runs here

    loaded_models.clear()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DDI Prediction API",
    description="Explainable Drug-Drug Interaction Prediction using GNNs",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/app")
async def serve_frontend():
    """Serve the DDI Explorer frontend."""
    return FileResponse("static/index.html")


@app.get("/")
def root():
    """Return API metadata and list of loaded models."""
    return {
        "name": "DDI Prediction API",
        "models_loaded": list(loaded_models.keys()),
        "total_drugs": NUM_NODES,
        "endpoints": ["/predict", "/explain", "/rag", "/models", "/drugs", "/health"],
    }


@app.get("/health")
def health():
    """Return health status and number of loaded models."""
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "device": str(DEVICE),
    }


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    """List all configured models with their available explainability methods.

    Returns:
        List of ModelInfo objects. Each entry includes the model name,
        description, architecture type, whether molecular features are used,
        and which explainability methods are available for that variant.
    """
    result = []
    for name, config in MODELS.items():
        model_type = config["type"]
        use_features = config.get("use_features", False)

        # Available methods depend on architecture and input modality
        methods = []
        if model_type in ("graphsage", "gat"):
            methods.append("perturbation")
        if model_type == "gat":
            methods.append("attention")
        if use_features:
            methods.append("integrated_gradients")

        result.append(ModelInfo(
            name=name,
            description=config["description"],
            type=model_type,
            use_features=use_features,
            available_methods=methods,
        ))
    return result


@app.get("/drugs")
def list_drugs():
    """Return drug index-to-name mapping for frontend autocomplete.

    Reads from a pre-built JSON file generated during data preprocessing.
    Returns an empty dict if the file is not found rather than raising,
    so the frontend degrades gracefully to manual index entry.

    Returns:
        dict mapping str(node_idx) -> drug_name.
    """
    path = "data/rag/idx_to_name.json"
    if not os.path.exists(path):
        logger.warning(f"Drug name mapping not found at {path}")
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load drug name mapping: {e}")
        return {}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict interaction probability for a drug pair.

    Runs a forward pass through the requested model and returns both the
    raw logit score and the sigmoid probability. The MLP model uses only
    molecular fingerprints; GNN models additionally use the graph edge index.

    Args:
        req: PredictRequest with drug_a (int), drug_b (int), model_name (str).

    Returns:
        PredictResponse with probability [0, 1] and raw logit score.

    Raises:
        HTTPException 404: If the requested model is not loaded.
    """
    if req.model_name not in loaded_models:
        raise HTTPException(
            404,
            f"Model '{req.model_name}' not loaded. "
            f"Available: {list(loaded_models.keys())}"
        )

    bundle = loaded_models[req.model_name]
    model     = bundle["model"]
    predictor = bundle["predictor"]
    config    = bundle["config"]

    with torch.no_grad():
        if config["type"] == "mlp":
            emb = model.encode(fingerprints.to(DEVICE))
        elif config.get("use_features") and fingerprints is not None:
            emb = model.encode(fingerprints.to(DEVICE), edge_index.to(DEVICE))
        else:
            emb = model.encode(None, edge_index.to(DEVICE))

        raw_score = predictor(
            emb[req.drug_a].unsqueeze(0),
            emb[req.drug_b].unsqueeze(0)
        ).item()
        probability = torch.sigmoid(torch.tensor(raw_score)).item()

    return PredictResponse(
        drug_a=req.drug_a,
        drug_b=req.drug_b,
        model_name=req.model_name,
        probability=round(probability, 6),
        raw_score=round(raw_score, 6),
    )


@app.post("/explain")
def explain(req: ExplainRequest):
    """Run explainability analysis on a drug pair.

    Dispatches to the appropriate explainability method based on req.method:
    - perturbation: masks neighbors one by one, measures score delta.
    - attention: extracts learned attention weights from GAT layers.
    - integrated_gradients: attributes prediction to fingerprint bits via IG.

    IG visualization is only generated if req.include_visualization=True
    and the drug has a SMILES string in the mapping.

    Args:
        req: ExplainRequest with drug_a, drug_b, model_name, method, top_k,
             and include_visualization.

    Returns:
        dict with model_name, method, drug_a, drug_b, analysis results,
        and optionally visualizations (base64 PNG per drug).

    Raises:
        HTTPException 404: Model not loaded.
        HTTPException 400: Method not available for this model variant.
        HTTPException 500: Fingerprints not loaded (required for IG).
    """
    if req.model_name not in loaded_models:
        raise HTTPException(
            404,
            f"Model '{req.model_name}' not loaded. "
            f"Available: {list(loaded_models.keys())}"
        )

    bundle     = loaded_models[req.model_name]
    model      = bundle["model"]
    predictor  = bundle["predictor"]
    config     = bundle["config"]
    method     = req.method
    model_type = config["type"]
    use_features = config.get("use_features", False)

    # Determine available methods for this model variant
    available_methods = []
    if model_type in ("graphsage", "gat"):
        available_methods.append("perturbation")
    if model_type == "gat":
        available_methods.append("attention")
    if use_features:
        available_methods.append("integrated_gradients")

    if method not in available_methods:
        raise HTTPException(
            400,
            f"Method '{method}' not available for {req.model_name}. "
            f"Available: {available_methods}"
        )

    result = {
        "model_name": req.model_name,
        "method": method,
        "drug_a": req.drug_a,
        "drug_b": req.drug_b,
    }

    if method == "perturbation":
        result["analysis"] = perturbation_analysis(
            model, predictor, req.drug_a, req.drug_b,
            edge_index, DEVICE,
            fingerprints if use_features else None,
            top_k=req.top_k,
        )

    elif method == "attention":
        result["analysis"] = attention_analysis(
            model, req.drug_a, req.drug_b,
            edge_index, DEVICE,
            fingerprints if use_features else None,
            top_k=req.top_k,
        )

    elif method == "integrated_gradients":
        if fingerprints is None:
            raise HTTPException(500, "Fingerprints not loaded — cannot run IG.")

        ig_result = integrated_gradients(
            model, predictor, req.drug_a, req.drug_b,
            fingerprints, DEVICE,
            edge_index=edge_index if model_type != "mlp" else None,
            top_k=req.top_k,
        )
        result["analysis"] = ig_result

        # Optional molecular visualization — skipped if SMILES unavailable
        if req.include_visualization:
            visualizations = {}
            for drug_key in ["drug_a", "drug_b"]:
                drug_idx = ig_result[drug_key]["node_id"]
                smiles = drug_smiles.get(drug_idx)
                if not smiles:
                    continue
                all_attrs = (
                    ig_result[drug_key]["top_positive_bits"]
                    + ig_result[drug_key]["top_negative_bits"]
                )
                img_b64 = visualize_ig_on_molecule(smiles, all_attrs, drug_key)
                if img_b64:
                    visualizations[drug_key] = {
                        "smiles": smiles,
                        "image_base64": img_b64,
                        "format": "png",
                    }
            if visualizations:
                result["visualizations"] = visualizations

    return result


@app.post("/rag")
def rag_explain(req: PredictRequest):
    """Generate a natural language DDI explanation via the RAG pipeline.

    Runs a prediction forward pass to obtain the interaction probability,
    then passes the drug indices and probability to the RAG pipeline which
    retrieves pharmacological context from ChromaDB and generates an
    explanation using the Claude API.

    Args:
        req: PredictRequest with drug_a, drug_b, model_name.

    Returns:
        dict with the generated natural language explanation and supporting
        context retrieved from the knowledge base.

    Raises:
        HTTPException 404: If the requested model is not loaded.
    """
    if req.model_name not in loaded_models:
        raise HTTPException(
            404,
            f"Model '{req.model_name}' not loaded. "
            f"Available: {list(loaded_models.keys())}"
        )

    bundle    = loaded_models[req.model_name]
    model     = bundle["model"]
    predictor = bundle["predictor"]
    config    = bundle["config"]

    with torch.no_grad():
        if config["type"] == "mlp":
            emb = model.encode(fingerprints.to(DEVICE))
        elif config.get("use_features") and fingerprints is not None:
            emb = model.encode(fingerprints.to(DEVICE), edge_index.to(DEVICE))
        else:
            emb = model.encode(None, edge_index.to(DEVICE))

        raw_score = predictor(
            emb[req.drug_a].unsqueeze(0),
            emb[req.drug_b].unsqueeze(0)
        ).item()
        probability = torch.sigmoid(torch.tensor(raw_score)).item()

    return generate_explanation(req.drug_a, req.drug_b, probability)

