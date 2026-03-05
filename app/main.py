"""
DDI Prediction API — Explainable Drug-Drug Interaction Prediction

Endpoints:
    GET  /                  -> API info
    GET  /models            -> List available models
    POST /predict           -> Predict interaction probability
    POST /explain           -> Run explainability analysis
    GET  /health            -> Health check
"""
import os
import csv
import logging
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

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

# Global state
loaded_models = {}
fingerprints = None
edge_index = None
drug_smiles = {}


def load_graph_data():
    from ogb.linkproppred import PygLinkPropPredDataset
    logger.info("Loading OGB-DDI dataset...")
    dataset = PygLinkPropPredDataset(name="ogbl-ddi")
    split = dataset.get_edge_split()
    train_edges = split["train"]["edge"]
    if not isinstance(train_edges, torch.Tensor):
        train_edges = torch.from_numpy(train_edges)
    ei = torch.cat([train_edges.t(), train_edges.flip(1).t()], dim=1)
    logger.info(f"Graph loaded: {ei.shape[1]} edges")
    return ei


def load_smiles_mapping(path: str) -> dict:
    mapping = {}
    if not os.path.exists(path):
        logger.warning(f"SMILES mapping not found at {path}")
        return mapping
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_key = "node idx" if "node idx" in row else "node_idx"
            mapping[int(row[idx_key])] = row["smiles"]
    logger.info(f"Loaded SMILES for {len(mapping)} drugs")
    return mapping


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_models, fingerprints, edge_index, drug_smiles

    logger.info("=" * 60)
    logger.info("Starting DDI Prediction API")
    logger.info("=" * 60)

    # Load fingerprints
    if os.path.exists(FINGERPRINTS_PATH):
        fingerprints = torch.load(FINGERPRINTS_PATH, map_location=DEVICE, weights_only=False)
        logger.info(f"Fingerprints loaded: {fingerprints.shape}")
    else:
        logger.warning(f"Fingerprints not found at {FINGERPRINTS_PATH}")

    # Load graph
    edge_index = load_graph_data()

    # Load SMILES
    drug_smiles = load_smiles_mapping(DRUG_MAPPING_PATH)

    # Load models
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
    # Initialize RAG
    init_rag()
    logger.info("API ready!")
    logger.info("=" * 60)

    yield
    loaded_models.clear()
    logger.info("Shutdown complete")


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

@app.get("/app")
async def serve_frontend():
    return FileResponse("static/index.html")



@app.get("/")
def root():
    return {
        "name": "DDI Prediction API",
        "models_loaded": list(loaded_models.keys()),
        "total_drugs": NUM_NODES,
        "endpoints": ["/predict", "/explain", "/models", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(loaded_models), "device": str(DEVICE)}


@app.get("/models", response_model=list[ModelInfo])
def list_models():
    result = []
    for name, config in MODELS.items():
        model_type = config["type"]
        use_features = config.get("use_features", False)
        methods = []
        if model_type in ("graphsage", "gat"):
            methods.append("perturbation")
        if model_type == "gat":
            methods.append("attention")
        if use_features:
            methods.append("integrated_gradients")
        result.append(ModelInfo(
            name=name, description=config["description"],
            type=model_type, use_features=use_features,
            available_methods=methods,
        ))
    return result


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model_name not in loaded_models:
        raise HTTPException(404, f"Model '{req.model_name}' not loaded. Available: {list(loaded_models.keys())}")

    bundle = loaded_models[req.model_name]
    model = bundle["model"]
    predictor = bundle["predictor"]
    config = bundle["config"]

    with torch.no_grad():
        is_mlp = config["type"] == "mlp"
        if is_mlp:
            emb = model.encode(fingerprints.to(DEVICE))
        elif config.get("use_features") and fingerprints is not None:
            emb = model.encode(fingerprints.to(DEVICE), edge_index.to(DEVICE))
        else:
            emb = model.encode(None, edge_index.to(DEVICE))

        raw_score = predictor(
            emb[req.drug_a].unsqueeze(0), emb[req.drug_b].unsqueeze(0)
        ).item()
        probability = torch.sigmoid(torch.tensor(raw_score)).item()

    return PredictResponse(
        drug_a=req.drug_a, drug_b=req.drug_b, model_name=req.model_name,
        probability=round(probability, 6), raw_score=round(raw_score, 6),
    )


@app.post("/explain")
def explain(req: ExplainRequest):
    if req.model_name not in loaded_models:
        raise HTTPException(404, f"Model '{req.model_name}' not loaded. Available: {list(loaded_models.keys())}")

    bundle = loaded_models[req.model_name]
    model = bundle["model"]
    predictor = bundle["predictor"]
    config = bundle["config"]
    method = req.method
    model_type = config["type"]
    use_features = config.get("use_features", False)

    # Validate method
    available_methods = []
    if model_type in ("graphsage", "gat"):
        available_methods.append("perturbation")
    if model_type == "gat":
        available_methods.append("attention")
    if use_features:
        available_methods.append("integrated_gradients")

    if method not in available_methods:
        raise HTTPException(400, f"Method '{method}' not available for {req.model_name}. Available: {available_methods}")

    result = {"model_name": req.model_name, "method": method,
              "drug_a": req.drug_a, "drug_b": req.drug_b}

    if method == "perturbation":
        result["analysis"] = perturbation_analysis(
            model, predictor, req.drug_a, req.drug_b,
            edge_index, DEVICE, fingerprints if use_features else None,
            top_k=req.top_k,
        )

    elif method == "attention":
        result["analysis"] = attention_analysis(
            model, req.drug_a, req.drug_b,
            edge_index, DEVICE, fingerprints if use_features else None,
            top_k=req.top_k,
        )

    elif method == "integrated_gradients":
        if fingerprints is None:
            raise HTTPException(500, "Fingerprints not loaded")

        ig_result = integrated_gradients(
            model, predictor, req.drug_a, req.drug_b,
            fingerprints, DEVICE,
            edge_index=edge_index if model_type != "mlp" else None,
            top_k=req.top_k,
        )
        result["analysis"] = ig_result

        if req.include_visualization:
            visualizations = {}
            for drug_key in ["drug_a", "drug_b"]:
                drug_idx = ig_result[drug_key]["node_id"]
                if drug_idx in drug_smiles:
                    smiles = drug_smiles[drug_idx]
                    all_attrs = (
                        ig_result[drug_key]["top_positive_bits"]
                        + ig_result[drug_key]["top_negative_bits"]
                    )
                    img_b64 = visualize_ig_on_molecule(smiles, all_attrs, drug_key)
                    if img_b64:
                        visualizations[drug_key] = {
                            "smiles": smiles, "image_base64": img_b64, "format": "png",
                        }
            if visualizations:
                result["visualizations"] = visualizations

    return result


@app.post("/rag")
def rag_explain(req: PredictRequest):
    """Generate natural language explanation using RAG pipeline."""
    if req.model_name not in loaded_models:
        raise HTTPException(404, f"Model '{req.model_name}' not loaded.")

    # First get prediction
    bundle = loaded_models[req.model_name]
    model = bundle["model"]
    predictor = bundle["predictor"]
    config = bundle["config"]

    with torch.no_grad():
        is_mlp = config["type"] == "mlp"
        if is_mlp:
            emb = model.encode(fingerprints.to(DEVICE))
        elif config.get("use_features") and fingerprints is not None:
            emb = model.encode(fingerprints.to(DEVICE), edge_index.to(DEVICE))
        else:
            emb = model.encode(None, edge_index.to(DEVICE))

        raw_score = predictor(
            emb[req.drug_a].unsqueeze(0), emb[req.drug_b].unsqueeze(0)
        ).item()
        probability = torch.sigmoid(torch.tensor(raw_score)).item()

    return generate_explanation(req.drug_a, req.drug_b, probability)


@app.get("/drugs")
def list_drugs():
    """Return drug index to name mapping for search."""
    import json, os
    path = "data/rag/idx_to_name.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
