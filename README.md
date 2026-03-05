---
title: DDI Prediction
emoji: 💊
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# Explainable DDI Prediction

![CI](https://github.com/DaniAGonzalez/explainable-ddi-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

Explainable drug-drug interaction prediction using graph neural networks with biological validation. Predicts whether two drugs interact and **explains why** using three complementary methods.

## Key Features

- **7 model variants**: MLP baseline, GraphSAGE (×2), GAT (×4)
- **3 explainability methods**: perturbation analysis, attention weights, integrated gradients
- **Biological validation**: Fisher's exact test against DrugBank CYP enzyme annotations
- **RAG pipeline**: natural language explanations using DrugBank + PubChem + DailyMed + Claude API
- **Interactive frontend**: DDI Explorer with molecular visualizations

## Quick Start
```bash
git clone https://github.com/DaniAGonzalez/explainable-ddi-prediction.git
cd explainable-ddi-prediction
docker-compose up
```

API at `http://localhost:8000` — open `DDI_Explorer.html` in browser.

## API Endpoints

| Method | Endpoint   | Description                                |
|--------|------------|--------------------------------------------|
| GET    | /          | API info and loaded models                 |
| GET    | /health    | Health check (status, model count)         |
| GET    | /models    | List all models with available XAI methods |
| POST   | /predict   | Predict interaction probability            |
| POST   | /explain   | Run explainability analysis                |

### Example: Predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"drug_a": 0, "drug_b": 1, "model_name": "gat_skip_feat"}'
```
```json
{
  "drug_a": 0,
  "drug_b": 1,
  "model_name": "gat_skip_feat",
  "probability": 0.872341,
  "raw_score": 1.920145
}
```

## Models

| Model             | AUC    | XAI Methods                          |
|-------------------|--------|--------------------------------------|
| MLP Baseline      | 96.9%  | Integrated Gradients                 |
| GraphSAGE Topo    | 98.1%  | Perturbation                         |
| GraphSAGE Feat    | 97.6%  | Perturbation, Integrated Gradients   |
| GAT Base          | 93.5%  | Perturbation, Attention              |
| GAT Skip          | 94.8%  | Perturbation, Attention              |
| GAT Base+Feat     | 91.9%  | Perturbation, Attention, IG          |
| GAT Skip+Feat     | 97.5%  | Perturbation, Attention, IG          |

## Project Structure
```
explainable-ddi-prediction/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Model configurations
│   ├── schemas.py              # Request/response validation
│   ├── middleware.py            # Request logging
│   ├── models/                 # GraphSAGE, GAT, MLP encoders
│   └── explainability/         # Perturbation, attention, IG
├── checkpoints/                # 7 trained model weights (.pt)
├── data/                       # Fingerprints, SMILES, DrugBank
├── tests/                      # pytest API tests
├── logs/                       # Request logs (auto-created)
├── .github/workflows/ci.yml   # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Testing
```bash
pytest tests/ -v
```

## Tech Stack

**ML**: PyTorch, PyTorch Geometric, OGB  
**API**: FastAPI, Pydantic  
**XAI**: Perturbation, Attention, Integrated Gradients  
**RAG**: ChromaDB, sentence-transformers, Claude API  
**Data**: DrugBank, PubChem, DailyMed  
**Infra**: Docker, GitHub Actions

## Author

Dr. Daniela A. Gonzalez
