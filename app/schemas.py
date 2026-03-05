"""Pydantic models for API request/response validation."""
from pydantic import BaseModel, Field
from typing import Optional


class PredictRequest(BaseModel):
    drug_a: int = Field(..., ge=0, lt=4267, description="Drug A node index (0-4266)")
    drug_b: int = Field(..., ge=0, lt=4267, description="Drug B node index (0-4266)")
    model_name: str = Field(
        default="gat_skip_feat",
        description="Model to use for prediction",
    )


class PredictResponse(BaseModel):
    drug_a: int
    drug_b: int
    model_name: str
    probability: float
    raw_score: float


class ExplainRequest(BaseModel):
    drug_a: int = Field(..., ge=0, lt=4267)
    drug_b: int = Field(..., ge=0, lt=4267)
    model_name: str = Field(default="gat_skip_feat")
    method: str = Field(
        default="perturbation",
        description="Explainability method: perturbation, attention, integrated_gradients",
    )
    top_k: int = Field(default=10, ge=1, le=50)
    include_visualization: bool = Field(
        default=True,
        description="Include molecular structure images (for IG only)",
    )


class ModelInfo(BaseModel):
    name: str
    description: str
    type: str
    use_features: bool
    available_methods: list[str]
