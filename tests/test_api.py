"""
Tests for DDI Prediction API endpoints.

Integration tests that run against the live Docker API at localhost:8000.
All 7 models must be loaded and the API must be running before executing.

Note on coverage: pytest-cov reports 0% local coverage because tests call
the API over HTTP rather than importing application code directly. This is
an acknowledged limitation — the API runs inside Docker and cannot be
imported as a module in the test environment. Functional correctness is
validated end-to-end through HTTP assertions.

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # shorter tracebacks
"""

import pytest
import httpx

BASE = "http://localhost:8000"


@pytest.fixture(scope="module")
def client():
    """Shared HTTP client for all tests in this module.

    scope='module' means one client instance is created per test file,
    avoiding repeated connection setup overhead across test classes.
    timeout=120 accommodates slow model inference on CPU fallback.
    """
    return httpx.Client(base_url=BASE, timeout=120)


# ---------------------------------------------------------------------------
# Health and metadata endpoints
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for health check and metadata endpoints."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 with all 7 models loaded."""
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
        assert r.json()["models_loaded"] == 7

    def test_root_returns_endpoints(self, client):
        """Root endpoint lists all available API routes."""
        r = client.get("/")
        assert r.status_code == 200
        assert "/predict" in r.json()["endpoints"]

    def test_models_list(self, client):
        """Models endpoint returns all 7 configured model variants."""
        r = client.get("/models")
        assert r.status_code == 200
        assert len(r.json()) == 7


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

class TestPredict:
    """Tests for the /predict endpoint covering valid inputs and edge cases."""

    def test_predict_valid_pair(self, client):
        """Valid drug pair returns probability in [0, 1]."""
        r = client.post("/predict", json={
            "drug_a": 0, "drug_b": 1, "model_name": "mlp_baseline"
        })
        assert r.status_code == 200
        assert 0.0 <= r.json()["probability"] <= 1.0

    def test_predict_all_models(self, client):
        """All 7 models return 200 for the same drug pair.

        Verifies that every model checkpoint loaded correctly and can
        run inference without error.
        """
        for m in client.get("/models").json():
            r = client.post("/predict", json={
                "drug_a": 10, "drug_b": 20, "model_name": m["name"]
            })
            assert r.status_code == 200, (
                f"Model '{m['name']}' failed with status {r.status_code}: {r.text}"
            )

    def test_predict_invalid_drug_index(self, client):
        """Drug index exceeding the graph size returns 422 (validation error)."""
        r = client.post("/predict", json={
            "drug_a": 99999, "drug_b": 0, "model_name": "mlp_baseline"
        })
        assert r.status_code == 422

    def test_predict_negative_index(self, client):
        """Negative drug index returns 422 (Pydantic rejects it)."""
        r = client.post("/predict", json={
            "drug_a": -1, "drug_b": 0, "model_name": "mlp_baseline"
        })
        assert r.status_code == 422

    def test_predict_unknown_model(self, client):
        """Unknown model name returns 404."""
        r = client.post("/predict", json={
            "drug_a": 0, "drug_b": 1, "model_name": "nonexistent"
        })
        assert r.status_code == 404

    def test_predict_same_drug(self, client):
        """Self-interaction (drug_a == drug_b) is accepted and returns a score.

        The API does not reject self-pairs — this is a valid query that
        returns a meaningful score for some architectures.
        """
        r = client.post("/predict", json={
            "drug_a": 50, "drug_b": 50, "model_name": "mlp_baseline"
        })
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Explainability endpoint
# ---------------------------------------------------------------------------

class TestExplain:
    """Tests for the /explain endpoint covering all three ExAI methods
    and invalid method-model combinations."""

    def test_perturbation_graphsage(self, client):
        """Perturbation analysis runs successfully on a GraphSAGE model."""
        r = client.post("/explain", json={
            "drug_a": 0, "drug_b": 1,
            "model_name": "graphsage_baseline",
            "method": "perturbation",
            "top_k": 5,
        })
        assert r.status_code == 200

    def test_attention_gat(self, client):
        """Attention weight extraction runs successfully on a GAT model."""
        r = client.post("/explain", json={
            "drug_a": 0, "drug_b": 1,
            "model_name": "gat_base",
            "method": "attention",
            "top_k": 5,
        })
        assert r.status_code == 200

    def test_ig_mlp(self, client):
        """Integrated Gradients runs successfully on the MLP baseline."""
        r = client.post("/explain", json={
            "drug_a": 0, "drug_b": 1,
            "model_name": "mlp_baseline",
            "method": "integrated_gradients",
            "top_k": 5,
            "include_visualization": False,
        })
        assert r.status_code == 200

    def test_invalid_method_for_model(self, client):
        """Requesting perturbation on MLP returns 400 — method not available.

        MLP has no graph topology, so perturbation analysis is not supported.
        Only integrated_gradients is available for the MLP baseline.
        """
        r = client.post("/explain", json={
            "drug_a": 0, "drug_b": 1,
            "model_name": "mlp_baseline",
            "method": "perturbation",
        })
        assert r.status_code == 400

    def test_attention_on_graphsage(self, client):
        """Requesting attention on GraphSAGE returns 400 — GAT-only method.

        Attention weights are only available for GAT variants, which have
        explicit attention coefficients. GraphSAGE uses mean aggregation
        with no learnable attention.
        """
        r = client.post("/explain", json={
            "drug_a": 0, "drug_b": 1,
            "model_name": "graphsage_baseline",
            "method": "attention",
        })
        assert r.status_code == 400
