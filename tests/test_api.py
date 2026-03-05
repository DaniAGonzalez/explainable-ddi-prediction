"""
Tests for DDI Prediction API endpoints.
Tests against the running API at localhost:8000.

Run with: pytest tests/ -v
"""
import pytest
import httpx

BASE = "http://localhost:8000"


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE, timeout=120)


class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
        assert r.json()["models_loaded"] == 7

    def test_root_returns_endpoints(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "/predict" in r.json()["endpoints"]

    def test_models_list(self, client):
        r = client.get("/models")
        assert r.status_code == 200
        assert len(r.json()) == 7


class TestPredict:
    def test_predict_valid_pair(self, client):
        r = client.post("/predict", json={"drug_a": 0, "drug_b": 1, "model_name": "mlp_baseline"})
        assert r.status_code == 200
        assert 0.0 <= r.json()["probability"] <= 1.0

    def test_predict_all_models(self, client):
        for m in client.get("/models").json():
            r = client.post("/predict", json={"drug_a": 10, "drug_b": 20, "model_name": m["name"]})
            assert r.status_code == 200, f"Failed for {m['name']}"

    def test_predict_invalid_drug_index(self, client):
        r = client.post("/predict", json={"drug_a": 99999, "drug_b": 0, "model_name": "mlp_baseline"})
        assert r.status_code == 422

    def test_predict_negative_index(self, client):
        r = client.post("/predict", json={"drug_a": -1, "drug_b": 0, "model_name": "mlp_baseline"})
        assert r.status_code == 422

    def test_predict_unknown_model(self, client):
        r = client.post("/predict", json={"drug_a": 0, "drug_b": 1, "model_name": "nonexistent"})
        assert r.status_code == 404

    def test_predict_same_drug(self, client):
        r = client.post("/predict", json={"drug_a": 50, "drug_b": 50, "model_name": "mlp_baseline"})
        assert r.status_code == 200


class TestExplain:
    def test_perturbation_graphsage(self, client):
        r = client.post("/explain", json={"drug_a": 0, "drug_b": 1, "model_name": "graphsage_baseline", "method": "perturbation", "top_k": 5})
        assert r.status_code == 200

    def test_attention_gat(self, client):
        r = client.post("/explain", json={"drug_a": 0, "drug_b": 1, "model_name": "gat_base", "method": "attention", "top_k": 5})
        assert r.status_code == 200

    def test_ig_mlp(self, client):
        r = client.post("/explain", json={"drug_a": 0, "drug_b": 1, "model_name": "mlp_baseline", "method": "integrated_gradients", "top_k": 5, "include_visualization": False})
        assert r.status_code == 200

    def test_invalid_method_for_model(self, client):
        r = client.post("/explain", json={"drug_a": 0, "drug_b": 1, "model_name": "mlp_baseline", "method": "perturbation"})
        assert r.status_code == 400

    def test_attention_on_graphsage(self, client):
        r = client.post("/explain", json={"drug_a": 0, "drug_b": 1, "model_name": "graphsage_baseline", "method": "attention"})
        assert r.status_code == 400
