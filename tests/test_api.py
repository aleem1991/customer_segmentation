import os
import sys
from fastapi.testclient import TestClient

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.api import app

def test_health_check() -> None:
    """Verifies that the /health API endpoint responds with success status."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

def test_predict_endpoint() -> None:
    """Verifies that a valid customer prediction payload yields correct structure and values."""
    payload = {
        "recency": 10,
        "frequency": 5,
        "monetary": 250.0,
        "basket_size": 8.0,
        "avg_days_between": 20.0,
        "recent_orders_ratio": 0.8,
        "is_uk": 1
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "risk_tier" in data
        assert "recommendation" in data
        assert isinstance(data["churn_probability"], float)

def test_predict_batch_endpoint() -> None:
    """Verifies that a valid list of customer profiles returns batched prediction arrays."""
    payload = {
        "customers": [
            {
                "recency": 10,
                "frequency": 5,
                "monetary": 250.0,
                "basket_size": 8.0
            },
            {
                "recency": 200,
                "frequency": 1,
                "monetary": 50.0,
                "basket_size": 2.0
            }
        ]
    }
    with TestClient(app) as client:
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
        for pred in data["predictions"]:
            assert "churn_probability" in pred
            assert "risk_tier" in pred
            assert "recommendation" in pred

def test_monitor_drift_endpoint() -> None:
    """Verifies that the /monitor endpoint returns the statistical drift analysis report."""
    with TestClient(app) as client:
        response = client.get("/monitor")
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "drift_status" in data
        assert "message" in data

def test_shadow_stats_endpoint() -> None:
    """Verifies that the /shadow_stats endpoint returns Champion vs Challenger metrics."""
    with TestClient(app) as client:
        response = client.get("/shadow_stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "champion_mean" in data
        assert "challenger_mean" in data
        assert "mean_absolute_deviation" in data
        assert "agreement_rate" in data
        assert "recent_logs" in data
