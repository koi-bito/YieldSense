"""Unit and integration tests for FastAPI service endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_model_manager
from src.api.main import app
from src.ingestion.loader import load_from_duckdb


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client with preloaded artifacts."""
    manager = get_model_manager()
    manager.load_artifacts()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_sensor_dict():
    """Extract a real lot's sensor dictionary from DuckDB."""
    df = load_from_duckdb("clean_lots")
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    first_lot = df.iloc[0]
    return {s: float(first_lot[s]) for s in sensor_cols}


def test_health_endpoint(client):
    """Verify /health reports healthy status and loaded artifacts."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True
    assert data["expected_sensors_count"] == 442


def test_model_info_endpoint(client):
    """Verify /model-info returns active model configuration and metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "XGBClassifier"
    assert "evaluation_metrics" in data
    assert "best_hyperparameters" in data


def test_analyze_lot_valid(client, sample_sensor_dict):
    """Verify /analyze-lot processes valid lot and returns full SHAP breakdown."""
    payload = {
        "lot_id": "TEST_LOT_001",
        "sensors": sample_sensor_dict,
    }
    response = client.post("/analyze-lot", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["lot_id"] == "TEST_LOT_001"
    assert data["prediction"] in ["PASS", "FAIL"]
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert 0.0 <= data["recovery_score"] <= 1.0
    assert isinstance(data["recovery_candidate"], bool)
    assert len(data["top_sensors"]) == 5
    assert "sensor" in data["top_sensors"][0]
    assert "shap_value" in data["top_sensors"][0]


def test_analyze_lot_missing_sensors_error(client):
    """Verify /analyze-lot rejects requests with too many missing sensor readings."""
    payload = {
        "lot_id": "TEST_LOT_EMPTY",
        "sensors": {"sensor_001": 1.0},  # Missing 441 of 442 sensors
    }
    response = client.post("/analyze-lot", json=payload)
    assert response.status_code == 422
    assert "Too many missing sensors" in response.json()["detail"]


def test_batch_analyze(client, sample_sensor_dict):
    """Verify /batch-analyze processes multiple lots and returns ranked output."""
    payload = {
        "lots": [
            {"lot_id": "LOT_BATCH_1", "sensors": sample_sensor_dict},
            {"lot_id": "LOT_BATCH_2", "sensors": sample_sensor_dict},
        ]
    }
    response = client.post("/batch-analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["total_lots"] == 2
    assert len(data["results"]) == 2


def test_analyze_lot_with_acceptable_missing_sensors(client, sample_sensor_dict):
    """Verify missing sensors (<20%) are imputed with training median and analysis succeeds."""
    # Omit 10 sensors (well within 20% limit)
    partial_sensors = dict(sample_sensor_dict)
    for k in list(partial_sensors.keys())[:10]:
        del partial_sensors[k]

    payload = {
        "lot_id": "TEST_LOT_IMPUTED",
        "sensors": partial_sensors,
    }
    response = client.post("/analyze-lot", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["lot_id"] == "TEST_LOT_IMPUTED"
    assert data["prediction"] in ["PASS", "FAIL"]
    assert len(data["top_sensors"]) > 0
