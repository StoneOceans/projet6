from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def test_root_route():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "API de scoring opérationnelle"
    assert data["documentation"] == "/docs"
    assert data["health"] == "/health"


def test_health_route():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"


def test_invalid_payload_without_features():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"wrong_field": {}},
        )

    assert response.status_code == 422


def test_empty_features_are_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": {}},
        )

    assert response.status_code == 422


@patch("app.main.predict_client")
def test_valid_prediction(mock_predict_client):
    mock_predict_client.return_value = {
        "prediction": 0,
        "probability": 0.20,
        "threshold": 0.48,
        "decision": "Crédit accordé",
        "inference_time_ms": 3.5,
        "model_version": "1.0.0",
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "EXT_SOURCE_3": 0.75,
                    "EXT_SOURCE_2": 0.65,
                }
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 0
    assert data["probability"] == 0.20
    assert data["threshold"] == 0.48
    assert data["decision"] == "Crédit accordé"
    assert data["model_version"] == "1.0.0"
    assert "request_id" in data


@patch("app.main.predict_client")
def test_refused_credit_prediction(mock_predict_client):
    mock_predict_client.return_value = {
        "prediction": 1,
        "probability": 0.80,
        "threshold": 0.48,
        "decision": "Crédit refusé",
        "inference_time_ms": 4.0,
        "model_version": "1.0.0",
    }

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "features": {
                    "EXT_SOURCE_3": 0.10,
                    "EXT_SOURCE_2": 0.15,
                }
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["probability"] == 0.80
    assert data["decision"] == "Crédit refusé"


def test_monitoring_route():
    with TestClient(app) as client:
        response = client.get(
            "/monitoring/predictions?limit=10"
        )

    assert response.status_code == 200

    data = response.json()

    assert "count" in data
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_invalid_monitoring_limit():
    with TestClient(app) as client:
        response = client.get(
            "/monitoring/predictions?limit=0"
        )

    assert response.status_code == 422
