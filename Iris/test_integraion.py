import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    """
    Test the home endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_success():
    """
    Test the predict endpoint with valid data.
    """
    response = client.post("/api/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_invalid_data():
    """
    Test the predict endpoint with invalid data.
    """
    response = client.post("/api/predict", json={
        "sepal_length": "invalid",
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 422

def test_health_check():
    """
    Test the health check endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}