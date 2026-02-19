import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    with TestClient(app) as client:
        yield client

def test_home(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Wine Quality AI" in response.text

def test_predict_page(client):
    """Test the predict page endpoint"""
    response = client.get("/predict")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Predict Wine Quality" in response.text

def test_predict_endpoint(client):
    """Test the predict endpoint with valid data"""
    response = client.post("/predict", json={
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    })
    assert response.status_code == 200
    assert "predicted_quality" in response.json()

def test_predict_invalid_data(client):
    """Test the predict endpoint with invalid data"""
    response = client.post("/predict", json={
        "fixed_acidity": "invalid",
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    })
    assert response.status_code == 422
