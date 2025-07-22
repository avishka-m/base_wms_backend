import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
from .prophet_forecasting_fixed import router
import pandas as pd

# base_wms_backend/app/api/test_prophet_forecasting_fixed.py



app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def allow_manager(monkeypatch):
    monkeypatch.setattr(
        "base_wms_backend.app.api.prophet_forecasting_fixed.has_role",
        lambda roles: lambda: {"username": "test", "roles": ["Manager"]}
    )

def test_get_products(client, monkeypatch):
    csv_data = "product_id,category\n1,CatA\n2,CatB\n"
    m = mock_open(read_data=csv_data)
    monkeypatch.setattr("builtins.open", m)
    response = client.get("/products")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert {"id": "1", "category": "CatA"} in data["data"]
    assert {"id": "2", "category": "CatB"} in data["data"]

def test_get_categories(client, monkeypatch):
    csv_data = "product_id,category\n1,CatA\n2,CatB\n3,CatA\n"
    m = mock_open(read_data=csv_data)
    monkeypatch.setattr("builtins.open", m)
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert sorted(data["data"]) == ["CatA", "CatB"]

@patch("base_wms_backend.app.api.prophet_forecasting_fixed.ProphetCategoryPredictor")
@patch("base_wms_backend.app.api.prophet_forecasting_fixed.PROPHET_CONFIG", {"CatA": {"x": 1}})
def test_category_predict_success(mock_predictor, client, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = True
    today = datetime.now().date()
    future_dates = [today + timedelta(days=i) for i in range(3)]
    df = pd.DataFrame({
        "ds": pd.to_datetime([str(d) for d in future_dates]),
        "yhat": [10, 20, 30],
        "yhat_lower": [8, 18, 28],
        "yhat_upper": [12, 22, 32]
    })
    mock_instance.predict.return_value = df
    mock_predictor.return_value = mock_instance

    response = client.get("/category/CatA/predict?prediction_horizon_days=3")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["category"] == "CatA"
    assert len(data["forecast_data"]) == 3

@patch("base_wms_backend.app.api.prophet_forecasting_fixed.PROPHET_CONFIG", {})
def test_category_predict_no_config(client):
    response = client.get("/category/UnknownCat/predict")
    assert response.status_code == 404
    assert "No Prophet config found" in response.json()["detail"]

@patch("base_wms_backend.app.api.prophet_forecasting_fixed.PROPHET_CONFIG", {"CatA": {"x": 1}})
def test_category_predict_no_model_file(client, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: False)
    response = client.get("/category/CatA/predict")
    assert response.status_code == 404
    assert "No trained model found" in response.json()["detail"]

@patch("base_wms_backend.app.api.prophet_forecasting_fixed.ProphetCategoryPredictor")
@patch("base_wms_backend.app.api.prophet_forecasting_fixed.PROPHET_CONFIG", {"CatA": {"x": 1}})
def test_category_predict_model_load_fail(mock_predictor, client, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = False
    mock_predictor.return_value = mock_instance
    response = client.get("/category/CatA/predict")
    assert response.status_code == 500
    assert "Failed to load model" in response.json()["detail"]import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
from .prophet_forecasting_fixed import router
import pandas as pd

# base_wms_backend/app/api/test_prophet_forecasting_fixed.py



app = FastAPI()
app.include_router(router)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def allow_manager(monkeypatch):
    monkeypatch.setattr(
        "base_wms_backend.app.api.prophet_forecasting_fixed.has_role",
        lambda roles: lambda: {"username": "test", "roles": ["Manager"]}
    )

def test_get_products(client, monkeypatch):
    csv_data = "product_id,category\n1,CatA\n2,CatB\n"
    m = mock_open(read_data=csv_data)
    monkeypatch.setattr("builtins.open", m)
    response = client.get("/products")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert {"id": "1", "category": "CatA"} in data["data"]
    assert {"id": "2", "category": "CatB"} in data["data"]

def test_get_categories(client, monkeypatch):
    csv_data = "product_id,category\n1,CatA\n2,CatB\n3,CatA\n"
    m = mock_open(read_data=csv_data)
    monkeypatch.setattr("builtins.open", m)
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert sorted(data["data"]) == ["CatA", "CatB"]

@patch("base_wms_backend.app.api.prophet_forecasting_fixed.ProphetCategoryPredictor")
@patch("base_wms_backend.app.api.prophet_forecasting_fixed.PROPHET_CONFIG", {"CatA": {"x": 1}})
def test_category_predict_success(mock_predictor, client, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = True
    today = datetime.now().date()
    future_dates = [today + timedelta(days=i) for i in range(3)]
    df = pd.DataFrame({
        "ds": pd.to_datetime([str(d) for d in future_dates]),
        "yhat": [10, 20, 30],
        "yhat_lower": [8, 18, 28],
        "yhat_upper": [12, 22, 32]
    })
    mock_instance.predict.return_value = df
    mock_predictor.return_value = mock_instance

    response = client.get("/category/CatA/predict?prediction_horizon_days=3")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["category"] == "CatA"
    assert len(data["forecast_data"]) == 3

@patch("base_wms_backend.app.api.prophet_forecasting_fixed.PROPHET_CONFIG", {})
def test_category_predict_no_config(client):
    response = client.get("/category/UnknownCat/predict")
    assert response.status_code == 404
    assert "No Prophet config found" in response.json()["detail"]

@patch("base_wms_backend.app.api.prophet_forecasting_fixed.PROPHET_CONFIG", {"CatA": {"x": 1}})
def test_category_predict_no_model_file(client, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: False)
    response = client.get("/category/CatA/predict")
    assert response.status_code == 404
    assert "No trained model found" in response.json()["detail"]

@patch("base_wms_backend.app.api.prophet_forecasting_fixed.ProphetCategoryPredictor")
@patch("base_wms_backend.app.api.prophet_forecasting_fixed.PROPHET_CONFIG", {"CatA": {"x": 1}})
def test_category_predict_model_load_fail(mock_predictor, client, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)
    mock_instance = MagicMock()
    mock_instance.load_model.return_value = False
    mock_predictor.return_value = mock_instance
    response = client.get("/category/CatA/predict")
    assert response.status_code == 500
    assert "Failed to load model" in response.json()["detail"]