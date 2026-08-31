from __future__ import annotations

import pandas as pd
import pytest

from app.shared import clean_table
from financial_ops.analytics import customer_360, executive_metrics
from financial_ops.churn_model import load_churn_artifact, predict_churn
from financial_ops.data import load_customers, load_monthly_revenue, load_transactions
from financial_ops.forecasting import forecast_kpis, forecast_results


def test_data_loading_shapes() -> None:
    assert load_customers().shape[0] == 20_000
    assert load_transactions().shape[0] == 329_202
    assert load_monthly_revenue().shape[0] == 36


def test_churn_model_prediction() -> None:
    artifact = load_churn_artifact(retrain_if_missing=True)
    result = predict_churn(customer_id="CUST_000003", artifact=artifact)
    assert result["model_name"] == "XGBoost"
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_level"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert result["features_used"]


def test_customer_360_lookup() -> None:
    profile = customer_360("CUST_000003")
    assert profile["customer"]["customer_id"] == "CUST_000003"
    assert "recommendation" in profile["health"]
    assert profile["recent_transactions"].shape[0] > 0


def test_forecast_outputs() -> None:
    forecast = forecast_results(6)
    kpis = forecast_kpis(6)
    assert len(forecast) == 6
    assert kpis["forecast_horizon"] == 6
    assert kpis["projected_revenue"] > 0


def test_executive_metrics() -> None:
    metrics = executive_metrics()
    assert float(metrics["Total Revenue"]) > 0
    assert int(metrics["Total Customers"]) == 20_000


def test_display_table_normalization_handles_currency_values() -> None:
    frame = pd.DataFrame({
        "customer_id": ["CUST_000001"],
        "net_revenue": ["$122,960.01"],
        "profit": [112.5],
        "status": ["Active"],
    })
    cleaned = clean_table(frame)
    assert cleaned["net_revenue"].iloc[0] == 122960.01
    assert pd.api.types.is_numeric_dtype(cleaned["net_revenue"])
    assert cleaned["status"].iloc[0] == "Active"


def test_api_endpoints() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    assert client.get("/health").status_code == 200

    prediction = client.post("/predict/churn", json={"customer_id": "CUST_000003"}).json()
    assert 0.0 <= prediction["churn_probability"] <= 1.0

    customer = client.get("/customer/CUST_000003").json()
    assert customer["customer"]["customer_id"] == "CUST_000003"

    metrics = client.get("/model/metrics").json()
    assert metrics["best_model"] == "XGBoost"

    forecast = client.get("/forecast/revenue?horizon=3").json()
    assert len(forecast["forecast"]) == 3
