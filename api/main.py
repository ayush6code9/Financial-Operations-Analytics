from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from financial_ops.analytics import customer_360, json_ready
from financial_ops.churn_model import load_churn_artifact, model_metrics, predict_churn
from financial_ops.forecasting import forecast_kpis, forecast_model_comparison, forecast_results, historical_revenue

app = FastAPI(
    title="Financial Operations Analytics API",
    version="1.0.0",
    description="Customer churn, customer 360, model metrics, and revenue forecast endpoints.",
)


class ChurnPredictionRequest(BaseModel):
    customer_id: str | None = Field(default=None, description="Existing customer ID used as the default profile.")
    features: dict[str, Any] = Field(default_factory=dict, description="Actual raw model feature overrides.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/churn")
def predict_churn_endpoint(request: ChurnPredictionRequest) -> dict[str, Any]:
    try:
        artifact = load_churn_artifact()
        result = predict_churn(features=request.features, customer_id=request.customer_id, artifact=artifact)
        return json_ready(result)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/customer/{customer_id}")
def customer_endpoint(customer_id: str) -> dict[str, Any]:
    try:
        return json_ready(customer_360(customer_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/model/metrics")
def model_metrics_endpoint() -> dict[str, Any]:
    return json_ready(model_metrics())


@app.get("/forecast/revenue")
def forecast_endpoint(horizon: int = Query(default=12, ge=1, le=12)) -> dict[str, Any]:
    return json_ready(
        {
            "kpis": forecast_kpis(horizon),
            "historical": historical_revenue(),
            "forecast": forecast_results(horizon),
            "model_comparison": forecast_model_comparison(),
        }
    )
