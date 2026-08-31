from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data import load_monthly_revenue, load_output_csv, load_output_text


def historical_revenue() -> pd.DataFrame:
    data = load_monthly_revenue().copy()
    return data[["year_month", "net_revenue", "profit", "active_customers", "growth_rate"]]


def forecast_results(horizon: int = 12) -> pd.DataFrame:
    forecast = load_output_csv("forecast_results.csv").copy()
    forecast["year_month"] = pd.to_datetime(forecast["year_month"], errors="coerce")
    return forecast.head(max(1, min(int(horizon), len(forecast))))


def forecast_summary() -> str:
    return load_output_text("forecast_summary.txt")


def forecast_kpis(horizon: int = 12) -> dict[str, Any]:
    historical = historical_revenue()
    forecast = forecast_results(horizon)
    return {
        "historical_months": int(len(historical)),
        "forecast_horizon": int(len(forecast)),
        "last_observed_revenue": float(historical["net_revenue"].iloc[-1]),
        "projected_revenue": float(forecast["forecasted_net_revenue"].sum()),
        "average_forecast": float(forecast["forecasted_net_revenue"].mean()),
    }


def forecast_model_comparison() -> pd.DataFrame:
    """Return saved comparison if present; otherwise compute the notebook's baseline MAE only."""
    try:
        return load_output_csv("forecast_model_comparison.csv")
    except FileNotFoundError:
        history = historical_revenue().set_index("year_month")["net_revenue"].astype(float)
        test_size = 6
        train = history.iloc[:-test_size]
        test = history.iloc[-test_size:]
        baseline = pd.Series([train.iloc[-1]] * len(test), index=test.index)
        mae = float(np.mean(np.abs(test - baseline)))
        return pd.DataFrame(
            [
                {
                    "model": "Baseline",
                    "mae": mae,
                    "notes": "Computed from the notebook's six-month holdout rule.",
                },
                {
                    "model": "ARIMA",
                    "mae": np.nan,
                    "notes": "Evaluated in notebook; install statsmodels or regenerate outputs to persist MAE.",
                },
            ]
        )
