from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .churn_model import existing_prediction_for_customer, predict_churn
from .data import load_customers, load_monthly_revenue, load_output_csv, load_transactions, metric_lookup


HIGH_VALUE_SEGMENTS = {"Champions", "Big Spenders", "Loyal High Value", "At Risk High Value"}


def executive_metrics() -> dict[str, Any]:
    customers = load_customers()
    transactions = load_transactions()
    metrics = metric_lookup("executive_kpis.csv")
    metrics.setdefault("Total Customers", int(len(customers)))
    metrics.setdefault("Total Transactions", int(len(transactions)))
    metrics.setdefault("Average Customer Value", float(customers["total_revenue"].mean()))
    metrics.setdefault("High-Risk Customers", int(len(load_output_csv("high_risk_customers.csv"))))
    return metrics


def monthly_revenue_trend() -> pd.DataFrame:
    return load_monthly_revenue().copy()


def segment_summary() -> pd.DataFrame:
    return load_output_csv("segment_summary.csv")


def profitability_by_customer_segment() -> pd.DataFrame:
    data = load_output_csv("customer_profitability.csv")
    return (
        data.groupby("customer_segment", observed=False)
        .agg(
            customers=("customer_id", "nunique"),
            total_revenue=("customer_revenue", "sum"),
            total_profit=("customer_profit", "sum"),
            average_margin=("customer_profit_margin", "mean"),
        )
        .reset_index()
        .sort_values("total_profit", ascending=False)
    )


def churn_distribution() -> pd.DataFrame:
    predictions = load_output_csv("churn_predictions.csv")
    return (
        predictions.groupby("risk_segment", observed=False)
        .agg(customers=("customer_id", "count"), average_probability=("churn_probability", "mean"))
        .reset_index()
    )


def cohort_retention_long() -> pd.DataFrame:
    return load_output_csv("customer_retention_long.csv")


def _score_band(value: float, p75: float, p25: float, high_label: str, mid_label: str, low_label: str) -> str:
    if value >= p75:
        return high_label
    if value <= p25:
        return low_label
    return mid_label


def customer_360(customer_id: str) -> dict[str, Any]:
    customers = load_customers()
    match = customers[customers["customer_id"].astype(str) == str(customer_id)]
    if match.empty:
        raise KeyError(f"Customer not found: {customer_id}")
    customer = match.iloc[0].to_dict()

    rfm = load_output_csv("rfm_table.csv")
    rfm_match = rfm[rfm["customer_id"].astype(str) == str(customer_id)]
    profitability = load_output_csv("customer_profitability.csv")
    profit_match = profitability[profitability["customer_id"].astype(str) == str(customer_id)]
    prediction = existing_prediction_for_customer(customer_id)
    live_prediction = predict_churn(customer_id=customer_id)

    transactions = load_transactions()
    customer_txn = transactions[transactions["customer_id"].astype(str) == str(customer_id)].copy()
    customer_txn["transaction_date"] = pd.to_datetime(customer_txn["transaction_date"], errors="coerce")
    customer_txn = customer_txn.sort_values("transaction_date", ascending=False)

    cohort_info: dict[str, Any] = {}
    if customer.get("acquisition_month"):
        cohort_summary = load_output_csv("cohort_summary.csv")
        label = str(customer["acquisition_month"])[:7]
        cohort_match = cohort_summary[cohort_summary["cohort_month_label"].astype(str) == label]
        if not cohort_match.empty:
            cohort_info = cohort_match.iloc[0].to_dict()

    revenue = float(customer.get("total_revenue", 0) or 0)
    profit = float(customer.get("total_profit", 0) or 0)
    revenue_quantiles = customers["total_revenue"].quantile([0.25, 0.75])
    profit_quantiles = customers["total_profit"].quantile([0.25, 0.75])
    risk = live_prediction["risk_level"]
    rfm_segment = str(rfm_match.iloc[0]["rfm_segment"]) if not rfm_match.empty else "Unavailable"
    value_band = _score_band(revenue, revenue_quantiles.loc[0.75], revenue_quantiles.loc[0.25], "High", "Moderate", "Low")
    profit_band = _score_band(profit, profit_quantiles.loc[0.75], profit_quantiles.loc[0.25], "High", "Moderate", "Low")

    if risk == "High Risk" and (value_band == "High" or rfm_segment in HIGH_VALUE_SEGMENTS):
        recommendation = "Prioritize retention intervention; the account combines meaningful value with elevated churn risk."
    elif rfm_segment in HIGH_VALUE_SEGMENTS and profit_band == "High":
        recommendation = "Protect and expand this customer through loyalty, renewal confidence, or premium support."
    elif value_band == "Low" and risk == "High Risk":
        recommendation = "Use low-cost retention outreach and investigate whether the customer fits the target segment."
    else:
        recommendation = live_prediction["recommendation"]

    return {
        "customer": customer,
        "rfm": rfm_match.iloc[0].to_dict() if not rfm_match.empty else {},
        "profitability": profit_match.iloc[0].to_dict() if not profit_match.empty else {},
        "existing_prediction": prediction,
        "live_prediction": live_prediction,
        "cohort": cohort_info,
        "recent_transactions": customer_txn.head(10),
        "health": {
            "value": value_band,
            "churn_risk": risk.replace(" Risk", ""),
            "profitability": profit_band,
            "rfm_segment": rfm_segment,
            "recommendation": recommendation,
        },
    }


def business_insights() -> dict[str, pd.DataFrame]:
    files = {
        "Executive": "executive_business_insights.csv",
        "Executive Recommendations": "executive_recommendations.csv",
        "Churn": "churn_business_insights.csv",
        "RFM": "rfm_business_insights.csv",
        "Cohort": "cohort_business_insights.csv",
        "Cohort Recommendations": "cohort_business_recommendations.csv",
        "Marketing Strategy": "marketing_strategy_by_segment.csv",
    }
    return {label: load_output_csv(path) for label, path in files.items()}


def json_ready(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.where(pd.notna(value), None).to_dict()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value
