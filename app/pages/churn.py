from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared import clean_table, customer_ids, get_churn_artifact, pct
from app.ui import render_chart, render_kpi_grid, render_page_header, render_section_heading, render_table
from financial_ops.churn_model import feature_row_for_customer, model_metrics, predict_churn
from financial_ops.data import load_output_csv

render_page_header(
    "Customer Churn Prediction",
    "Live churn inference using the retained XGBoost pipeline and the underlying business features.",
    status="Model active",
    tone="warning",
)

artifact = get_churn_artifact()
ids = customer_ids()
selected_id = st.selectbox("Customer ID", ids, index=0)
base_row = feature_row_for_customer(selected_id, artifact)

info = st.columns(2)
with info[0]:
    st.caption(f"Model: {artifact['model_name']}")
with info[1]:
    st.caption(f"Live feature set: {len(artifact['feature_columns'])} inputs")

NUMERIC_INPUTS = [
    "days_since_last_transaction",
    "tenure_months",
    "usage_score",
    "login_frequency",
    "nps_score",
    "support_tickets",
    "monthly_recurring_revenue",
    "total_transactions",
    "total_revenue",
    "total_profit",
    "average_transaction_value",
    "customer_lifetime_value",
    "discount_percentage",
]
CATEGORICAL_INPUTS = [
    "subscription_plan",
    "contract_type",
    "company_size",
    "industry",
    "acquisition_channel",
    "payment_method",
    "country",
    "region",
]


def numeric_value(row: pd.Series, column: str) -> float:
    try:
        return float(row.get(column, artifact["defaults"].get(column, 0)))
    except (TypeError, ValueError):
        return 0.0


def categorical_value(row: pd.Series, column: str) -> str:
    value = str(row.get(column, artifact["defaults"].get(column, "")))
    options = artifact["categorical_options"].get(column, [])
    if value not in options:
        options = [value, *options]
    return value

with st.form("churn_prediction_form"):
    overrides: dict[str, Any] = {}
    left, middle, right = st.columns(3)
    numeric_columns = [left, middle, right]
    for index, column in enumerate(NUMERIC_INPUTS):
        container = numeric_columns[index % 3]
        step = 1.0 if column in {"days_since_last_transaction", "tenure_months", "total_transactions"} else 0.1
        overrides[column] = container.number_input(
            column.replace("_", " ").title(),
            value=numeric_value(base_row, column),
            step=step,
        )

    cat_left, cat_right = st.columns(2)
    for index, column in enumerate(CATEGORICAL_INPUTS):
        options = artifact["categorical_options"].get(column, [])
        current = categorical_value(base_row, column)
        if current not in options:
            options = [current, *options]
        target = cat_left if index % 2 == 0 else cat_right
        overrides[column] = target.selectbox(
            column.replace("_", " ").title(),
            options=options,
            index=options.index(current) if current in options else 0,
        )

    submitted = st.form_submit_button("Predict Churn")

result = predict_churn(features=overrides, customer_id=selected_id, artifact=artifact) if submitted else predict_churn(customer_id=selected_id, artifact=artifact)

render_kpi_grid(
    [
        {"label": "Churn Probability", "value": pct(result["churn_probability"])},
        {"label": "Risk Level", "value": result["risk_level"]},
        {"label": "Predicted Churn", "value": "Yes" if result["predicted_churn"] else "No"},
        {"label": "Model Used", "value": result["model_name"]},
    ],
    columns=4,
)

render_section_heading("Business interpretation", "Primary churn drivers and the recommended intervention path")
for factor in result["factors"]:
    st.write(f"- {factor}")
st.write(result["recommendation"])

predictions = load_output_csv("churn_predictions.csv")
risk_summary = (
    predictions.groupby("risk_segment", observed=False)
    .agg(customers=("customer_id", "count"), average_probability=("churn_probability", "mean"))
    .reset_index()
)
fig = px.bar(
    risk_summary,
    x="risk_segment",
    y="customers",
    color="risk_segment",
    color_discrete_map={"Low Risk": "#16a34a", "Medium Risk": "#d97706", "High Risk": "#dc2626"},
    title="Portfolio Risk Queue",
    labels={"risk_segment": "Risk Segment", "customers": "Customers"},
)
fig.update_layout(template="plotly_dark", height=340, showlegend=False, margin=dict(l=10, r=10, t=35, b=10))
render_chart(fig, height=340)

metrics = model_metrics()
model_table = metrics["metrics"].copy()
render_section_heading("Model performance", "Evaluation metrics from the retained churn model ensemble")
render_table(model_table, height=220)

with st.expander("Actual raw model features"):
    feature_table = pd.DataFrame(
        {
            "feature": artifact["feature_columns"],
            "type": [
                "numeric" if feature in artifact["numeric_features"] else "categorical"
                for feature in artifact["feature_columns"]
            ],
        }
    )
    render_table(feature_table, height=260)
