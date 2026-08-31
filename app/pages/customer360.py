from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared import clean_table, customer_ids, money, number, pct
from app.ui import render_kpi_grid, render_page_header, render_section_heading, render_table
from financial_ops.analytics import customer_360

render_page_header(
    "Customer 360",
    "Customer-level health, lifecycle value, profitability, and transaction profile in a single operating view.",
    status="Portfolio view",
    tone="success",
)

selected_id = st.selectbox("Customer ID", customer_ids(), index=0)
profile = customer_360(selected_id)
customer = profile["customer"]
rfm = profile["rfm"]
profit = profile["profitability"]
prediction = profile["existing_prediction"] or profile["live_prediction"]
health = profile["health"]

render_kpi_grid(
    [
        {"label": "Customer Revenue", "value": money(customer.get("total_revenue"))},
        {"label": "Total Profit", "value": money(customer.get("total_profit"))},
        {"label": "Transactions", "value": number(customer.get("total_transactions"))},
        {"label": "Churn Probability", "value": pct(prediction.get("churn_probability"))},
        {"label": "RFM Segment", "value": health["rfm_segment"]},
    ],
    columns=5,
)

render_section_heading("Customer health", health["recommendation"])
health_cols = st.columns(4)
health_cols[0].metric("Value", health["value"])
health_cols[1].metric("Churn Risk", health["churn_risk"])
health_cols[2].metric("Profitability", health["profitability"])
health_cols[3].metric("Risk Segment", prediction.get("risk_segment", profile["live_prediction"]["risk_level"]))

left, right = st.columns(2)
with left:
    render_section_heading("Customer attributes", "Recorded profile and account setup details")
    attribute_fields = [
        "customer_id",
        "country",
        "region",
        "industry",
        "company_size",
        "subscription_plan",
        "contract_type",
        "acquisition_channel",
        "tenure_months",
        "usage_score",
        "login_frequency",
        "nps_score",
    ]
    render_table(pd.DataFrame([{"metric": field, "value": customer.get(field)} for field in attribute_fields]), height=280)

with right:
    render_section_heading("RFM and profitability", "Account economics and segment behavior")
    rfm_profit = [
        {"metric": "Recency", "value": rfm.get("recency")},
        {"metric": "Frequency", "value": rfm.get("frequency")},
        {"metric": "Monetary", "value": money(rfm.get("monetary"), compact=False) if rfm else None},
        {"metric": "RFM Score", "value": rfm.get("rfm_score")},
        {"metric": "Profit Margin", "value": pct(profit.get("customer_profit_margin")) if profit else None},
        {"metric": "Customer Cost", "value": money(profit.get("customer_cost"), compact=False) if profit else None},
    ]
    render_table(pd.DataFrame(rfm_profit), height=280)

cohort = profile["cohort"]
if cohort:
    render_section_heading("Cohort context", "Acquisition cohort performance and retention")
    cohort_cols = st.columns(4)
    cohort_cols[0].metric("Cohort", str(cohort.get("cohort_month_label")))
    cohort_cols[1].metric("Cohort Size", number(cohort.get("cohort_size")))
    cohort_cols[2].metric("Month 3 Retention", pct(cohort.get("month_3_retention")))
    cohort_cols[3].metric("Average Retention", pct(cohort.get("average_customer_retention")))

render_section_heading("Recent transactions", "Most recent customer activity and commercial behavior")
recent = profile["recent_transactions"]
display_columns = [
    "transaction_date",
    "transaction_category",
    "billing_cycle",
    "net_revenue",
    "profit",
    "transaction_status",
    "payment_delay_days",
]
existing_columns = [column for column in display_columns if column in recent.columns]
render_table(recent[existing_columns], height=260)
