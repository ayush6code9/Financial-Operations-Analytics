from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared import clean_table, money, number, pct
from app.ui import render_chart, render_kpi_grid, render_page_header, render_section_heading, render_table
from financial_ops.analytics import (
    churn_distribution,
    cohort_retention_long,
    executive_metrics,
    monthly_revenue_trend,
    profitability_by_customer_segment,
    segment_summary,
)
from financial_ops.data import load_customers, load_transactions

render_page_header(
    "Executive Overview",
    "A single operating view of revenue performance, churn risk, customer value, and profitability.",
    status="Live data",
    tone="info",
)

metrics = executive_metrics()
customers = load_customers()
transactions = load_transactions()
high_risk = int((metrics.get("High-Risk Customers") or 0))

kpis = [
    {"label": "Total Revenue", "value": money(metrics.get("Total Revenue")), "help": "Net revenue across all analyzed customers."},
    {"label": "Total Customers", "value": number(metrics.get("Total Customers", len(customers))), "help": "Active customers in the current portfolio."},
    {"label": "Transactions", "value": number(metrics.get("Total Transactions", len(transactions))), "help": "Recorded customer transactions in the source dataset."},
    {"label": "Churn Rate", "value": pct(metrics.get("Churn Rate"), already_percent=True), "help": "Observed churn rate across the customer base."},
    {"label": "High-Risk Customers", "value": number(high_risk), "help": "Customer count flagged as elevated risk by the model."},
    {"label": "Total Profit", "value": money(metrics.get("Total Profit")), "help": "Total net profit realized by the portfolio."},
    {"label": "Profit Margin", "value": pct(metrics.get("Profit Margin"), already_percent=True), "help": "Operating profit margin across the revenue stream."},
    {"label": "Forecast Revenue", "value": money(metrics.get("Forecast Revenue")), "help": "Projected revenue over the next forecast horizon."},
    {"label": "Average CLV", "value": money(metrics.get("Average CLV")), "help": "Average customer lifetime value across retained accounts."},
]
render_kpi_grid(kpis, columns=4)

revenue = monthly_revenue_trend()
fig_revenue = px.line(
    revenue,
    x="year_month",
    y="net_revenue",
    markers=True,
    labels={"year_month": "Month", "net_revenue": "Net Revenue"},
    line_shape="spline",
    color_discrete_sequence=["#1d4ed8"],
)
fig_revenue.update_layout(
    title="Monthly Revenue Trend",
    xaxis_title="Month",
    yaxis_title="Net revenue",
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=35, b=10),
    hovermode="x unified",
)
render_chart(fig_revenue, height=360)

left, right = st.columns(2)
segments = segment_summary().sort_values("segment_revenue", ascending=False)
fig_segments = px.bar(
    segments.head(10),
    x="segment_revenue",
    y="rfm_segment",
    orientation="h",
    color="revenue_share",
    color_continuous_scale=["#e2e8f0", "#0f766e"],
    labels={"segment_revenue": "Revenue", "rfm_segment": "RFM Segment", "revenue_share": "Revenue Share"},
    title="Revenue Contribution by RFM Segment",
)
fig_segments.update_layout(template="plotly_dark", height=360, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=35, b=10))
with left:
    render_section_heading("Revenue mix", "Largest customer value segments by revenue contribution")
    render_chart(fig_segments, height=360)

risk = churn_distribution()
fig_risk = px.bar(
    risk,
    x="risk_segment",
    y="customers",
    color="risk_segment",
    color_discrete_map={"Low Risk": "#16a34a", "Medium Risk": "#d97706", "High Risk": "#dc2626"},
    labels={"risk_segment": "Risk Segment", "customers": "Customers"},
    title="Predicted Churn Risk Distribution",
)
fig_risk.update_layout(template="plotly_dark", height=360, showlegend=False, margin=dict(l=10, r=10, t=35, b=10))
with right:
    render_section_heading("Churn mix", "Portfolio risk segmentation based on the live churn model")
    render_chart(fig_risk, height=360)

left2, right2 = st.columns(2)
profit = profitability_by_customer_segment()
fig_profit = px.bar(
    profit,
    x="customer_segment",
    y="total_profit",
    color="average_margin",
    color_continuous_scale=["#fef2f2", "#0f766e"],
    labels={"customer_segment": "Customer Segment", "total_profit": "Total Profit", "average_margin": "Avg Margin"},
    title="Profitability by Customer Segment",
)
fig_profit.update_layout(template="plotly_dark", height=360, xaxis_tickangle=-25, margin=dict(l=10, r=10, t=35, b=10))
with left2:
    render_section_heading("Profitability", "Gross contribution by customer segment")
    render_chart(fig_profit, height=360)

cohort = cohort_retention_long()
cohort = cohort[cohort["cohort_index"] <= 12]
cohort_avg = cohort.groupby("cohort_index", as_index=False)["retention_rate"].mean()
fig_cohort = px.line(
    cohort_avg,
    x="cohort_index",
    y="retention_rate",
    markers=True,
    labels={"cohort_index": "Customer month", "retention_rate": "Average retention"},
    title="Average Cohort Retention Curve",
    color_discrete_sequence=["#7c3aed"],
)
fig_cohort.update_layout(template="plotly_dark", height=360, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=35, b=10))
with right2:
    render_section_heading("Cohort recovery", "Observed retention curve across the first 12 months")
    render_chart(fig_cohort, height=360)

render_section_heading("Executive signals", "Top value segments from the current portfolio summary")
signals = clean_table(segments[["rfm_segment", "customers", "segment_revenue", "revenue_share"]].head(5))
render_table(signals, height=220)
