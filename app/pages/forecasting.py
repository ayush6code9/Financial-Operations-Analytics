from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared import money
from app.ui import render_chart, render_kpi_grid, render_page_header, render_section_heading, render_table
from financial_ops.forecasting import (
    forecast_kpis,
    forecast_model_comparison,
    forecast_results,
    forecast_summary,
    historical_revenue,
)

render_page_header(
    "Revenue Forecasting",
    "Historical performance and forward-looking revenue outlook from the retained time-series workflow.",
    status="ARIMA-based",
    tone="info",
)

horizon = st.slider("Forecast Horizon", min_value=1, max_value=12, value=12, step=1)
history = historical_revenue()
forecast = forecast_results(horizon)
kpis = forecast_kpis(horizon)

render_kpi_grid(
    [
        {"label": "Historical Months", "value": str(kpis["historical_months"])},
        {"label": "Forecast Horizon", "value": f"{kpis['forecast_horizon']} months"},
        {"label": "Last Observed Revenue", "value": money(kpis["last_observed_revenue"])},
        {"label": "Projected Revenue", "value": money(kpis["projected_revenue"])},
    ],
    columns=4,
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history["year_month"],
        y=history["net_revenue"],
        mode="lines+markers",
        name="Historical Revenue",
        line=dict(color="#2563eb", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast["year_month"],
        y=forecast["forecasted_net_revenue"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#dc2626", width=2),
    )
)
fig.update_layout(
    template="plotly_dark",
    height=430,
    title="Historical and Forecasted Net Revenue",
    xaxis_title="Month",
    yaxis_title="Net Revenue",
    margin=dict(l=10, r=10, t=35, b=10),
    hovermode="x unified",
)
render_chart(fig, height=430)

left, right = st.columns(2)
with left:
    render_section_heading("Forecast output", "Prepared forward-looking revenue path")
    render_table(forecast, height=260)

with right:
    render_section_heading("Model comparison", "Saved baseline and model comparison information")
    comparison = forecast_model_comparison()
    render_table(comparison, height=260)

render_section_heading("Executive forecast summary", "Narrative summary from the exported forecast results")
st.write(forecast_summary())
