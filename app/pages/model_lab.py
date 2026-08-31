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

from app.shared import get_churn_artifact, pct
from app.ui import render_chart, render_kpi_grid, render_page_header, render_section_heading, render_table
from financial_ops.churn_model import model_metrics

render_page_header(
    "ML Model Lab",
    "Evaluation, feature importance, and explainability for the retained churn model artifact.",
    status="Monitoring",
    tone="neutral",
)

artifact = get_churn_artifact()
metrics = model_metrics()
model_table = metrics["metrics"].sort_values("roc_auc", ascending=False)
best = model_table.iloc[0]

render_kpi_grid(
    [
        {"label": "Best Model", "value": str(best["model"])},
        {"label": "ROC-AUC", "value": pct(best["roc_auc"])},
        {"label": "F1", "value": pct(best["f1"])},
        {"label": "Recall", "value": pct(best["recall"])},
        {"label": "Precision", "value": pct(best["precision"])},
    ],
    columns=5,
)

fig = px.bar(
    model_table.melt(
        id_vars="model",
        value_vars=["precision", "recall", "f1", "roc_auc", "average_precision"],
        var_name="metric",
        value_name="score",
    ),
    x="metric",
    y="score",
    color="model",
    barmode="group",
    labels={"metric": "Metric", "score": "Score", "model": "Model"},
    title="Churn Model Comparison",
    color_discrete_sequence=px.colors.qualitative.Safe,
)
fig.update_layout(template="plotly_dark", height=420, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=35, b=10))
render_chart(fig, height=420)

left, right = st.columns(2)
with left:
    render_section_heading("Holdout metrics", "Model evaluation results from the saved artifact")
    render_table(model_table, height=240)

with right:
    render_section_heading("Cross-validation metrics", "Validation stability across the ensemble fit")
    render_table(metrics["cross_validation"], height=240)

fi = metrics["feature_importance"].head(15)
fig_fi = px.bar(
    fi,
    x="importance",
    y="feature",
    orientation="h",
    title="XGBoost Feature Importance",
    color="importance",
    color_continuous_scale=["#dbeafe", "#2563eb"],
)
fig_fi.update_layout(template="plotly_dark", height=400, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=35, b=10))
render_chart(fig_fi, height=400)

shap = metrics["shap_importance"].head(15)
fig_shap = px.bar(
    shap,
    x="mean_abs_shap",
    y="feature",
    orientation="h",
    title="Global SHAP Feature Impact",
    color="mean_abs_shap",
    color_continuous_scale=["#fef3c7", "#b45309"],
)
fig_shap.update_layout(template="plotly_dark", height=400, yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=35, b=10))
render_chart(fig_shap, height=400)

with st.expander("Saved model artifact"):
    st.write(f"Artifact model: **{artifact['model_name']}**")
    st.write(f"Training rows: **{artifact['training_rows']:,}**")
    st.write(f"Raw feature columns: **{len(artifact['feature_columns'])}**")
    render_table(artifact["raw_feature_importance"].head(20), height=220)
