from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared import apply_theme

st.set_page_config(
    page_title="Financial Operations Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

st.sidebar.markdown(
    """
    <div class="brand-block">
        <div class="brand-title">Financial Operations</div>
        <div class="brand-subtitle">Analytics Suite</div>
    </div>
    """,
    unsafe_allow_html=True,
)

pages = [
    st.Page("pages/overview.py", title="Executive Overview", icon="🏢"),
    st.Page("pages/churn.py", title="Customer Churn Prediction", icon="⚠️"),
    st.Page("pages/customer360.py", title="Customer 360", icon="👤"),
    st.Page("pages/forecasting.py", title="Revenue Forecasting", icon="📈"),
    st.Page("pages/model_lab.py", title="ML Model Lab", icon="🧠"),
    st.Page("pages/insights.py", title="Business Insights", icon="💡"),
]

navigation = st.navigation(pages, position="sidebar")
with st.sidebar:
    st.caption("Portfolio-grade operating view")
    st.caption("Last refreshed from source data outputs")

navigation.run()
