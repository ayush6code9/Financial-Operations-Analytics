from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.shared import clean_table
from app.ui import render_page_header, render_section_heading, render_table
from financial_ops.analytics import business_insights
from financial_ops.data import load_output_csv

render_page_header(
    "Business Insights",
    "Operationally relevant findings and action-oriented recommendations synthesized from the project outputs.",
    status="Executive summary",
    tone="success",
)

insights = business_insights()

for title in ["Executive", "Executive Recommendations", "Churn", "RFM", "Cohort", "Cohort Recommendations"]:
    render_section_heading(title, "Dataset-backed business signal")
    render_table(insights[title], height=200)

render_section_heading("Segment actions", "Recommended retention and acquisition strategies by segment")
strategy = insights["Marketing Strategy"]
render_table(strategy, height=220)

render_section_heading("Profitability signals", "Portfolio profitability context from the summary outputs")
profitability = load_output_csv("executive_profitability_summary.csv")
render_table(profitability, height=220)
