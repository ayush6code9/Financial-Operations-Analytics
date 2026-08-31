from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b0f14;
            --panel: #151c24;
            --panel-strong: #19222d;
            --border: #263241;
            --text: #f5f7fa;
            --muted: #a7b0bc;
            --primary: #3b82f6;
            --primary-soft: #172b49;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --info: #38bdf8;
            --shadow: 0 10px 24px rgba(0, 0, 0, 0.2);
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: var(--bg);
        }
        [data-testid="stHeader"] { color: var(--text); }
        .stApp .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1480px;
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1rem 0.9rem;
            box-shadow: var(--shadow);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 600;
            font-size: 0.78rem;
            letter-spacing: 0.01em;
        }
        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 700;
            font-size: clamp(1.15rem, 1.2vw + 0.8rem, 1.9rem);
        }
        div[data-testid="stDataFrame"], .stDataFrame {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--panel-strong);
            box-shadow: var(--shadow);
        }
        section[data-testid="stSidebar"] {
            background: #0d131a;
            border-right: 1px solid var(--border);
            width: min(320px, 82vw) !important;
            min-width: min(320px, 82vw) !important;
        }
        [data-testid="stSidebarNav"] {
            gap: 0.3rem;
        }
        [data-testid="stSidebarNav"] a {
            border-radius: 10px;
            padding: 0.7rem 0.8rem;
            margin: 0.1rem 0.35rem;
            color: var(--text);
            font-weight: 600;
        }
        [data-testid="stSidebarNav"] a:hover {
            background: #182433;
            border: 1px solid transparent;
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: var(--primary-soft);
            color: #ffffff;
            border: 1px solid #28558f;
        }
        .brand-block {
            padding: 1rem 0.8rem 0.55rem;
            margin: 0.25rem 0.5rem 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        .brand-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #c3ccd7;
            font-weight: 700;
        }
        .brand-subtitle {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--text);
            margin-top: 0.2rem;
        }
        .page-header {
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            margin-bottom: 1rem;
        }
        .page-title {
            font-size: clamp(1.8rem, 2vw + 1rem, 2.8rem);
            line-height: 1.15;
            font-weight: 800;
            color: var(--text);
        }
        .page-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
        }
        .section-header {
            margin: 1.25rem 0 0.75rem;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
        }
        .section-subtitle {
            color: var(--muted);
            font-size: 0.83rem;
            margin-top: 0.2rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 2rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border: 1px solid transparent;
            width: fit-content;
            margin-left: auto;
            margin-top: 0.3rem;
        }
        .status-success { background: rgba(34,197,94,0.14); color: #86efac; border-color: rgba(34,197,94,0.35); }
        .status-warning { background: rgba(245,158,11,0.14); color: #fcd34d; border-color: rgba(245,158,11,0.35); }
        .status-danger { background: rgba(239,68,68,0.14); color: #fca5a5; border-color: rgba(239,68,68,0.35); }
        .status-info { background: rgba(56,189,248,0.14); color: #7dd3fc; border-color: rgba(56,189,248,0.35); }
        .status-neutral { background: rgba(167,176,188,0.12); color: #d7dde5; border-color: rgba(167,176,188,0.3); }
        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }
        h1, h2, h3, h4, h5 { letter-spacing: -0.01em; }
        .stTabs [role="tablist"] {
            gap: 0.4rem;
        }
        .stTabs [role="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 0.6rem 0.9rem;
        }
        .stTabs [role="tab"] p, [data-testid="stSidebarNav"] a span {
            color: var(--muted);
        }
        .stTabs [aria-selected="true"] p, [data-testid="stSidebarNav"] a[aria-current="page"] span {
            color: var(--text);
        }
        input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div,
        [data-testid="stFileUploaderDropzone"] {
            background: #10161d !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }
        input:focus, textarea:focus, [data-baseweb="select"] > div:focus-within,
        [data-baseweb="input"] > div:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 1px var(--primary) !important;
        }
        [data-baseweb="popover"], [role="listbox"], [data-baseweb="menu"] {
            background: #151c24 !important;
            color: var(--text) !important;
        }
        [role="option"]:hover, [data-baseweb="menu"] li:hover { background: #223249 !important; }
        button[kind="primary"] { background: var(--primary); color: #ffffff; }
        button[kind="secondary"] { background: #19222d; color: var(--text); border-color: var(--border); }
        [data-testid="stAlert"] { background: #151c24; color: var(--text); border-color: var(--border); }
        [data-testid="stExpander"] { background: var(--panel); border-color: var(--border); }
        [data-testid="stCaptionContainer"], .stMarkdown, label { color: var(--muted); }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, label { color: var(--text); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _parse_numeric_string(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        cleaned = text.replace("$", "").replace(",", "").replace("%", "")
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            return float(cleaned)
        except ValueError:
            return value
    return value


def clean_table(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        if non_null.empty:
            continue
        converted = non_null.map(_parse_numeric_string)
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in converted if item is not None):
            frame[column] = series.map(_parse_numeric_string)
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.where(pd.notna(frame), None)


def money(value: Any, compact: bool = True) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if compact:
        if abs(number) >= 1_000_000_000:
            return f"${number / 1_000_000_000:.2f}B"
        if abs(number) >= 1_000_000:
            return f"${number / 1_000_000:.1f}M"
        if abs(number) >= 1_000:
            return f"${number / 1_000:.1f}K"
    return f"${number:,.2f}"


def number(value: Any) -> str:
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def pct(value: Any, already_percent: bool = False) -> str:
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not already_percent:
        number_value *= 100
    return f"{number_value:.1f}%"


@st.cache_resource(show_spinner=False)
def get_churn_artifact() -> dict[str, Any]:
    from financial_ops.churn_model import load_churn_artifact

    return load_churn_artifact()


@st.cache_data(show_spinner=False)
def customer_ids() -> list[str]:
    from financial_ops.data import load_customers

    return load_customers()["customer_id"].astype(str).sort_values().tolist()
