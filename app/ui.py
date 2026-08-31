from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.shared import clean_table


def render_page_header(title: str, subtitle: str, status: str | None = None, tone: str = "neutral") -> None:
    left, right = st.columns([4, 1])
    with left:
        st.markdown(
            f"""
            <div class="page-header">
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if status:
        with right:
            st.markdown(
                f"""
                <div class="status-pill status-{tone}">{status}</div>
                """,
                unsafe_allow_html=True,
            )


def render_kpi_grid(items: Iterable[dict[str, Any]], columns: int = 4) -> None:
    items = list(items)
    if not items:
        return
    grid = st.columns(columns)
    for index, item in enumerate(items):
        with grid[index % columns]:
            st.metric(
                label=item.get("label", "Metric"),
                value=item.get("value", "-"),
                delta=item.get("delta"),
                help=item.get("help"),
            )


def render_section_heading(title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-title">{title}</div>
            {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart(fig: go.Figure | Any, height: int = 350) -> None:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="#10161D",
        plot_bgcolor="#10161D",
        font=dict(color="#F5F7FA"),
        title_font=dict(color="#F5F7FA"),
        legend=dict(font=dict(color="#F5F7FA")),
        xaxis=dict(gridcolor="#263241", zerolinecolor="#263241", tickfont=dict(color="#A7B0BC"), title_font=dict(color="#A7B0BC")),
        yaxis=dict(gridcolor="#263241", zerolinecolor="#263241", tickfont=dict(color="#A7B0BC"), title_font=dict(color="#A7B0BC")),
    )
    st.plotly_chart(fig, width="stretch", height=height)


def render_table(frame: pd.DataFrame, *, height: int | None = None) -> None:
    cleaned = clean_table(frame)
    st.dataframe(cleaned, hide_index=True, width="stretch", height=height)
