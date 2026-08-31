from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import DATA_DIR, OUTPUTS_DIR


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, **kwargs)


@lru_cache(maxsize=32)
def load_customers() -> pd.DataFrame:
    return _read_csv(DATA_DIR / "financial_customers_clean.csv")


@lru_cache(maxsize=32)
def load_transactions() -> pd.DataFrame:
    return _read_csv(DATA_DIR / "financial_transactions_clean.csv")


@lru_cache(maxsize=32)
def load_monthly_revenue() -> pd.DataFrame:
    data = _read_csv(DATA_DIR / "monthly_revenue_clean.csv")
    data["year_month"] = pd.to_datetime(data["year_month"], errors="coerce")
    return data.sort_values("year_month").reset_index(drop=True)


@lru_cache(maxsize=64)
def load_output_csv(name: str) -> pd.DataFrame:
    return _read_csv(OUTPUTS_DIR / name)


@lru_cache(maxsize=16)
def load_output_text(name: str) -> str:
    path = OUTPUTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def metric_lookup(name: str = "executive_kpis.csv") -> dict[str, Any]:
    data = load_output_csv(name)
    if not {"metric", "value"}.issubset(data.columns):
        return {}
    return dict(zip(data["metric"].astype(str), data["value"]))


def clean_records(data: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    frame = data.head(limit).copy() if limit else data.copy()
    frame = frame.where(pd.notna(frame), None)
    return frame.to_dict(orient="records")
