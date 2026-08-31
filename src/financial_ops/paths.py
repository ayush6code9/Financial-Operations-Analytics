from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from scripts, Streamlit pages, or tests."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "data" / "processed").exists() and (candidate / "outputs").exists():
            return candidate
    return current


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
CHURN_MODEL_PATH = MODELS_DIR / "churn_xgboost_pipeline.joblib"
