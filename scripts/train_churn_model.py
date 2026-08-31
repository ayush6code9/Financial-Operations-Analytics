from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from financial_ops.churn_model import train_churn_artifact
from financial_ops.paths import CHURN_MODEL_PATH


def main() -> None:
    artifact = train_churn_artifact()
    print(f"Saved churn model: {CHURN_MODEL_PATH}")
    print(f"Model: {artifact['model_name']}")
    print(f"Training rows: {artifact['training_rows']:,}")
    print(f"Raw model features: {len(artifact['feature_columns'])}")


if __name__ == "__main__":
    main()
