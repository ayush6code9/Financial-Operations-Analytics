from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import load_customers, load_output_csv, load_transactions
from .paths import CHURN_MODEL_PATH, MODELS_DIR

CUSTOMER_ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn_target"
HIGH_RISK_THRESHOLD = 0.65
MEDIUM_RISK_THRESHOLD = 0.35
RANDOM_STATE = 42


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def risk_level(probability: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    return "Low Risk"


def aggregate_transaction_behavior(transactions: pd.DataFrame) -> pd.DataFrame:
    """Replicate the customer-level transaction feature logic from the churn notebook."""
    data = transactions.copy()
    if "transaction_date" in data.columns:
        data["transaction_date"] = pd.to_datetime(data["transaction_date"], errors="coerce")

    if "transaction_status" in data.columns:
        status = data["transaction_status"].astype(str).str.lower()
        data["txn_completed_flag"] = status.eq("completed").astype(int)
        data["txn_failed_flag"] = status.eq("failed").astype(int)
        data["txn_refunded_flag"] = status.eq("refunded").astype(int)

    if "transaction_category" in data.columns:
        category = data["transaction_category"].astype(str).str.lower()
        data["txn_downgrade_flag"] = category.eq("downgrade").astype(int)
        data["txn_upgrade_flag"] = category.eq("upgrade").astype(int)
        data["txn_support_flag"] = category.eq("support").astype(int)

    aggregation_map = {
        "transaction_id": "count",
        "net_revenue": "sum",
        "profit": "sum",
        "gross_amount": "sum",
        "refund_amount": "sum",
        "payment_delay_days": "mean",
        "transaction_date": "max",
        "txn_completed_flag": "sum",
        "txn_failed_flag": "sum",
        "txn_refunded_flag": "sum",
        "txn_downgrade_flag": "sum",
        "txn_upgrade_flag": "sum",
        "txn_support_flag": "sum",
    }
    existing_map = {key: value for key, value in aggregation_map.items() if key in data.columns}
    aggregated = data.groupby(CUSTOMER_ID_COLUMN).agg(existing_map).reset_index()
    aggregated = aggregated.rename(
        columns={
            "transaction_id": "txn_count_from_transactions",
            "net_revenue": "txn_net_revenue_sum",
            "profit": "txn_profit_sum",
            "gross_amount": "txn_gross_amount_sum",
            "refund_amount": "txn_refund_amount_sum",
            "payment_delay_days": "txn_avg_payment_delay_days",
            "transaction_date": "txn_last_transaction_date",
            "txn_completed_flag": "txn_completed_count",
            "txn_failed_flag": "txn_failed_count",
            "txn_refunded_flag": "txn_refunded_count",
            "txn_downgrade_flag": "txn_downgrade_count",
            "txn_upgrade_flag": "txn_upgrade_count",
            "txn_support_flag": "txn_support_count",
        }
    )

    if "txn_count_from_transactions" in aggregated.columns:
        denominator = aggregated["txn_count_from_transactions"].replace(0, np.nan)
        for numerator, rate_column in [
            ("txn_failed_count", "txn_failed_rate"),
            ("txn_refunded_count", "txn_refunded_rate"),
            ("txn_downgrade_count", "txn_downgrade_rate"),
            ("txn_upgrade_count", "txn_upgrade_rate"),
            ("txn_support_count", "txn_support_rate"),
        ]:
            if numerator in aggregated.columns:
                aggregated[rate_column] = (aggregated[numerator] / denominator).fillna(0)
    return aggregated


def add_date_parts(data: pd.DataFrame, date_columns: Iterable[str]) -> pd.DataFrame:
    output = data.copy()
    for column in date_columns:
        if column not in output.columns:
            continue
        parsed = pd.to_datetime(output[column], errors="coerce")
        output[f"{column}_year"] = parsed.dt.year
        output[f"{column}_month"] = parsed.dt.month
        output[f"{column}_quarter"] = parsed.dt.quarter
    return output


def engineer_customer_features(customers: pd.DataFrame, transaction_features: pd.DataFrame) -> pd.DataFrame:
    features = customers.copy()
    if "churn" in features.columns and TARGET_COLUMN not in features.columns:
        features[TARGET_COLUMN] = features["churn"].astype(int)

    features = add_date_parts(features, ["signup_date", "last_transaction_date", "acquisition_month"])

    if CUSTOMER_ID_COLUMN in features.columns and CUSTOMER_ID_COLUMN in transaction_features.columns:
        features = features.merge(transaction_features, on=CUSTOMER_ID_COLUMN, how="left")

    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    features[numeric_columns] = features[numeric_columns].fillna(0)
    return recompute_derived_features(features)


def recompute_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.copy()
    if {"support_tickets", "tenure_months"}.issubset(features.columns):
        features["support_tickets_per_tenure_month"] = safe_divide(
            pd.to_numeric(features["support_tickets"], errors="coerce"),
            pd.to_numeric(features["tenure_months"], errors="coerce") + 1,
        )
    if {"total_revenue", "total_transactions"}.issubset(features.columns):
        features["revenue_per_customer_transaction"] = safe_divide(
            pd.to_numeric(features["total_revenue"], errors="coerce"),
            pd.to_numeric(features["total_transactions"], errors="coerce"),
        )
    if {"total_profit", "total_revenue"}.issubset(features.columns):
        features["customer_profit_margin_calculated"] = safe_divide(
            pd.to_numeric(features["total_profit"], errors="coerce"),
            pd.to_numeric(features["total_revenue"], errors="coerce"),
        )
    if {"discount_percentage", "monthly_recurring_revenue"}.issubset(features.columns):
        features["discount_to_mrr_ratio"] = safe_divide(
            pd.to_numeric(features["discount_percentage"], errors="coerce"),
            pd.to_numeric(features["monthly_recurring_revenue"], errors="coerce"),
        )
    if {"login_frequency", "tenure_months"}.issubset(features.columns):
        features["login_frequency_per_tenure_month"] = safe_divide(
            pd.to_numeric(features["login_frequency"], errors="coerce"),
            pd.to_numeric(features["tenure_months"], errors="coerce") + 1,
        )
    return features


def select_model_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], pd.DataFrame]:
    """Use the notebook's leakage exclusions and feature selection rules."""
    exclude_columns = {
        TARGET_COLUMN,
        "churn",
        CUSTOMER_ID_COLUMN,
        "signup_date",
        "last_transaction_date",
        "acquisition_month",
        "txn_last_transaction_date",
        "is_active",
        "risk_score",
        "customer_segment",
        "customer_profitability",
    }
    exclude_columns = {column for column in exclude_columns if column in data.columns}
    candidate_features = data.drop(columns=list(exclude_columns), errors="ignore").copy()

    single_value_columns = [
        column for column in candidate_features.columns if candidate_features[column].nunique(dropna=True) <= 1
    ]
    high_cardinality_columns = [
        column
        for column in candidate_features.select_dtypes(exclude=np.number).columns
        if candidate_features[column].nunique(dropna=True) > 50
    ]
    candidate_features = candidate_features.drop(
        columns=single_value_columns + high_cardinality_columns,
        errors="ignore",
    )

    numeric_features = candidate_features.select_dtypes(include=np.number).columns.tolist()
    categorical_features = candidate_features.select_dtypes(exclude=np.number).columns.tolist()

    audit_records: list[dict[str, str]] = []
    for column in sorted(exclude_columns):
        audit_records.append({"column": column, "action": "excluded", "reason": "identifier_target_or_leakage"})
    for column in single_value_columns:
        audit_records.append({"column": column, "action": "excluded", "reason": "single_value"})
    for column in high_cardinality_columns:
        audit_records.append({"column": column, "action": "excluded", "reason": "high_cardinality"})
    for column in numeric_features:
        audit_records.append({"column": column, "action": "included", "reason": "numeric_feature"})
    for column in categorical_features:
        audit_records.append({"column": column, "action": "included", "reason": "categorical_feature"})

    return (
        candidate_features,
        data[TARGET_COLUMN].astype(int),
        numeric_features,
        categorical_features,
        pd.DataFrame(audit_records),
    )


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", build_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )


@lru_cache(maxsize=1)
def build_modeling_frame() -> pd.DataFrame:
    return engineer_customer_features(load_customers(), aggregate_transaction_behavior(load_transactions()))


def _feature_defaults(X: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in numeric_features:
        defaults[column] = float(pd.to_numeric(X[column], errors="coerce").median())
    for column in categorical_features:
        mode = X[column].dropna().mode()
        defaults[column] = str(mode.iloc[0]) if not mode.empty else ""
    return defaults


def _numeric_quantiles(X: pd.DataFrame, numeric_features: list[str]) -> dict[str, dict[str, float]]:
    quantiles: dict[str, dict[str, float]] = {}
    for column in numeric_features:
        series = pd.to_numeric(X[column], errors="coerce")
        quantiles[column] = {
            "p25": float(series.quantile(0.25)),
            "median": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
        }
    return quantiles


def _categorical_options(X: pd.DataFrame, categorical_features: list[str]) -> dict[str, list[str]]:
    return {
        column: sorted([str(value) for value in X[column].dropna().unique().tolist()])
        for column in categorical_features
    }


def _category_churn_rates(modeling_df: pd.DataFrame, categorical_features: list[str]) -> dict[str, dict[str, float]]:
    rates: dict[str, dict[str, float]] = {}
    for column in categorical_features:
        if column in modeling_df.columns:
            series = modeling_df.groupby(column, observed=False)[TARGET_COLUMN].mean()
            rates[column] = {str(key): float(value) for key, value in series.items()}
    return rates


def _raw_feature_name(processed_feature: str, feature_columns: Iterable[str]) -> str:
    cleaned = processed_feature.replace("numeric__", "").replace("categorical__", "")
    for column in sorted(feature_columns, key=len, reverse=True):
        if cleaned == column or cleaned.startswith(f"{column}_"):
            return column
    return cleaned


def _feature_importance_payload(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"]), pd.DataFrame(columns=["feature", "importance"])

    processed_names = preprocessor.get_feature_names_out()
    processed_importance = (
        pd.DataFrame({"feature": processed_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    raw_importance = processed_importance.copy()
    raw_importance["raw_feature"] = raw_importance["feature"].map(lambda name: _raw_feature_name(name, feature_columns))
    raw_importance = (
        raw_importance.groupby("raw_feature", as_index=False)["importance"]
        .sum()
        .rename(columns={"raw_feature": "feature"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return processed_importance, raw_importance


def train_churn_artifact(model_path: str | None = None) -> dict[str, Any]:
    """Train the final XGBoost pipeline from the real processed data and save it."""
    path = CHURN_MODEL_PATH if model_path is None else model_path
    path = str(path)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    modeling_df = build_modeling_frame()
    X, y, numeric_features, categorical_features, feature_audit = select_model_features(modeling_df)

    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    estimator = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negative_count / max(positive_count, 1),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )
    pipeline.fit(X, y)

    processed_importance, raw_importance = _feature_importance_payload(pipeline, X, X.columns.tolist())
    payload: dict[str, Any] = {
        "model_name": "XGBoost",
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_audit": feature_audit,
        "defaults": _feature_defaults(X, numeric_features, categorical_features),
        "numeric_quantiles": _numeric_quantiles(X, numeric_features),
        "categorical_options": _categorical_options(X, categorical_features),
        "category_churn_rates": _category_churn_rates(modeling_df, categorical_features),
        "processed_feature_importance": processed_importance,
        "raw_feature_importance": raw_importance,
        "thresholds": {
            "medium": MEDIUM_RISK_THRESHOLD,
            "high": HIGH_RISK_THRESHOLD,
        },
        "training_rows": int(len(X)),
        "target_source": "financial_customers_clean.csv:churn",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, path)
    return payload


def load_churn_artifact(retrain_if_missing: bool = True) -> dict[str, Any]:
    if CHURN_MODEL_PATH.exists():
        return joblib.load(CHURN_MODEL_PATH)
    if not retrain_if_missing:
        raise FileNotFoundError(f"Churn model artifact not found: {CHURN_MODEL_PATH}")
    return train_churn_artifact()


def feature_row_for_customer(customer_id: str, artifact: dict[str, Any] | None = None) -> pd.Series:
    artifact = artifact or load_churn_artifact()
    modeling_df = build_modeling_frame()
    matches = modeling_df[modeling_df[CUSTOMER_ID_COLUMN].astype(str) == str(customer_id)]
    if matches.empty:
        raise KeyError(f"Customer not found: {customer_id}")
    row = matches.iloc[0].copy()
    return row.reindex(artifact["feature_columns"]).fillna(pd.Series(artifact["defaults"]))


def default_feature_row(artifact: dict[str, Any] | None = None) -> pd.Series:
    artifact = artifact or load_churn_artifact()
    return pd.Series(artifact["defaults"]).reindex(artifact["feature_columns"])


def prepare_prediction_frame(
    features: dict[str, Any] | pd.Series | pd.DataFrame | None = None,
    customer_id: str | None = None,
    artifact: dict[str, Any] | None = None,
) -> pd.DataFrame:
    artifact = artifact or load_churn_artifact()
    if isinstance(features, pd.DataFrame):
        frame = features.copy()
    else:
        base = feature_row_for_customer(customer_id, artifact) if customer_id else default_feature_row(artifact)
        if features is not None:
            incoming = features.to_dict() if isinstance(features, pd.Series) else dict(features)
            for key, value in incoming.items():
                if key in base.index:
                    base[key] = value
        frame = pd.DataFrame([base])

    for column, default in artifact["defaults"].items():
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default)

    frame = recompute_derived_features(frame)
    return frame.reindex(columns=artifact["feature_columns"])


def prediction_factors(row: pd.Series, probability: float, artifact: dict[str, Any]) -> list[str]:
    quantiles = artifact.get("numeric_quantiles", {})
    factors: list[str] = []

    def high(column: str, label: str) -> None:
        if column in row.index and column in quantiles and float(row[column]) >= quantiles[column]["p75"]:
            factors.append(label)

    def low(column: str, label: str) -> None:
        if column in row.index and column in quantiles and float(row[column]) <= quantiles[column]["p25"]:
            factors.append(label)

    high("days_since_last_transaction", "High recency since last transaction")
    low("usage_score", "Low product usage score")
    low("login_frequency", "Low login frequency")
    low("nps_score", "Low customer NPS")
    high("support_tickets", "High support ticket volume")
    low("tenure_months", "Short customer tenure")
    low("total_transactions", "Lower transaction activity")

    category_rates = artifact.get("category_churn_rates", {})
    for column, label in [
        ("contract_type", "contract type"),
        ("subscription_plan", "subscription plan"),
        ("company_size", "company size"),
    ]:
        value = str(row.get(column, ""))
        rates = category_rates.get(column, {})
        if value in rates and rates[value] >= 0.30:
            factors.append(f"Elevated historical churn for {value} {label}")

    if not factors:
        top_features = artifact.get("raw_feature_importance", pd.DataFrame())
        if isinstance(top_features, pd.DataFrame) and not top_features.empty:
            factors = [f"Model signal: {feature}" for feature in top_features["feature"].head(3).tolist()]

    if probability >= HIGH_RISK_THRESHOLD and "High predicted probability from the XGBoost churn model" not in factors:
        factors.insert(0, "High predicted probability from the XGBoost churn model")
    return factors[:5]


def business_recommendation(probability: float, row: pd.Series) -> str:
    total_revenue = float(row.get("total_revenue", 0) or 0)
    if probability >= HIGH_RISK_THRESHOLD and total_revenue >= 10000:
        return "Prioritize this account for retention outreach with a value-protection offer and customer success follow-up."
    if probability >= HIGH_RISK_THRESHOLD:
        return "Place this customer in the retention queue and diagnose recent engagement or product-fit issues."
    if probability >= MEDIUM_RISK_THRESHOLD:
        return "Monitor the customer and trigger a light-touch engagement campaign before risk increases."
    return "Maintain the current lifecycle motion and look for expansion opportunities if value indicators are strong."


def predict_churn(
    features: dict[str, Any] | pd.Series | pd.DataFrame | None = None,
    customer_id: str | None = None,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = artifact or load_churn_artifact()
    frame = prepare_prediction_frame(features=features, customer_id=customer_id, artifact=artifact)
    probability = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    row = frame.iloc[0]
    return {
        "churn_probability": probability,
        "predicted_churn": int(probability >= 0.50),
        "risk_level": risk_level(probability),
        "model_name": artifact["model_name"],
        "factors": prediction_factors(row, probability, artifact),
        "recommendation": business_recommendation(probability, row),
        "features_used": artifact["feature_columns"],
    }


def existing_prediction_for_customer(customer_id: str) -> dict[str, Any] | None:
    predictions = load_output_csv("churn_predictions.csv")
    match = predictions[predictions[CUSTOMER_ID_COLUMN].astype(str) == str(customer_id)]
    if match.empty:
        return None
    record = match.iloc[0].to_dict()
    record["risk_level"] = str(record.get("risk_segment", risk_level(float(record["churn_probability"]))))
    return record


def model_metrics() -> dict[str, Any]:
    metrics = load_output_csv("model_evaluation_metrics.csv")
    cv_metrics = load_output_csv("cross_validation_metrics.csv")
    feature_importance = load_output_csv("churn_feature_importance.csv")
    shap_importance = load_output_csv("shap_feature_importance.csv")
    return {
        "metrics": metrics,
        "cross_validation": cv_metrics,
        "feature_importance": feature_importance,
        "shap_importance": shap_importance,
        "best_model": str(metrics.sort_values("roc_auc", ascending=False).iloc[0]["model"]),
    }
