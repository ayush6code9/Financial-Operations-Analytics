# Financial Operations Analytics

> An end-to-end financial analytics and predictive ML platform for revenue forecasting, customer churn prediction, customer segmentation, cohort analysis, profitability analysis, and executive decision support.

### 🚀 [Live Dashboard](https://financial-operations-analytics-1.onrender.com/)
**Try the complete interactive application →**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Analytics-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Churn%20Prediction-189FDD)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

---

## Executive Summary

Financial Operations Analytics is an end-to-end analytics project that transforms customer, transaction, and revenue data into decision-ready financial and customer insights.

The platform combines data processing, exploratory analysis, statistical analytics, machine learning, time-series forecasting, customer segmentation, cohort analysis, profitability analysis, explainable AI, and interactive business intelligence in a single deployable application.

```text
Operational Data
      ↓
Data Cleaning & Processing
      ↓
Exploratory & Statistical Analysis
      ↓
Feature Engineering
      ↓
ML + Forecasting
      ↓
Business Insights
      ↓
Interactive Dashboard
      ↓
Production Deployment
```

---

## Key Business Results

| KPI | Result |
|---|---:|
| Customers analyzed | 20,000 |
| Transactions analyzed | 329,202 |
| Historical revenue | $530.3M |
| Total profit | $438.9M |
| Profit margin | 82.8% |
| Observed churn rate | 27.8% |
| High-risk customers flagged | 5,284 |
| Average customer lifetime value | $25.8K |
| Best churn model | XGBoost (ROC-AUC 0.988, F1 0.936) |
| Forecast horizon | 12 months |
| Forecasted revenue | $709.5M |
| Month-1 retention | 93.1% |
| Month-6 retention | 70.3% |
| Top RFM revenue segment | Big Spenders — $226.2M |

These KPIs are surfaced live through the deployed executive dashboard.

---

## Business Problems Solved

**Revenue & Forecasting** — Analyze revenue trends and seasonality, evaluate historical performance, and generate forward-looking revenue forecasts.

**Customer Retention** — Identify customers likely to churn, prioritize high-risk accounts, and translate behavioral signals into retention opportunities.

**Customer Value** — Identify high-value customers, compare segments, and understand behavioral and monetary differences between them.

**Cohort & Retention** — Measure cohort performance over time and identify the strongest and weakest customer cohorts.

**Profitability** — Analyze customer- and product-level profitability to identify high-value and low-value areas of the business.

---

## Application Pages

| Page | What it demonstrates |
|---|---|
| Executive Overview | Revenue, profit, churn, customer, cohort, and segment KPIs; monthly revenue trends and forecast |
| Customer Churn Prediction | Live XGBoost inference with actual model features and risk-based recommendations |
| Customer 360 | Customer-level view combining profile, RFM, churn risk, profitability, cohort context, and transaction history |
| Revenue Forecasting | Historical revenue, exported ARIMA forecast, forecast KPIs, and model context |
| ML Model Lab | Model comparison, holdout metrics, cross-validation, feature importance, and SHAP explainability |
| Business Insights | Dataset-backed recommendations from churn, cohort, RFM, and profitability analysis |

---

## Analytics & Machine Learning

### 1. Data Acquisition & Processing
Reusable modules handle customer data, transaction data, and monthly revenue data, producing cleaned, analysis-ready datasets and downstream analytical outputs.

```text
src/financial_ops/
```

### 2. Exploratory Data Analysis
Covers data quality checks, missing-value and outlier analysis, univariate/bivariate/multivariate analysis, and revenue, customer, and transaction behavior patterns.

```text
notebooks/02_Exploratory_Data_Analysis.ipynb
```

### 3. Revenue Analytics & Forecasting
Monthly revenue trends, distribution, time-series analysis, and forecast generation/evaluation.

```text
notebooks/03_Revenue_Time_Series_Forecasting.ipynb
```
Key outputs: `outputs/forecast_results.csv`, `outputs/forecast_summary.txt`, `outputs/revenue_analysis.csv`

### 4. Customer Churn Prediction
An XGBoost classification pipeline predicts churn from behavioral and transactional features.

```text
Customer + Transaction Data → Feature Engineering → XGBoost Classification
    → Model Evaluation → Cross Validation → Feature Importance
    → SHAP Explainability → Customer Risk Scoring
```

Model comparison (holdout metrics, `outputs/model_evaluation_metrics.csv`):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.965 | 0.949 | 0.924 | 0.936 | 0.988 |
| Logistic Regression | 0.956 | 0.923 | 0.919 | 0.921 | 0.984 |
| Random Forest | 0.953 | 0.956 | 0.872 | 0.912 | 0.981 |
| Decision Tree | 0.944 | 0.885 | 0.918 | 0.901 | 0.969 |
| Baseline | 0.722 | 0.000 | 0.000 | 0.000 | 0.500 |

Trained artifact: `models/churn_xgboost_pipeline.joblib`
Explainability outputs: `outputs/churn_feature_importance.csv`, `outputs/shap_feature_importance.csv`, `outputs/figures/shap_feature_impact.png`

```text
notebooks/04_Churn_Prediction.ipynb
```

### 5. RFM Customer Segmentation
Customers scored on Recency, Frequency, and Monetary value.

```text
notebooks/06_RFM_Customer_Segmentation.ipynb
```
Key outputs: `outputs/rfm_table.csv`, `outputs/segment_summary.csv`, `outputs/marketing_strategy_by_segment.csv`

### 6. Cohort & Retention Analysis
Cohort definitions, retention matrices, monthly retention trends, and best/worst-performing cohorts.

```text
notebooks/05_Cohort_Analysis.ipynb
```
Key outputs: `outputs/cohort_summary.csv`, `outputs/monthly_retention_trend.csv`, `outputs/best_cohorts.csv`, `outputs/worst_cohorts.csv`

### 7. Customer Profitability
Customer- and product-level profitability, margin analysis, and top/low-value customer identification.

```text
notebooks/07_Profitability_Analysis.ipynb
```
Key outputs: `outputs/customer_profitability.csv`, `outputs/product_profitability.csv`, `outputs/profitability_summary.csv`

---

## Architecture

```text
                     ┌──────────────────────┐
                     │     Source Data      │
                     │ Customers            │
                     │ Transactions         │
                     │ Monthly Revenue      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │ Data Processing      │
                     │ Cleaning             │
                     │ Validation           │
                     │ Feature Engineering  │
                     └──────────┬───────────┘
                                │
                                ▼
              ┌──────────────────────────────────┐
              │          Analytics Layer         │
              │                                  │
              │ EDA                              │
              │ Revenue Analytics                │
              │ Forecasting                      │
              │ Churn Prediction                 │
              │ RFM Segmentation                 │
              │ Cohort Analysis                  │
              │ Profitability Analysis           │
              └───────────────┬──────────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Analytical Artifacts    │
                 │ CSV / TXT / Models      │
                 │ Visualizations          │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Streamlit Dashboard     │
                 │ Executive KPIs          │
                 │ Customer Analytics      │
                 │ ML Insights             │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Docker + Render         │
                 │ Production Deployment   │
                 └─────────────────────────┘
```

---

## Tech Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Plotly |
| Machine Learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Forecasting | Time-series forecasting / ARIMA (statsmodels) |
| Dashboard | Streamlit |
| API | FastAPI, Pydantic, Uvicorn |
| SQL | PostgreSQL (CTEs, constraints, business queries) |
| Testing | Pytest |
| Deployment | Docker, Render |
| Version Control | Git, GitHub |

---

## Project Structure

```text
Financial-Operations-Analytics/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── app/
│   ├── dashboard.py
│   ├── shared.py
│   ├── ui.py
│   └── pages/
│       ├── overview.py
│       ├── churn.py
│       ├── customer360.py
│       ├── forecasting.py
│       ├── model_lab.py
│       └── insights.py
│
├── data/
│   └── processed/
│       ├── financial_customers_clean.csv
│       ├── financial_transactions_clean.csv
│       └── monthly_revenue_clean.csv
│
├── models/
│   └── churn_xgboost_pipeline.joblib
│
├── notebooks/
│   ├── 01_Data_Acquisition.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Revenue_Time_Series_Forecasting.ipynb
│   ├── 04_Churn_Prediction.ipynb
│   ├── 05_Cohort_Analysis.ipynb
│   ├── 06_RFM_Customer_Segmentation.ipynb
│   ├── 07_Profitability_Analysis.ipynb
│   └── 08_Executive_Dashboard.ipynb
│
├── outputs/
│   ├── analytical CSV outputs
│   ├── business insights
│   ├── forecasting results
│   ├── model evaluation results
│   └── visualizations
│
├── scripts/
│   └── train_churn_model.py
│
├── sql/
│   ├── 01_database_setup.sql
│   ├── 02_create_tables.sql
│   ├── 03_import_clean_data.sql
│   └── 04_business_questions.sql
│
├── src/
│   └── financial_ops/
│       ├── analytics.py
│       ├── churn_model.py
│       ├── data.py
│       ├── forecasting.py
│       └── paths.py
│
├── tests/
│   └── test_core.py
│
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
├── requirements-app.txt
└── README.md
```

---

## API

A FastAPI layer is included under `api/main.py`, exposing endpoints for churn prediction, customer lookup, model metrics, and revenue forecasts.

Run locally:

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs` while the service is running.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API health check |
| POST | `/predict/churn` | Predict churn probability and risk level |
| GET | `/customer/{customer_id}` | Return Customer 360 analytics |
| GET | `/model/metrics` | Return churn model metrics, CV metrics, feature importance, SHAP output |
| GET | `/forecast/revenue?horizon=12` | Return historical and forecasted revenue |

Example churn request:

```json
{
  "customer_id": "CUST_000003",
  "features": {
    "days_since_last_transaction": 176,
    "usage_score": 46.6,
    "login_frequency": 14.1,
    "nps_score": 11,
    "subscription_plan": "Professional",
    "contract_type": "Annual"
  }
}
```

The Streamlit dashboard is the primary production-facing interface; the API is optional for the live demo since the dashboard reads the same local data, outputs, and model artifact directly.

---

## Docker

```bash
docker build -t financial-operations-analytics .
docker run -p 8501:8501 financial-operations-analytics
```

Or run both the Streamlit app and FastAPI service together:

```bash
docker compose up --build
```

- Streamlit: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

The production deployment on Render uses this same Dockerized Streamlit image. `requirements-app.txt` is a smaller runtime dependency file for the Streamlit/FastAPI demo; the full `requirements.txt` remains available for notebooks and extended analysis.

---

## Testing

```bash
pytest
```

Test configuration lives in `pytest.ini`; tests are deterministic and do not retrain models on every run.

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/ayush6code9/Financial-Operations-Analytics.git
cd Financial-Operations-Analytics

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements-app.txt

# 4. Start the dashboard
streamlit run app/dashboard.py
```

---

## Deployment

The application is deployed as a Dockerized Streamlit service on Render:

```text
GitHub Repository → Render → Docker Container → Streamlit Dashboard → Public Live Application
```

### 🚀 [Financial Operations Analytics — Live Dashboard](https://financial-operations-analytics-1.onrender.com/)

---

## Business Value

**Customer Retention** — Identify high-risk customers and prioritize retention efforts.
**Revenue Planning** — Use historical performance and forecasting to support forward-looking revenue planning.
**Customer Value** — Identify high-value customers and understand behavioral differences between segments.
**Profitability** — Evaluate customer and product economics to identify profitable and low-value areas.
**Executive Decision Support** — Provide management with a consolidated view of financial and customer performance.

---

## Portfolio Highlights

This project demonstrates the complete applied analytics lifecycle:

- Data ingestion, validation, and cleaning
- SQL schema design and business queries
- Exploratory data analysis and KPI design
- Feature engineering
- Time-series revenue forecasting
- Supervised churn modeling, model comparison, and cross-validation
- Explainability through feature importance and SHAP outputs
- Customer segmentation (RFM) and cohort/retention analysis
- Profitability analysis
- Reusable, modular Python analytics code
- Interactive Streamlit dashboard
- FastAPI backend
- Automated testing with Pytest
- Docker containerization and cloud deployment

---

## 🔗 Explore the Project

🚀 [Open the Live Dashboard](https://financial-operations-analytics-1.onrender.com/)
💻 [View the GitHub Repository](https://github.com/ayush6code9/Financial-Operations-Analytics)