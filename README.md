# YieldSense

**AI-Enabled NAND Test Yield & Recovery Analytics Pipeline**  
*Reverse-engineered for Micron SpecTek — MSB Test Solutions Engineering*

[![CI Pipeline](https://github.com/koi-bito/yieldsense/actions/workflows/ci.yml/badge.svg)](https://github.com/koi-bito/yieldsense/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

---

## Overview

**YieldSense** is an end-to-end, containerized data and ML engineering pipeline built to analyze semiconductor manufacturing test data, isolate root causes of parametric wafer failures, and rank failing lots for recovery and re-binning. The architecture mirrors SpecTek’s core business: salvaging marginal memory components through data-driven parametric triaging.

### Core Capabilities
- **Automated Data Quality & Cleaning**: Ingests 590 raw sensor channels into **DuckDB**, drops unreliable features (>40% nulls, zero variance), and performs robust median imputation.
- **Parametric Statistical Analysis**: Computes point-biserial correlations, decomposes collinear tool interactions via **PCA** (95% variance in 161 components), and tracks Shewhart **3-sigma SPC drift** events across manufacturing lots.
- **Explainable Machine Learning**: Trains an **Optuna-tuned XGBoost** classifier with leak-free SMOTE oversampling and extracts per-lot root causes using **TreeSHAP**.
- **Yield Recovery Scoring**: Employs continuous boundary proximity and fault sparsity metrics to identify salvageable lots (**30.8% candidate recovery rate**).
- **Containerized REST API**: Production-style **FastAPI** service with `/analyze-lot`, `/batch-analyze`, `/health`, and `/model-info` routes.
- **LLM Engineering Reporting & Validation**: Generates plain-English engineering analyses with a deterministic **rules-based validation layer** that verifies numeric claims and detects hallucinations.

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           YIELDSENSE ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘
  Raw SECOM Data (590 Sensors, 1,567 Lots)
       │
       ▼
  [Data Ingestion & Cleaning] ───> DuckDB (raw_lots, clean_lots)
       │                           (442 Active Features, Median Imputed)
       ├─────────────────────────────────────────┬────────────────────────────┐
       ▼                                         ▼                            ▼
  [Statistical Analysis]                   [ML Pipeline]                [Production API]
  • Point-Biserial Correlation             • Stratified 80/20 Split     • POST /analyze-lot
  • PCA (95% Variance in 161 PCs)          • SMOTE (Train Fold Only)    • POST /batch-analyze
  • Shewhart 3-Sigma SPC Drift             • Optuna Hyperparameter Tune • GET /health
  • Physical Spec Envelope Check           • TreeSHAP Attribution       • GET /model-info
       │                                   • Composite Recovery Scoring       │
       └────────────────────┬────────────────────┘                            │
                            │                                                 ▼
                            ▼                                            Dockerized
                 [Context Aggregator JSON]                              (FastAPI App)
                            │
                            ▼
               [LLM Reporting Drafter] (Claude API / Mock)
                            │
                            ▼
              [Deterministic Validation Layer]
              • Whitelist Sensor Verification
              • Ground-Truth Number Verification
                            │
                            ▼
            Validated Engineering Report Artifact
```

---

## Quickstart (Under 5 Commands)

```bash
# 1. Clone the repository
git clone https://github.com/koi-bito/yieldsense.git
cd yieldsense

# 2. Set up virtual environment and install dependencies
python -m venv venv
./venv/Scripts/activate     # Windows (or: source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt

# 3. Run the complete pipeline (Ingest -> Analyze -> Train -> Score)
python -m src.ingestion && python -m src.analysis && python -m src.models

# 4. Generate the validated AI engineering report
python src/generate_report.py

# 5. Launch the REST API service
uvicorn src.api.main:app --reload --port 8000
```

> **Deployment Note**: Currently runs with a single uvicorn worker; multi-worker deployment requires a thread-safe model manager.

Interactive Swagger documentation is available at **`http://localhost:8000/docs`**.

---

## API Documentation

### 1. `POST /analyze-lot`
Accepts parametric sensor measurements for an incoming lot, runs the pipeline (clean → scale → predict → SHAP → recovery score), and returns root-cause attributions.

**Sample Request**:
```json
{
  "lot_id": "LOT_0042",
  "sensors": {
    "sensor_060": 1.25,
    "sensor_104": -0.85,
    "sensor_511": 0.42,
    "sensor_001": 300.2
  }
}
```

**Sample Response**:
```json
{
  "lot_id": "LOT_0042",
  "prediction": "FAIL",
  "failure_probability": 0.5842,
  "recovery_score": 0.6989,
  "recovery_candidate": true,
  "top_sensors": [
    {
      "sensor": "sensor_060",
      "shap_value": 0.1842,
      "scaled_reading": 1.42,
      "impact": "INCREASES_FAIL_RISK",
      "direction": "HIGH"
    },
    {
      "sensor": "sensor_104",
      "shap_value": 0.1205,
      "scaled_reading": -1.15,
      "impact": "INCREASES_FAIL_RISK",
      "direction": "LOW"
    }
  ]
}
```

### 2. `POST /batch-analyze`
Accepts a batch of lots and returns aggregated yield metrics and candidate recovery lists ranked by salvage potential.

### 3. `GET /health`
Returns service readiness, active feature dimensions, and loaded model checkpoints.

### 4. `GET /model-info`
Returns current model type, cross-validation metrics, and Optuna hyperparameters.

---

## Docker & Containerization

Build and run using Docker Compose:

```bash
# Build and run API service
docker compose up -d

# Verify container health
curl http://localhost:8000/health
```

---

## Key Engineering Decisions

| Dilemma | Chosen Solution | Engineering Rationale |
|---|---|---|
| **Database Engine** | **DuckDB** | Columnar embedded OLAP database; 10x faster than SQLite on wide 590-column numeric matrices without needing client-server setup. |
| **Scaler Placement** | **Phase 3 (Train Fold Only)** | Fitting `StandardScaler` during ingestion introduces data leakage across holdout splits. Fit strictly on train split and saved to `models/scaler.pkl`. |
| **Model Selection** | **XGBoost over Neural Net** | High-dimensional tabular data with severe imbalance favors gradient boosted trees. Provides native handling of non-linear interactions and exact `TreeSHAP` computation. |
| **Explainability** | **TreeSHAP over LIME** | SHAP fulfills game-theoretic efficiency and consistency; `TreeExplainer` computes exact global and local attributions in polynomial time. |
| **Hallucination Detection** | **Deterministic Regex & Whitelist Layer** | LLMs inevitably hallucinate floating-point numbers and feature names. The validator audits all generated text against ground-truth context before publication. |
| **Concurrency / Worker Model** | **Single Uvicorn Worker** | Runs with a single uvicorn worker; multi-worker deployment requires a thread-safe model manager. |

---

## Testing & Verification

Run the full automated test suite with coverage:

```bash
pytest --cov=src --cov-report=term-missing tests/
```

Test coverage includes:
- `test_ingestion.py`: Schema validation, null handling, label preservation, DuckDB roundtrips.
- `test_analysis.py`: Correlation ranks, PCA variance thresholds, SPC 3-sigma detection, recovery rules.
- `test_models.py`: Leak-free SMOTE splitting, classification metrics, monotonic recovery scoring.
- `test_api.py`: FastAPI endpoints, error handling, batch aggregation, input schema validation.
- `test_agent.py`: Context assembly, mock report generation, and hallucination rejection tests.

---

## Deliverables

- Detailed methodology write-up: [`METHODOLOGY.md`](file:///c:/GITHUB%20Repos/YieldSense/METHODOLOGY.md)
- Dataset documentation: [`DATASET.md`](file:///c:/GITHUB%20Repos/YieldSense/DATASET.md)
- Pre-generated sample report: [`reports/sample_engineering_report.md`](file:///c:/GITHUB%20Repos/YieldSense/reports/sample_engineering_report.md)
