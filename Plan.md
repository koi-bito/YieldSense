# YieldSense — Phase-Wise Project Plan

**AI-Enabled NAND Test Yield & Recovery Analytics Pipeline**
_Reverse-engineered for Micron JR108915 — MSB Test Solutions Engineering Internship_

---

## Overview

YieldSense is a production-grade data and ML engineering pipeline that ingests semiconductor manufacturing test data, identifies failure-mode signals via statistical analysis, predicts recoverable failing units using machine learning, and generates a validated, AI-assisted engineering report. The project directly mirrors the scope, deliverables, and skill requirements of the Micron SpecTek internship project.

**Tech stack:** Python, Pandas, Scikit-learn, XGBoost, SHAP, FastAPI, Docker, GitHub Actions, SQLite/DuckDB, Claude API, Pytest

**Dataset:** UCI SECOM Dataset (590 sensor readings per wafer, binary pass/fail labels) — publicly available semiconductor manufacturing test data

**End product:** A containerized REST API wrapping the full analytics pipeline, a CLI report generator producing a structured engineering report, and a fully documented GitHub repository.

---

## Phase 0 — Repository & Environment Setup

**Goal:** Establish the project skeleton, tooling, and reproducibility baseline before writing any analytical code.

### Tasks

- Initialize GitHub repository: `yieldsense` under the `koi-bito` handle
- Set up project structure:
  ```
  yieldsense/
  ├── data/
  │   ├── raw/              # SECOM dataset files (gitignored if large)
  │   └── processed/        # Cleaned, feature-engineered data
  ├── notebooks/            # Exploratory and analysis notebooks
  ├── src/
  │   ├── ingestion/        # Data loading and cleaning scripts
  │   ├── analysis/         # Statistical analysis modules
  │   ├── models/           # ML model training and evaluation
  │   ├── agent/            # LLM reporting agent
  │   └── api/              # FastAPI application
  ├── tests/                # Pytest test suite
  ├── reports/              # Auto-generated engineering reports (output)
  ├── Dockerfile
  ├── docker-compose.yml
  ├── requirements.txt
  ├── .github/workflows/    # CI pipeline
  └── README.md
  ```
- Configure Python virtual environment (`venv` or `conda`), pin all dependencies in `requirements.txt`
- Set up `pre-commit` hooks: Black (formatting), Flake8 (linting), isort (import sorting)
- Download and place the SECOM dataset (`secom.data`, `secom_labels.data`) under `data/raw/`
- Write a brief `DATASET.md` documenting the dataset source, license, and column semantics
- Create `.env.example` for environment variables (API keys, DB path, model path)

### Deliverable

A clean, importable repo with enforced formatting and a reproducible environment that any reviewer can clone and run.

---

## Phase 1 — Data Ingestion & Preparation

**Goal:** Build a reliable, documented ingestion pipeline that loads raw SECOM data, handles its known quality issues, and persists a clean version to a local database.

### Context

The SECOM dataset has ~590 sensor columns, ~1567 rows, significant missing values (some columns >40% null), and severe class imbalance (~93% pass, ~7% fail). This is intentional — it mirrors real manufacturing data quality.

### Tasks

**Ingestion script (`src/ingestion/loader.py`)**

- Load `secom.data` (space-delimited, no header) and `secom_labels.data` (label + timestamp)
- Merge on row index, assign meaningful column names (`sensor_001` through `sensor_590`, `label`, `timestamp`)
- Persist raw merged dataframe to SQLite/DuckDB as table `raw_lots`

**Cleaning script (`src/ingestion/cleaner.py`)**

- Drop columns with more than 40% missing values (document the threshold and rationale in code comments)
- Impute remaining missing values with column median (manufacturing convention — document why not mean)
- Remove near-zero variance features (sensors that never change carry no signal)
- Log every dropped column and imputation decision to a structured cleaning report (`data/processed/cleaning_log.json`)
- Persist cleaned dataframe as table `clean_lots`

**Feature engineering (`src/ingestion/features.py`)**

- Normalize sensor readings (StandardScaler) — fit on training split only to prevent data leakage
- Create a `lot_id` synthetic key for tracking individual lots through the pipeline
- Persist the fitted scaler as `models/scaler.pkl` for use at inference time

**Tests (`tests/test_ingestion.py`)**

- Assert output schema matches expected columns
- Assert no nulls remain in cleaned data
- Assert label distribution is preserved after cleaning

### Deliverable

`clean_lots` table in the local database, a cleaning log, a fitted scaler, and passing ingestion tests.

---

## Phase 2 — Statistical Analysis

**Goal:** Produce structured, reproducible statistical analyses that identify which sensor parameters correlate with test failures and where manufacturing drift is occurring. This is the "analytical tools and visualizations to study manufacturing data" deliverable.

### Tasks

**Correlation analysis (`src/analysis/correlation.py`)**

- Compute point-biserial correlation between each sensor and the pass/fail label
- Rank sensors by absolute correlation coefficient
- Output a ranked CSV: `reports/sensor_correlation_ranking.csv`

**Principal Component Analysis (`src/analysis/pca.py`)**

- Run PCA on the cleaned sensor matrix
- Determine components needed to explain 95% variance
- Output explained variance ratios and top sensor loadings per component
- Save PCA model as `models/pca.pkl`

**Statistical Process Control — Control Charts (`src/analysis/spc.py`)**

- For the top 10 correlated sensors, compute rolling mean and standard deviation over time (using the timestamp column)
- Flag lots where a sensor reading exceeds 3-sigma bounds as "drift events"
- Output a structured drift event log: `reports/spc_drift_events.csv`

**Recovery candidate identification (`src/analysis/recovery.py`)**

- Define "recovery candidate" logic: failing lots where fewer than N sensors are out-of-spec AND all out-of-spec readings are within a configurable margin of the pass boundary
- Output a candidate list: `reports/recovery_candidates.csv` with lot IDs and the sensors responsible

**Exploratory notebook (`notebooks/01_statistical_analysis.ipynb`)**

- Walk through all of the above with inline plots (matplotlib/seaborn): failure Pareto chart, sensor correlation heatmap, PCA scree plot, control charts for top sensors
- This notebook is documentation, not production code — it should be readable as a standalone analysis

### Deliverable

Four structured CSV report files, a fitted PCA model, and a documented analysis notebook.

---

## Phase 3 — Machine Learning Model

**Goal:** Train a validated classifier that predicts pass/fail and, critically, ranks failing lots by recovery likelihood using predicted probabilities and SHAP feature attribution.

### Tasks

**Train/test split (`src/models/train.py`)**

- Stratified 80/20 split to preserve the class imbalance ratio
- Apply SMOTE (Synthetic Minority Oversampling) on the training set only to handle imbalance — document the rationale and the risk of oversampling before splitting

**Model training**

- Train two models: Logistic Regression (baseline, interpretable) and XGBoost (primary)
- Hyperparameter tune XGBoost with Optuna or GridSearchCV — log best params to `models/xgb_params.json`
- Evaluate both on the test set: precision, recall, F1, ROC-AUC — log to `reports/model_evaluation.json`
- Save final XGBoost model as `models/xgb_model.pkl`

**SHAP explainability (`src/models/explain.py`)**

- Compute SHAP values for all test-set predictions using `shap.TreeExplainer`
- For each predicted-fail lot, rank the top 5 contributing sensors by |SHAP value|
- For recovery candidates (from Phase 2), generate a SHAP summary showing how close they are to the decision boundary
- Output `reports/shap_lot_explanations.json`: per-lot sensor attribution

**Recovery scoring (`src/models/recovery_scorer.py`)**

- Use predicted failure probability as a proxy for recovery likelihood: a lot predicted fail with probability 0.55 is more recoverable than one at 0.98
- Combine probability score + SHAP feature count into a composite `recovery_score`
- Output `reports/recovery_ranking.csv`: all failing lots ranked by recovery score with top contributing sensors

**Tests (`tests/test_models.py`)**

- Assert model file exists and loads cleanly
- Assert predictions are binary (0/1)
- Assert SHAP output contains expected lot IDs
- Assert recovery ranking CSV is non-empty and sorted correctly

**Model notebook (`notebooks/02_model_training.ipynb`)**

- Document the full training process, hyperparameter search results, evaluation metrics, SHAP summary plots, and a plain-English interpretation of the top failure-driving sensors

### Deliverable

Trained XGBoost model, SHAP explanation file, recovery ranking CSV, model evaluation metrics, and a documented model notebook.

---

## Phase 4 — FastAPI Service

**Goal:** Wrap the pipeline in a production-style REST API that accepts new lot data and returns predictions, SHAP attributions, and recovery scores. This is the "automation solution" deliverable.

### Tasks

**API application (`src/api/main.py`)**

- `POST /analyze-lot` — accepts a JSON payload of sensor readings for a single lot, runs the full pipeline (clean → scale → predict → SHAP → recovery score), returns:
  ```json
  {
    "lot_id": "LOT_001",
    "prediction": "FAIL",
    "failure_probability": 0.73,
    "recovery_score": 0.41,
    "recovery_candidate": true,
    "top_sensors": [
      { "sensor": "sensor_034", "shap_value": 0.21, "direction": "high" },
      { "sensor": "sensor_112", "shap_value": 0.18, "direction": "low" }
    ]
  }
  ```
- `POST /batch-analyze` — accepts a list of lots, runs the pipeline in batch, returns a ranked recovery list
- `GET /health` — returns service status and model version
- `GET /model-info` — returns model evaluation metrics and training date

**Input validation**

- Pydantic models for all request/response schemas
- Validate that incoming sensor count matches the trained model's expected feature count
- Return structured error responses (not raw exceptions)

**Tests (`tests/test_api.py`)**

- Use `TestClient` from FastAPI to test all endpoints
- Test with valid input, missing sensors, and malformed payloads
- Assert response schema matches Pydantic models

### Deliverable

A fully functional FastAPI service with documented endpoints, input validation, and passing API tests.

---

## Phase 5 — Containerization & CI/CD

**Goal:** Make the project reproducible, deployable, and professionally maintained — this directly mirrors the Docker + GitHub Actions stack from FlyRank.

### Tasks

**Dockerfile**

- Multi-stage build: builder stage installs dependencies, final stage copies only what's needed
- Non-root user for the container
- Copy model artifacts and data into the image at build time
- Expose port 8000

**docker-compose.yml**

- Service: `api` (the FastAPI app)
- Volume mount for `reports/` so generated reports are accessible on the host
- Environment variable passthrough for API keys

**GitHub Actions CI (`.github/workflows/ci.yml`)**

- Trigger: push and pull request to `main`
- Steps:
  1. Checkout code
  2. Set up Python
  3. Install dependencies
  4. Run linting (Flake8)
  5. Run full test suite (Pytest) with coverage report
  6. Build Docker image (confirm it builds without errors)
- Badge in README showing CI status

**Makefile (optional but recommended)**

- `make install` — set up environment
- `make test` — run full test suite
- `make run` — start API locally
- `make report` — run the report generator (Phase 6)
- `make docker-build` — build Docker image

### Deliverable

A containerized, CI-tested service that any engineer can clone, build, and run with two commands.

---

## Phase 6 — LLM Reporting Agent with Validation Layer

**Goal:** Build an LLM agent that takes the structured outputs from Phases 2 and 3 and drafts a plain-English engineering analysis, with a rules-based validation layer that flags any AI-generated claim not supported by the underlying data. This is the "responsible AI" and "validate AI-generated outputs" differentiator.

### Tasks

**Report assembler (`src/agent/assembler.py`)**

- Load all structured outputs: cleaning log, correlation ranking, SPC drift events, model evaluation, SHAP explanations, recovery ranking
- Assemble a structured context object (JSON) summarising all findings — this is what gets passed to the LLM, not raw data

**LLM report drafter (`src/agent/drafter.py`)**

- Call Claude API with the structured context and a system prompt instructing it to draft:
  - Executive summary (2–3 sentences)
  - Key failure-driving parameters with their correlation/SHAP evidence
  - Recovery opportunity summary (how many candidates, estimated recovery rate)
  - SPC drift observations
  - Methodology notes and assumptions
  - Limitations section
  - Proposed next steps
- The prompt explicitly instructs the model to cite the specific numbers from the context for every claim it makes (e.g., "Sensor 034 shows a correlation of 0.43 with failures" — not a vague statement)

**Validation layer (`src/agent/validator.py`)**

- Parse the LLM-generated report and extract all numeric claims (regex + structured parsing)
- Cross-reference each extracted claim against the source context object
- Flag claims where the number in the report does not match the source data within a tolerance
- Flag claims that reference sensors or metrics not present in the context at all (hallucination detection — draw on Deccan AI experience here)
- Output a validation result: `PASSED`, `WARNINGS` (some claims unverified), or `FAILED` (hallucinated data detected)
- Append a machine-readable validation block to the final report

**Report writer (`src/agent/writer.py`)**

- Combine the LLM draft + validation block into a final structured markdown report
- Save to `reports/engineering_report_<timestamp>.md`
- Include a header block: generated timestamp, model version, dataset info, validation status

**CLI entry point (`src/generate_report.py`)**

- `python src/generate_report.py` — runs the full report generation pipeline end-to-end
- Accepts optional flags: `--lot-id` to generate a report for a specific lot, `--batch` to run on all lots in the recovery ranking

**Tests (`tests/test_agent.py`)**

- Mock the Claude API response to test the validator against known-good and known-bad LLM outputs
- Assert that a fabricated number in the LLM response is correctly flagged
- Assert that a report with all valid claims passes validation

### Deliverable

A CLI-runnable report generator that produces a validated markdown engineering report. The validation layer is the core technical contribution of this phase.

---

## Phase 7 — Documentation & GitHub Polish

**Goal:** Make the repository a professional engineering artifact that speaks for itself without any explanation needed in an interview.

### Tasks

**README.md**

- Project summary (2–3 sentences, jargon-free)
- Architecture diagram (ASCII or a simple SVG) showing data flow: raw data → ingestion → stats → ML → API / report generator
- Quickstart: clone, install, run API, generate report — should work in under 5 commands
- Endpoint documentation (request/response examples for each API route)
- Dataset section: what SECOM is, why it maps to NAND test-flow data, where to download it
- Design decisions section: why XGBoost over a neural net, why SHAP over LIME, why the validation layer works the way it does — this is what you talk to in an interview
- Limitations and future work section (honest, specific — mirrors what the JD asks for in the final report)
- CI badge

**Inline documentation**

- Every module has a module-level docstring explaining its role in the pipeline
- Every public function has a docstring with Args, Returns, and a one-line description
- Non-obvious logic has inline comments explaining the "why", not the "what"

**`METHODOLOGY.md`**

- A standalone document (separate from README) walking through the statistical and ML choices in depth
- This is the "technical report covering methodology, findings, validation results, limitations, and proposed next steps" that the JD lists as a deliverable — write it as if submitting it to SpecTek engineers
- Sections: Problem framing, Dataset and preprocessing decisions, Statistical analysis approach, ML model selection and evaluation, Recovery scoring methodology, AI-assisted reporting and validation approach, Limitations, Proposed next steps

**Sample report**

- Commit one pre-generated `reports/sample_engineering_report.md` to the repo so reviewers can read the end product without running the pipeline

**Final repo checklist**

- [ ] All tests pass (`pytest` exits 0)
- [ ] Docker image builds cleanly
- [ ] CI badge is green
- [ ] No hardcoded API keys or local paths anywhere
- [ ] `.env.example` documents all required environment variables
- [ ] Sample report is present and readable
- [ ] README quickstart works end-to-end on a fresh clone

---

## End Product Summary

| Artifact                               | Description                                                              |
| -------------------------------------- | ------------------------------------------------------------------------ |
| `src/`                                 | Full pipeline source code — ingestion, stats, ML, API, agent             |
| `tests/`                               | Pytest suite covering all pipeline stages                                |
| `Dockerfile` + `docker-compose.yml`    | Containerized, deployable service                                        |
| `.github/workflows/ci.yml`             | Automated CI on every push                                               |
| `POST /analyze-lot` API                | Accepts sensor data, returns prediction + SHAP + recovery score          |
| `POST /batch-analyze` API              | Batch lot analysis with ranked recovery output                           |
| `python src/generate_report.py`        | CLI that produces a validated engineering report                         |
| `reports/sample_engineering_report.md` | Pre-generated example of the full report output                          |
| `METHODOLOGY.md`                       | Deep-dive technical document — maps 1:1 to JD's final report deliverable |
| `README.md`                            | Professional repo documentation with quickstart and design decisions     |

---

## Resume Bullet (draft — refine after build)

> Built YieldSense, a containerized ML pipeline for semiconductor test-yield analytics; applied XGBoost with SHAP attribution to rank NAND lot recovery candidates on the UCI SECOM dataset, achieving 0.87 ROC-AUC; built a FastAPI service wrapping the pipeline and an LLM reporting agent with a rules-based hallucination-detection layer that cross-validates all AI-generated claims against source data before report generation.
