"""Centralized configuration and paths for YieldSense."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Dataset Files
RAW_DATA_PATH = RAW_DATA_DIR / "secom.data"
RAW_LABELS_PATH = RAW_DATA_DIR / "secom_labels.data"
CLEANING_LOG_PATH = PROCESSED_DATA_DIR / "cleaning_log.json"

# Database Configuration
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "yieldsense.duckdb")))

# Model & Scaler Artifacts
SCALER_PATH = MODELS_DIR / "scaler.pkl"
PCA_PATH = MODELS_DIR / "pca.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.pkl"
XGB_PARAMS_PATH = MODELS_DIR / "xgb_params.json"

# Report Artifacts
SENSOR_CORRELATION_PATH = REPORTS_DIR / "sensor_correlation_ranking.csv"
SPC_DRIFT_EVENTS_PATH = REPORTS_DIR / "spc_drift_events.csv"
RECOVERY_CANDIDATES_PATH = REPORTS_DIR / "recovery_candidates.csv"
MODEL_EVAL_PATH = REPORTS_DIR / "model_evaluation.json"
SHAP_EXPLANATIONS_PATH = REPORTS_DIR / "shap_lot_explanations.json"
RECOVERY_RANKING_PATH = REPORTS_DIR / "recovery_ranking.csv"

# Preprocessing Thresholds
MISSING_THRESHOLD = 0.40  # Drop features with >40% missing values
VARIANCE_THRESHOLD = 0.0  # Drop zero/near-zero variance features

# Reproducibility & Model Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Recovery Candidate Definition
RECOVERY_MAX_OOB_SENSORS = int(os.getenv("RECOVERY_MAX_OOB_SENSORS", "3"))
RECOVERY_MARGIN_STD_FACTOR = float(os.getenv("RECOVERY_MARGIN_STD_FACTOR", "0.5"))

# API & LLM Agent
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
