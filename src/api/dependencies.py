"""Stateful dependency loader and cache for YieldSense API."""

import json
from typing import Any, Dict, List, Optional

import joblib
from xgboost import XGBClassifier

from src.config import (
    CLEANING_LOG_PATH,
    MODEL_EVAL_PATH,
    SCALER_PATH,
    XGB_MODEL_PATH,
    XGB_PARAMS_PATH,
)


class ModelManager:
    """Manages cached model, scaler, and feature expectations for inference."""

    def __init__(self):
        self.model: Optional[XGBClassifier] = None
        self.scaler: Optional[Any] = None
        self.cleaning_log: Dict[str, Any] = {}
        self.eval_metrics: Dict[str, Any] = {}
        self.best_params: Dict[str, Any] = {}
        self.expected_sensors: List[str] = []
        self.sensor_medians: Dict[str, float] = {}

    def load_artifacts(self) -> None:
        """Load trained model, fitted scaler, and metadata from disk."""
        if XGB_MODEL_PATH.exists():
            self.model = joblib.load(XGB_MODEL_PATH)
        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
            if hasattr(self.scaler, "feature_names_in_"):
                self.expected_sensors = list(self.scaler.feature_names_in_)

        if CLEANING_LOG_PATH.exists():
            with open(CLEANING_LOG_PATH, "r", encoding="utf-8") as f:
                self.cleaning_log = json.load(f)

        # Fallback or supplement for expected sensors
        if not self.expected_sensors and CLEANING_LOG_PATH.exists():
            # If scaler didn't store feature names, determine from cleaning log or duckdb
            from src.ingestion.features import get_sensor_columns
            from src.ingestion.loader import load_from_duckdb

            clean_df = load_from_duckdb("clean_lots")
            self.expected_sensors = get_sensor_columns(clean_df)

        from src.config import DB_PATH

        if DB_PATH.exists():
            try:
                from src.ingestion.loader import load_from_duckdb

                clean_df = load_from_duckdb("clean_lots")
                if self.expected_sensors:
                    valid_cols = [
                        c for c in self.expected_sensors if c in clean_df.columns
                    ]
                    self.sensor_medians = clean_df[valid_cols].median().to_dict()
            except Exception:
                pass

        if MODEL_EVAL_PATH.exists():
            with open(MODEL_EVAL_PATH, "r", encoding="utf-8") as f:
                self.eval_metrics = json.load(f)

        if XGB_PARAMS_PATH.exists():
            with open(XGB_PARAMS_PATH, "r", encoding="utf-8") as f:
                self.best_params = json.load(f)

    def get_sensor_imputation_value(self, sensor_name: str) -> float:
        """Return the training median (or scaler mean / 0.0 fallback) for a sensor."""
        if sensor_name in self.sensor_medians:
            return float(self.sensor_medians[sensor_name])
        if (
            hasattr(self.scaler, "mean_")
            and self.expected_sensors
            and sensor_name in self.expected_sensors
        ):
            idx = self.expected_sensors.index(sensor_name)
            return float(self.scaler.mean_[idx])
        return 0.0

    @property
    def is_ready(self) -> bool:
        """Check if model and scaler are loaded and ready to serve."""
        return self.model is not None and self.scaler is not None


# Global singleton instance
manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Dependency provider for FastAPI route handlers."""
    if not manager.is_ready:
        manager.load_artifacts()
    return manager
