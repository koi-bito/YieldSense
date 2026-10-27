"""SHAP feature attribution and explainability for XGBoost predictions."""

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import shap
from xgboost import XGBClassifier

from src.config import (
    DB_PATH,
    SCALER_PATH,
    SHAP_EXPLANATIONS_PATH,
    XGB_MODEL_PATH,
)
from src.ingestion.features import get_sensor_columns, load_scaler
from src.ingestion.loader import load_from_duckdb


def compute_shap_explanations(
    model: XGBClassifier,
    X_scaled: np.ndarray,
    sensor_names: List[str],
    lot_ids: List[str],
    top_n_features: int = 5,
) -> Dict[str, Any]:
    """Compute per-lot SHAP feature attributions using TreeExplainer.

    Args:
        model: Trained XGBoost classifier.
        X_scaled: Scaled feature matrix (rows = lots, cols = sensors).
        sensor_names: List of sensor column names.
        lot_ids: Identifier for each row in X_scaled.
        top_n_features: Number of top driving sensors to extract per lot.

    Returns:
        Dict[str, Any]: Structured dictionary mapping lot_id to its top contributing sensors.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # In binary classification, shap_values is a 2D array [n_samples, n_features]
    if isinstance(shap_values, list):
        shap_matrix = shap_values[1]
    else:
        shap_matrix = shap_values

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    per_lot_explanations: Dict[str, Any] = {}

    for idx, lot_id in enumerate(lot_ids):
        row_shap = shap_matrix[idx]
        row_features = X_scaled[idx]

        # Rank sensors by absolute SHAP impact
        top_indices = np.argsort(np.abs(row_shap))[::-1][:top_n_features]

        contributions = []
        for feat_idx in top_indices:
            s_name = sensor_names[feat_idx]
            s_shap = float(row_shap[feat_idx])
            s_val = float(row_features[feat_idx])

            # Direction relative to mean: high (>0 z-score), low (<0 z-score), or nominal (at mean)
            if s_val > 0.0:
                direction = "HIGH"
            elif s_val < 0.0:
                direction = "LOW"
            else:
                direction = "NOMINAL"

            contributions.append(
                {
                    "sensor": s_name,
                    "shap_value": round(s_shap, 5),
                    "scaled_reading": round(s_val, 4),
                    "impact": (
                        "INCREASES_FAIL_RISK" if s_shap > 0 else "DECREASES_FAIL_RISK"
                    ),
                    "direction": direction,
                }
            )

        per_lot_explanations[lot_id] = {
            "prediction": "FAIL" if predictions[idx] == 1 else "PASS",
            "failure_probability": round(float(probabilities[idx]), 4),
            "top_contributing_sensors": contributions,
        }

    return per_lot_explanations


def explain_single_lot(
    model: XGBClassifier,
    scaler: Any,
    sensor_names: List[str],
    sensor_values: Dict[str, float],
    top_n_features: int = 5,
) -> Dict[str, Any]:
    """Explain a single lot's prediction using SHAP (for real-time inference API).

    Args:
        model: Trained XGBoost classifier.
        scaler: Fitted StandardScaler.
        sensor_names: Expected list of sensor names in exact order.
        sensor_values: Dict mapping sensor name to measured numeric value.
        top_n_features: Top N sensors to return.

    Returns:
        Dict[str, Any]: Prediction, probability, and top SHAP feature attributions.
    """
    # Build feature vector matching model's expected column order
    raw_vals = []
    for idx, s in enumerate(sensor_names):
        if s in sensor_values and sensor_values[s] is not None:
            raw_vals.append(float(sensor_values[s]))
        elif hasattr(scaler, "mean_") and idx < len(scaler.mean_):
            raw_vals.append(float(scaler.mean_[idx]))
        else:
            raw_vals.append(0.0)
    raw_vector = np.array([raw_vals], dtype=float)
    scaled_vector = scaler.transform(raw_vector)

    prob = float(model.predict_proba(scaled_vector)[0, 1])
    pred = "FAIL" if prob >= 0.5 else "PASS"

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(scaled_vector)

    if isinstance(shap_vals, list):
        shap_row = shap_vals[1][0]
    elif shap_vals.ndim == 2:
        shap_row = shap_vals[0]
    else:
        shap_row = shap_vals

    top_idx = np.argsort(np.abs(shap_row))[::-1][:top_n_features]

    top_sensors = []
    for idx in top_idx:
        s_name = sensor_names[idx]
        s_shap = float(shap_row[idx])
        s_val = float(scaled_vector[0, idx])
        if s_val > 0.0:
            direction = "HIGH"
        elif s_val < 0.0:
            direction = "LOW"
        else:
            direction = "NOMINAL"

        top_sensors.append(
            {
                "sensor": s_name,
                "shap_value": round(s_shap, 4),
                "scaled_reading": round(s_val, 4),
                "impact": (
                    "INCREASES_FAIL_RISK" if s_shap > 0 else "DECREASES_FAIL_RISK"
                ),
                "direction": direction,
            }
        )

    return {
        "prediction": pred,
        "failure_probability": round(prob, 4),
        "top_contributing_sensors": top_sensors,
    }


def run_shap_analysis(
    db_path: Path = DB_PATH,
    output_path: Path = SHAP_EXPLANATIONS_PATH,
) -> Dict[str, Any]:
    """Generate SHAP explanations for clean lots and save to JSON.

    Args:
        db_path: Path to DuckDB database.
        output_path: Destination JSON path.

    Returns:
        Dict[str, Any]: Dictionary of per-lot explanations.
    """
    df = load_from_duckdb("clean_lots", db_path=db_path)
    sensor_names = get_sensor_columns(df)
    lot_ids = df["lot_id"].tolist()

    scaler = load_scaler(SCALER_PATH)
    model = joblib.load(XGB_MODEL_PATH)

    X_raw = df[sensor_names].values
    X_scaled = scaler.transform(X_raw)

    explanations = compute_shap_explanations(
        model=model,
        X_scaled=X_scaled,
        sensor_names=sensor_names,
        lot_ids=lot_ids,
        top_n_features=5,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2)

    return explanations


if __name__ == "__main__":
    exps = run_shap_analysis()
    print(f"Generated SHAP explanations for {len(exps)} lots.")
    first_lot = next(iter(exps))
    print(f"Sample explanation for {first_lot}:", exps[first_lot])
