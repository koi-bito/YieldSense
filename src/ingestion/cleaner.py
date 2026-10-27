"""Data cleaning, missing value handling, and zero-variance feature removal."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.config import (
    CLEANING_LOG_PATH,
    DB_PATH,
    MISSING_THRESHOLD,
    VARIANCE_THRESHOLD,
)
from src.ingestion.loader import load_from_duckdb, save_to_duckdb


def clean_secom_data(
    df: pd.DataFrame,
    missing_threshold: float = MISSING_THRESHOLD,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw semiconductor sensor readings.

    Processing steps:
    1. Identify sensor columns (sensor_001..sensor_590).
    2. Drop columns exceeding missingness threshold (default 40%). Rationale:
       Features with high missingness introduce excessive noise when imputed.
    3. Impute remaining missing sensor values with median. Rationale:
       Median is robust to extreme sensor anomalies and sensor drift, whereas
       mean is heavily distorted by measurement spikes.
    4. Remove zero or near-zero variance features. Rationale:
       Sensors with zero variance across all lots carry no discriminative power.
    5. Construct structured cleaning metadata log.

    Args:
        df: Merged raw DataFrame containing sensor columns, label, timestamp, lot_id.
        missing_threshold: Maximum allowable fraction of missing values [0.0, 1.0].
        variance_threshold: Minimum allowable variance.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and cleaning metadata log.
    """
    cleaned_df = df.copy()

    # Identify metadata columns vs sensor columns
    meta_cols = [c for c in ["lot_id", "timestamp", "label"] if c in cleaned_df.columns]
    sensor_cols = [c for c in cleaned_df.columns if c not in meta_cols]

    initial_sensor_count = len(sensor_cols)
    total_lots = len(cleaned_df)

    # 1. Missingness analysis & dropping
    missing_fraction = cleaned_df[sensor_cols].isnull().mean()
    high_missing_cols = missing_fraction[
        missing_fraction > missing_threshold
    ].index.tolist()
    high_missing_info = {
        col: round(float(missing_fraction[col]), 4) for col in high_missing_cols
    }

    # Drop high missing sensor columns
    cleaned_df = cleaned_df.drop(columns=high_missing_cols)
    remaining_sensors = [c for c in sensor_cols if c not in high_missing_cols]

    # 2. Median imputation for remaining sensor columns
    imputation_map: Dict[str, float] = {}
    for col in remaining_sensors:
        median_val = float(cleaned_df[col].median())
        # If an entire column was NaN (caught by high_missing, but safety check)
        if np.isnan(median_val):
            median_val = 0.0
        imputation_map[col] = median_val
        cleaned_df[col] = cleaned_df[col].fillna(median_val)

    # 3. Near-zero variance filtering
    variances = cleaned_df[remaining_sensors].var(ddof=0)
    zero_var_cols = variances[variances <= variance_threshold].index.tolist()

    cleaned_df = cleaned_df.drop(columns=zero_var_cols)
    final_sensors = [c for c in remaining_sensors if c not in zero_var_cols]

    # 4. Construct audit log
    cleaning_log = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_lots": total_lots,
        "initial_sensor_count": initial_sensor_count,
        "missing_threshold": missing_threshold,
        "dropped_high_missing_count": len(high_missing_cols),
        "dropped_high_missing_columns": high_missing_info,
        "imputed_columns_count": len(remaining_sensors),
        "imputation_strategy": "median",
        "variance_threshold": variance_threshold,
        "dropped_zero_variance_count": len(zero_var_cols),
        "dropped_zero_variance_columns": zero_var_cols,
        "final_sensor_count": len(final_sensors),
        "final_columns_count": len(cleaned_df.columns),
        "label_distribution": {
            "pass_count": (
                int((cleaned_df["label"] == 0).sum()) if "label" in cleaned_df else None
            ),
            "fail_count": (
                int((cleaned_df["label"] == 1).sum()) if "label" in cleaned_df else None
            ),
            "fail_rate": (
                float((cleaned_df["label"] == 1).mean())
                if "label" in cleaned_df
                else None
            ),
        },
    }

    return cleaned_df, cleaning_log


def run_cleaner(
    db_path: Path = DB_PATH,
    log_path: Path = CLEANING_LOG_PATH,
    missing_threshold: float = MISSING_THRESHOLD,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load 'raw_lots' from DuckDB, clean, persist 'clean_lots', and write cleaning log.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and audit dictionary.
    """
    raw_df = load_from_duckdb("raw_lots", db_path=db_path)
    clean_df, log_data = clean_secom_data(
        raw_df,
        missing_threshold=missing_threshold,
        variance_threshold=variance_threshold,
    )

    # Save cleaned table to DuckDB
    save_to_duckdb(clean_df, table_name="clean_lots", db_path=db_path, overwrite=True)

    # Persist audit JSON
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    return clean_df, log_data


if __name__ == "__main__":
    clean_df, log_info = run_cleaner()
    print("Data cleaning completed.")
    print(
        f"Sensors retained: {log_info['final_sensor_count']} of {log_info['initial_sensor_count']}"
    )
    print(f"Dropped for >40% missingness: {log_info['dropped_high_missing_count']}")
    print(f"Dropped for zero variance: {log_info['dropped_zero_variance_count']}")
