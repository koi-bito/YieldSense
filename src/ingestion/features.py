"""Feature extraction and preprocessing utilities."""

from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import SCALER_PATH


def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """Retrieve all sensor measurement column names sorted in order."""
    return [c for c in df.columns if c.startswith("sensor_")]


def split_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split clean lots DataFrame into features X, target y, and lot metadata.

    Args:
        df: DataFrame containing sensor columns, label, and metadata.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
            X (sensors only), y (label: 0 or 1), metadata (lot_id, timestamp).
    """
    sensor_cols = get_sensor_columns(df)
    X = df[sensor_cols].copy()

    if "label" in df.columns:
        y = df["label"].astype(int).copy()
    else:
        y = pd.Series(index=df.index, dtype=int)

    meta_cols = [c for c in ["lot_id", "timestamp"] if c in df.columns]
    meta = df[meta_cols].copy()

    return X, y, meta


def save_scaler(scaler: StandardScaler, path: Path = SCALER_PATH) -> None:
    """Save fitted StandardScaler to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path: Path = SCALER_PATH) -> StandardScaler:
    """Load fitted StandardScaler from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Scaler artifact not found at {path}")
    return joblib.load(path)
