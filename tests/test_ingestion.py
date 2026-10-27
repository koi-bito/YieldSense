"""Unit and integration tests for data ingestion and cleaning."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.cleaner import clean_secom_data
from src.ingestion.features import get_sensor_columns, split_features_and_targets
from src.ingestion.loader import load_from_duckdb, load_raw_secom, save_to_duckdb


def test_load_raw_secom():
    """Verify raw loader correctly ingests SECOM data and labels."""
    df = load_raw_secom()
    assert df.shape[0] == 1567
    assert df.shape[1] == 593  # 590 sensors + lot_id + label + timestamp
    assert "lot_id" in df.columns
    assert "label" in df.columns
    assert "timestamp" in df.columns
    assert df["lot_id"].iloc[0] == "LOT_0001"
    assert set(df["label"].unique()).issubset({0, 1})
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_clean_secom_data():
    """Verify cleaner removes nulls, drops high-missing and zero-variance columns."""
    raw_df = load_raw_secom()
    clean_df, log_data = clean_secom_data(raw_df, missing_threshold=0.40)

    # 1. Assert row count preserved
    assert len(clean_df) == len(raw_df)

    # 2. Assert no nulls remain in sensor features
    sensor_cols = get_sensor_columns(clean_df)
    assert clean_df[sensor_cols].isnull().sum().sum() == 0

    # 3. Assert high missing columns were dropped
    assert log_data["dropped_high_missing_count"] > 0
    for col in log_data["dropped_high_missing_columns"]:
        assert col not in clean_df.columns

    # 4. Assert zero variance columns were dropped
    assert log_data["dropped_zero_variance_count"] >= 0
    for col in log_data["dropped_zero_variance_columns"]:
        assert col not in clean_df.columns

    # 5. Assert label distribution preserved
    assert (clean_df["label"] == 1).sum() == (raw_df["label"] == 1).sum()
    assert (clean_df["label"] == 0).sum() == (raw_df["label"] == 0).sum()

    # 6. Check audit log keys
    assert "initial_sensor_count" in log_data
    assert "final_sensor_count" in log_data
    assert log_data["final_sensor_count"] == len(sensor_cols)


def test_duckdb_roundtrip():
    """Test saving and loading DataFrame to/from DuckDB."""
    sample_df = pd.DataFrame(
        {
            "lot_id": ["LOT_0001", "LOT_0002"],
            "sensor_001": [1.23, 4.56],
            "label": [0, 1],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "test.duckdb"
        save_to_duckdb(sample_df, table_name="test_lots", db_path=test_db)
        loaded_df = load_from_duckdb("test_lots", db_path=test_db)
        pd.testing.assert_frame_equal(sample_df, loaded_df)

        with pytest.raises(ValueError, match="Unauthorized table name"):
            save_to_duckdb(sample_df, table_name="users; DROP TABLE test_lots;", db_path=test_db)

        with pytest.raises(ValueError, match="Unauthorized table name"):
            load_from_duckdb("malicious_table", db_path=test_db)


def test_split_features_and_targets():
    """Test separating X, y, and metadata."""
    sample_df = pd.DataFrame(
        {
            "lot_id": ["LOT_0001", "LOT_0002"],
            "timestamp": pd.to_datetime(["2008-07-19 11:55:00", "2008-07-19 12:32:00"]),
            "sensor_001": [1.0, 2.0],
            "sensor_002": [3.0, 4.0],
            "label": [0, 1],
        }
    )

    X, y, meta = split_features_and_targets(sample_df)
    assert list(X.columns) == ["sensor_001", "sensor_002"]
    assert list(y) == [0, 1]
    assert list(meta["lot_id"]) == ["LOT_0001", "LOT_0002"]
