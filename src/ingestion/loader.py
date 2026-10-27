"""SECOM raw dataset loader and DuckDB persistence."""

from pathlib import Path

import duckdb
import pandas as pd

from src.config import DB_PATH, RAW_DATA_PATH, RAW_LABELS_PATH


def load_raw_secom(
    data_path: Path = RAW_DATA_PATH,
    labels_path: Path = RAW_LABELS_PATH,
) -> pd.DataFrame:
    """Load raw SECOM data and labels into a merged pandas DataFrame.

    Args:
        data_path: Path to space-delimited sensor measurements (secom.data).
        labels_path: Path to label and timestamp data (secom_labels.data).

    Returns:
        pd.DataFrame: Merged raw dataset with sensor_001..sensor_590,
                      label (0=pass, 1=fail), timestamp, and lot_id.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Sensor data file not found at {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels data file not found at {labels_path}")

    # Load sensor measurements: 590 space-separated numeric columns, no header
    sensor_df = pd.read_csv(data_path, sep=r"\s+", header=None)
    num_sensors = sensor_df.shape[1]
    sensor_df.columns = [f"sensor_{i:03d}" for i in range(1, num_sensors + 1)]

    # Load labels and timestamps: col 0 = label (-1 pass, 1 fail), col 1 = timestamp
    labels_df = pd.read_csv(
        labels_path,
        sep=r"\s+",
        header=None,
        names=["raw_label", "timestamp"],
        quotechar='"',
    )

    # Convert timestamps
    labels_df["timestamp"] = pd.to_datetime(
        labels_df["timestamp"], format="%d/%m/%Y %H:%M:%S"
    )

    # Encode label: -1 (pass) -> 0, 1 (fail) -> 1
    labels_df["label"] = (labels_df["raw_label"] == 1).astype(int)

    # Assign synthetic tracking key LOT_0001 ...
    lot_series = pd.Series(
        [f"LOT_{i+1:04d}" for i in range(len(sensor_df))], name="lot_id"
    )

    # Merge into single unfragmented DataFrame
    merged_df = pd.concat(
        [lot_series, sensor_df, labels_df[["label", "timestamp"]]], axis=1
    )

    return merged_df


ALLOWED_TABLES = {"raw_lots", "clean_lots", "test_lots"}


def save_to_duckdb(
    df: pd.DataFrame,
    table_name: str,
    db_path: Path = DB_PATH,
    overwrite: bool = True,
) -> None:
    """Persist a DataFrame to a DuckDB table.

    Args:
        df: Pandas DataFrame to persist.
        table_name: Destination table name.
        db_path: Path to DuckDB file.
        overwrite: If True, replaces existing table.
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(
            f"Unauthorized table name '{table_name}'. Must be one of {ALLOWED_TABLES}."
        )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        if overwrite:
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
        else:
            con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
    finally:
        con.close()


def load_from_duckdb(table_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Read a table from DuckDB into a pandas DataFrame.

    Args:
        table_name: Name of table to read.
        db_path: Path to DuckDB database file.

    Returns:
        pd.DataFrame: Retrieved table data.
    """
    if table_name not in ALLOWED_TABLES:
        raise ValueError(
            f"Unauthorized table name '{table_name}'. Must be one of {ALLOWED_TABLES}."
        )
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {table_name}").df()
    finally:
        con.close()


def run_loader(
    data_path: Path = RAW_DATA_PATH,
    labels_path: Path = RAW_LABELS_PATH,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Run full ingestion load step and persist table 'raw_lots' to DuckDB.

    Returns:
        pd.DataFrame: Merged raw dataframe.
    """
    df = load_raw_secom(data_path=data_path, labels_path=labels_path)
    save_to_duckdb(df, table_name="raw_lots", db_path=db_path, overwrite=True)
    return df


if __name__ == "__main__":
    print("Executing SECOM data loader...")
    raw_df = run_loader()
    print(
        f"Loaded {raw_df.shape[0]} lots with {raw_df.shape[1]} columns into DuckDB 'raw_lots'."
    )
    print(f"Fail rate: {(raw_df['label'] == 1).mean():.2%}")
