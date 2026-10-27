"""Sensor correlation analysis with pass/fail manufacturing labels."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from src.config import DB_PATH, SENSOR_CORRELATION_PATH
from src.ingestion.features import get_sensor_columns
from src.ingestion.loader import load_from_duckdb


def compute_sensor_correlations(
    df: pd.DataFrame,
    target_col: str = "label",
) -> pd.DataFrame:
    """Compute point-biserial correlation and p-value for all sensors against binary target.

    Args:
        df: DataFrame containing sensor columns and the target column.
        target_col: Binary target column name (0=pass, 1=fail).

    Returns:
        pd.DataFrame: Sorted DataFrame with columns:
                      [sensor, correlation, abs_correlation, p_value, significance]
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    sensor_cols = get_sensor_columns(df)
    y = df[target_col].to_numpy()

    records = []
    for col in sensor_cols:
        x = df[col].to_numpy()
        # Point-biserial correlation is mathematically equivalent to Pearson r
        # between continuous x and binary y
        r, p_val = stats.pointbiserialr(y, x)

        # Handle edge cases (e.g., constant column)
        if np.isnan(r):
            r = 0.0
            p_val = 1.0

        records.append(
            {
                "sensor": col,
                "correlation": round(float(r), 5),
                "abs_correlation": round(abs(float(r)), 5),
                "p_value": (
                    float(f"{p_val:.3e}") if p_val < 1e-4 else round(float(p_val), 5)
                ),
                "is_significant_05": bool(p_val < 0.05),
            }
        )

    corr_df = pd.DataFrame(records)
    corr_df = corr_df.sort_values(by="abs_correlation", ascending=False).reset_index(
        drop=True
    )
    corr_df.insert(0, "rank", range(1, len(corr_df) + 1))
    return corr_df


def run_correlation_analysis(
    db_path: Path = DB_PATH,
    output_path: Path = SENSOR_CORRELATION_PATH,
) -> pd.DataFrame:
    """Load clean lots from DuckDB, compute correlations, and save CSV report.

    Args:
        db_path: Path to DuckDB database.
        output_path: Path to destination CSV.

    Returns:
        pd.DataFrame: Ranked correlation results.
    """
    df = load_from_duckdb("clean_lots", db_path=db_path)
    corr_df = compute_sensor_correlations(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(output_path, index=False)
    return corr_df


if __name__ == "__main__":
    ranked = run_correlation_analysis()
    print("Top 10 failure-correlated sensors:")
    print(
        ranked.head(10)[
            ["rank", "sensor", "correlation", "p_value", "is_significant_05"]
        ]
    )
