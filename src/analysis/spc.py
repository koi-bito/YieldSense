"""Statistical Process Control (SPC) control charts and drift event detection."""

from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.analysis.correlation import compute_sensor_correlations
from src.config import DB_PATH, SPC_DRIFT_EVENTS_PATH
from src.ingestion.loader import load_from_duckdb


def detect_spc_drift_events(
    df: pd.DataFrame,
    sensors: Optional[List[str]] = None,
    top_n_sensors: int = 10,
    window_size: int = 30,
    min_periods: int = 10,
    sigma_multiplier: float = 3.0,
) -> pd.DataFrame:
    """Detect manufacturing drift events where sensor readings exceed 3-sigma control limits.

    Computes moving center line (rolling mean) and 3-sigma control limits (UCL / LCL)
    chronologically. Readings breaching [LCL, UCL] are flagged as drift events.

    Args:
        df: Clean lots DataFrame with timestamp, lot_id, and sensor columns.
        sensors: Specific list of sensors to monitor. If None, uses top correlated sensors.
        top_n_sensors: Number of top correlated sensors to monitor if sensors is None.
        window_size: Moving average rolling window size.
        min_periods: Minimum observations required to establish control limits.
        sigma_multiplier: Number of standard deviations for limits (standard Shewhart 3-sigma).

    Returns:
        pd.DataFrame: Drift events with lot_id, timestamp, sensor, reading,
                      rolling_mean, rolling_std, lcl, ucl, deviation_sigmas, direction.
    """
    if "timestamp" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'timestamp' column for SPC temporal ordering."
        )

    # Chronological sort
    df_sorted = df.sort_values(by="timestamp").reset_index(drop=True)

    # Determine sensors to monitor
    if sensors is None:
        corr_df = compute_sensor_correlations(df_sorted)
        monitored_sensors = corr_df["sensor"].head(top_n_sensors).tolist()
    else:
        monitored_sensors = sensors

    events = []

    for sensor in monitored_sensors:
        series = df_sorted[sensor]
        rolling_mean = series.rolling(
            window=window_size, min_periods=min_periods
        ).mean()
        rolling_std = series.rolling(window=window_size, min_periods=min_periods).std()

        # If rolling std is zero or NaN, fallback to global std
        global_mean = series.mean()
        global_std = max(series.std(), 1e-6)

        rolling_mean = rolling_mean.fillna(global_mean)
        rolling_std = rolling_std.fillna(global_std).replace(0.0, global_std)

        ucl = rolling_mean + (sigma_multiplier * rolling_std)
        lcl = rolling_mean - (sigma_multiplier * rolling_std)

        # Vectorized check for violations
        high_breach = series > ucl
        low_breach = series < lcl

        breach_indices = df_sorted.index[high_breach | low_breach]

        for idx in breach_indices:
            val = float(series.iloc[idx])
            mean_val = float(rolling_mean.iloc[idx])
            std_val = float(rolling_std.iloc[idx])
            ucl_val = float(ucl.iloc[idx])
            lcl_val = float(lcl.iloc[idx])

            dev_sigmas = (val - mean_val) / std_val
            direction = "HIGH" if val > ucl_val else "LOW"

            events.append(
                {
                    "lot_id": (
                        df_sorted["lot_id"].iloc[idx]
                        if "lot_id" in df_sorted
                        else f"ROW_{idx}"
                    ),
                    "timestamp": df_sorted["timestamp"].iloc[idx].isoformat(),
                    "sensor": sensor,
                    "reading_value": round(val, 5),
                    "rolling_mean": round(mean_val, 5),
                    "rolling_std": round(std_val, 5),
                    "lcl_3sigma": round(lcl_val, 5),
                    "ucl_3sigma": round(ucl_val, 5),
                    "deviation_sigmas": round(float(dev_sigmas), 3),
                    "drift_direction": direction,
                    "actual_label": (
                        int(df_sorted["label"].iloc[idx])
                        if "label" in df_sorted
                        else None
                    ),
                }
            )

    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.sort_values(by="timestamp").reset_index(drop=True)

    return events_df


def run_spc_analysis(
    db_path: Path = DB_PATH,
    output_path: Path = SPC_DRIFT_EVENTS_PATH,
    top_n_sensors: int = 10,
) -> pd.DataFrame:
    """Execute SPC analysis on top correlated sensors and save drift events CSV.

    Args:
        db_path: Path to DuckDB database.
        output_path: Destination CSV path.
        top_n_sensors: Number of sensors to monitor.

    Returns:
        pd.DataFrame: Drift events log.
    """
    df = load_from_duckdb("clean_lots", db_path=db_path)
    events_df = detect_spc_drift_events(df, top_n_sensors=top_n_sensors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(output_path, index=False)
    return events_df


if __name__ == "__main__":
    drift_df = run_spc_analysis()
    print(f"SPC analysis detected {len(drift_df)} drift events across top 10 sensors.")
    if not drift_df.empty:
        print(
            drift_df.head(5)[
                [
                    "lot_id",
                    "timestamp",
                    "sensor",
                    "reading_value",
                    "deviation_sigmas",
                    "drift_direction",
                ]
            ]
        )
