"""Statistical recovery candidate identification for failing semiconductor lots."""

from pathlib import Path

import pandas as pd

from src.config import (
    DB_PATH,
    RECOVERY_CANDIDATES_PATH,
    RECOVERY_MARGIN_STD_FACTOR,
    RECOVERY_MAX_OOB_SENSORS,
)
from src.ingestion.features import get_sensor_columns
from src.ingestion.loader import load_from_duckdb


def compute_pass_envelope(
    pass_df: pd.DataFrame,
    spec_sigmas: float = 3.0,
) -> pd.DataFrame:
    """Establish parametric specification boundaries from passing lots.

    Args:
        pass_df: DataFrame of confirmed passing lots (label == 0).
        spec_sigmas: Number of standard deviations defining the pass envelope.

    Returns:
        pd.DataFrame with index = sensor, columns = [mean, std, lsl, usl]
        (Lower Spec Limit, Upper Spec Limit).
    """
    sensor_cols = get_sensor_columns(pass_df)
    means = pass_df[sensor_cols].mean()
    stds = pass_df[sensor_cols].std().replace(0.0, 1e-6)

    envelope = pd.DataFrame(
        {
            "mean": means,
            "std": stds,
            "lsl": means - (spec_sigmas * stds),
            "usl": means + (spec_sigmas * stds),
        }
    )
    return envelope


def identify_recovery_candidates(
    df: pd.DataFrame,
    max_oob_sensors: int = RECOVERY_MAX_OOB_SENSORS,
    margin_std_factor: float = RECOVERY_MARGIN_STD_FACTOR,
    spec_sigmas: float = 3.0,
) -> pd.DataFrame:
    """Identify failing lots that are near-miss candidates for recovery/re-binning.

    A failing lot is designated a recovery candidate if:
    1. It has <= max_oob_sensors (default 3) exceeding the normal pass envelope (3-sigma).
    2. All out-of-boundary (OOB) sensor readings are within `margin_std_factor * std`
       (default 0.5 sigma) of the boundary, indicating marginal failure rather than catastrophic defect.

    Args:
        df: Clean lots DataFrame with sensors, label, and lot_id.
        max_oob_sensors: Maximum allowable out-of-spec sensors.
        margin_std_factor: Allowed overshoot beyond spec boundary in units of std dev.
        spec_sigmas: Spec envelope width in standard deviations.

    Returns:
        pd.DataFrame: Failing lots with out-of-spec counts, offending sensors, and recovery status.
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain 'label' column.")

    sensor_cols = get_sensor_columns(df)
    pass_df = df[df["label"] == 0]
    fail_df = df[df["label"] == 1].copy()

    if pass_df.empty or fail_df.empty:
        raise ValueError("Dataset must contain both passing and failing lots.")

    envelope = compute_pass_envelope(pass_df, spec_sigmas=spec_sigmas)

    lsl = envelope["lsl"]
    usl = envelope["usl"]
    stds = envelope["std"]

    candidate_records = []

    for idx, row in fail_df.iterrows():
        lot_id = row.get("lot_id", f"LOT_{idx}")
        readings = row[sensor_cols]

        # Check which sensors exceed standard spec
        low_oob = readings < lsl
        high_oob = readings > usl
        oob_mask = low_oob | high_oob

        oob_sensors = readings.index[oob_mask].tolist()
        oob_count = len(oob_sensors)

        # Check if all OOB sensors are within recovery margin
        exceeds_recovery_limit = False
        max_excess_sigma = 0.0

        for s in oob_sensors:
            val = float(readings[s])
            s_lsl = float(lsl[s])
            s_usl = float(usl[s])
            s_std = float(stds[s])

            if val < s_lsl:
                excess_sigma = (s_lsl - val) / s_std
            else:
                excess_sigma = (val - s_usl) / s_std

            if excess_sigma > max_excess_sigma:
                max_excess_sigma = excess_sigma

            if excess_sigma > margin_std_factor:
                exceeds_recovery_limit = True

        # Candidate criteria: <= max_oob_sensors AND no severe breach
        is_candidate = (0 < oob_count <= max_oob_sensors) and (
            not exceeds_recovery_limit
        )

        candidate_records.append(
            {
                "lot_id": lot_id,
                "oob_sensor_count": oob_count,
                "max_excess_boundary_sigma": round(float(max_excess_sigma), 3),
                "is_recovery_candidate": is_candidate,
                "recovery_status": "RECOVERABLE" if is_candidate else "REJECT",
                "oob_sensors": ";".join(
                    oob_sensors[:10]
                ),  # Cap string representation to first 10
            }
        )

    candidate_df = pd.DataFrame(candidate_records)
    candidate_df = candidate_df.sort_values(
        by=["is_recovery_candidate", "oob_sensor_count"], ascending=[False, True]
    ).reset_index(drop=True)

    return candidate_df


def run_recovery_analysis(
    db_path: Path = DB_PATH,
    output_path: Path = RECOVERY_CANDIDATES_PATH,
    max_oob_sensors: int = RECOVERY_MAX_OOB_SENSORS,
    margin_std_factor: float = RECOVERY_MARGIN_STD_FACTOR,
) -> pd.DataFrame:
    """Load clean lots from DuckDB, evaluate recovery candidates, and save CSV.

    Args:
        db_path: Path to DuckDB database.
        output_path: Destination CSV path.
        max_oob_sensors: Maximum OOB sensors allowed.
        margin_std_factor: Allowed boundary overshoot in std devs.

    Returns:
        pd.DataFrame: Candidate identification report.
    """
    df = load_from_duckdb("clean_lots", db_path=db_path)
    candidates_df = identify_recovery_candidates(
        df,
        max_oob_sensors=max_oob_sensors,
        margin_std_factor=margin_std_factor,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_df.to_csv(output_path, index=False)
    return candidates_df


if __name__ == "__main__":
    candidates = run_recovery_analysis()
    total_failing = len(candidates)
    recoverable = candidates["is_recovery_candidate"].sum()
    print(f"Total failing lots evaluated: {total_failing}")
    print(
        f"Recoverable candidate lots: {recoverable} ({recoverable / total_failing:.1%})"
    )
