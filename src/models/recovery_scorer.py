"""Machine learning recovery scoring and lot ranking."""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.config import (
    DB_PATH,
    RECOVERY_CANDIDATES_PATH,
    RECOVERY_RANKING_PATH,
    SCALER_PATH,
    SHAP_EXPLANATIONS_PATH,
    XGB_MODEL_PATH,
)
from src.ingestion.features import get_sensor_columns, load_scaler
from src.ingestion.loader import load_from_duckdb


def calculate_recovery_score(
    failure_probability: float,
    adverse_sensor_count: int,
    is_rule_candidate: bool = False,
    max_adverse_sensors: int = 5,
) -> float:
    """Calculate composite recovery score [0.0, 1.0] for a failing lot.

    Higher score indicates greater likelihood of salvaging the lot via re-binning.
    Formula balances:
    - Boundary proximity (60%): P(Fail) close to 0.50 reflects marginal defect.
    - Fault sparsity (30%): Fewer adverse sensors means localized, repairable defects.
    - Physical rule match bonus (10%): Match with SPC 3-sigma recovery boundary.

    Args:
        failure_probability: Model predicted fail probability [0.0, 1.0].
        adverse_sensor_count: Number of sensors actively pushing towards failure.
        is_rule_candidate: Boolean indicating if lot passed Phase 2 physical recovery check.
        max_adverse_sensors: Normalizing denominator for adverse sensors.

    Returns:
        float: Recovery score in [0.0, 1.0].
    """
    if failure_probability < 0.5:
        # Lot is predicted pass; recovery score represents safety margin
        return 1.0

    # Proximity to decision boundary: 0.50 -> 1.0, 1.00 -> 0.0
    proximity_score = max(0.0, 1.0 - (failure_probability - 0.5) / 0.5)

    # Sparsity score: fewer offending sensors -> higher score
    sparsity_score = max(
        0.0, 1.0 - min(adverse_sensor_count, max_adverse_sensors) / max_adverse_sensors
    )

    # Physical rule bonus
    bonus = 0.10 if is_rule_candidate else 0.0

    score = (0.60 * proximity_score) + (0.30 * sparsity_score) + bonus
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def build_recovery_ranking(
    df: pd.DataFrame,
    model: Any,
    scaler: Any,
    shap_explanations: Optional[Dict[str, Any]] = None,
    rule_candidates_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generate ranked recovery candidate table for all predicted or actual failing lots.

    Args:
        df: Clean lots DataFrame with sensors, labels, and lot_id.
        model: Trained classifier.
        scaler: Fitted scaler.
        shap_explanations: Optional precomputed SHAP explanations dict.
        rule_candidates_df: Optional Phase 2 recovery candidates DataFrame.

    Returns:
        pd.DataFrame: Sorted recovery candidate rankings.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "lot_id",
                "actual_label",
                "predicted_label",
                "failure_probability",
                "recovery_score",
                "recovery_candidate",
                "top_adverse_sensors",
            ]
        )

    sensor_cols = get_sensor_columns(df)
    X_raw = df[sensor_cols].values
    X_scaled = scaler.transform(X_raw)

    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Set of rule-based candidates from Phase 2
    rule_candidate_set = set()
    if rule_candidates_df is not None and not rule_candidates_df.empty:
        rule_candidate_set = set(
            rule_candidates_df[rule_candidates_df["is_recovery_candidate"]]["lot_id"]
        )

    records = []
    for idx, row in df.iterrows():
        lot_id = row.get("lot_id", f"LOT_{idx}")
        prob = float(probs[idx])
        pred = int(preds[idx])
        actual = int(row.get("label", -1))

        # We evaluate recovery for all failing units (actual or predicted)
        if pred == 1 or actual == 1:
            # Extract adverse sensors from SHAP if available
            adverse_sensors = []
            if shap_explanations and lot_id in shap_explanations:
                lot_shap = shap_explanations[lot_id].get("top_contributing_sensors", [])
                adverse_sensors = [
                    s["sensor"]
                    for s in lot_shap
                    if s.get("impact") == "INCREASES_FAIL_RISK"
                ]

            is_rule_cand = lot_id in rule_candidate_set
            rec_score = calculate_recovery_score(
                failure_probability=prob,
                adverse_sensor_count=len(adverse_sensors),
                is_rule_candidate=is_rule_cand,
            )

            is_candidate = rec_score >= 0.40 or is_rule_cand

            records.append(
                {
                    "lot_id": lot_id,
                    "actual_label": "FAIL" if actual == 1 else "PASS",
                    "predicted_label": "FAIL" if pred == 1 else "PASS",
                    "failure_probability": round(prob, 4),
                    "recovery_score": rec_score,
                    "recovery_candidate": is_candidate,
                    "top_adverse_sensors": ";".join(adverse_sensors[:5]),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "rank",
                "lot_id",
                "actual_label",
                "predicted_label",
                "failure_probability",
                "recovery_score",
                "recovery_candidate",
                "top_adverse_sensors",
            ]
        )

    ranking_df = pd.DataFrame(records)
    ranking_df = ranking_df.sort_values(
        by=["recovery_candidate", "recovery_score"], ascending=[False, False]
    ).reset_index(drop=True)
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))

    return ranking_df


def run_recovery_scoring(
    db_path: Path = DB_PATH,
    output_path: Path = RECOVERY_RANKING_PATH,
) -> pd.DataFrame:
    """Execute recovery ranking pipeline using model, scaler, and SHAP outputs."""
    df = load_from_duckdb("clean_lots", db_path=db_path)
    model = joblib.load(XGB_MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)

    # Load SHAP explanations if available
    shap_data = None
    if SHAP_EXPLANATIONS_PATH.exists():
        import json

        with open(SHAP_EXPLANATIONS_PATH, "r", encoding="utf-8") as f:
            shap_data = json.load(f)

    # Load rule candidates if available
    rule_df = None
    if RECOVERY_CANDIDATES_PATH.exists():
        rule_df = pd.read_csv(RECOVERY_CANDIDATES_PATH)

    ranking_df = build_recovery_ranking(
        df=df,
        model=model,
        scaler=scaler,
        shap_explanations=shap_data,
        rule_candidates_df=rule_df,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(output_path, index=False)
    return ranking_df


if __name__ == "__main__":
    ranks = run_recovery_scoring()
    print(f"Generated recovery ranking for {len(ranks)} failing units.")
    if len(ranks) > 0:
        cands = int(ranks["recovery_candidate"].sum())
        print(f"High-confidence recovery candidates: {cands} ({cands / len(ranks):.1%})")
        print(
            ranks.head(5)[
                [
                    "rank",
                    "lot_id",
                    "failure_probability",
                    "recovery_score",
                    "recovery_candidate",
                ]
            ]
        )
    else:
        print("No failing lots identified in dataset; 0 recovery candidates.")
