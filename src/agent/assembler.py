"""Context assembler aggregating all pipeline outputs into a structured reporting payload."""

import json
from typing import Any, Dict, Optional

import pandas as pd

from src.config import (
    CLEANING_LOG_PATH,
    MODEL_EVAL_PATH,
    RECOVERY_CANDIDATES_PATH,
    RECOVERY_RANKING_PATH,
    REPORTS_DIR,
    SENSOR_CORRELATION_PATH,
    SHAP_EXPLANATIONS_PATH,
    SPC_DRIFT_EVENTS_PATH,
)


def assemble_engineering_context(
    lot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and condense all structured analytical outputs into an LLM-ready context object.

    Args:
        lot_id: Optional specific lot ID to focus the report on.

    Returns:
        Dict[str, Any]: Consolidated metrics, correlations, SPC drift events, and recovery stats.
    """
    context: Dict[str, Any] = {
        "report_metadata": {
            "dataset": "UCI SECOM Semiconductor Manufacturing Process",
            "domain": "NAND Flash / Semiconductor Wafer Parametric Test & Binning",
            "focused_lot_id": lot_id,
        }
    }

    # 1. Ingestion & Cleaning Audit
    if CLEANING_LOG_PATH.exists():
        with open(CLEANING_LOG_PATH, "r", encoding="utf-8") as f:
            cleaning_data = json.load(f)
            context["data_cleaning"] = {
                "total_lots": cleaning_data.get("total_lots"),
                "initial_sensors": cleaning_data.get("initial_sensor_count"),
                "dropped_high_missing_count": cleaning_data.get(
                    "dropped_high_missing_count"
                ),
                "dropped_zero_variance_count": cleaning_data.get(
                    "dropped_zero_variance_count"
                ),
                "retained_sensors": cleaning_data.get("final_sensor_count"),
                "pass_count": cleaning_data.get("label_distribution", {}).get(
                    "pass_count"
                ),
                "fail_count": cleaning_data.get("label_distribution", {}).get(
                    "fail_count"
                ),
                "fail_rate_pct": round(
                    (
                        cleaning_data.get("label_distribution", {}).get(
                            "fail_rate", 0.0
                        )
                        * 100
                    ),
                    2,
                ),
            }

    # 2. Top Correlated Sensors
    if SENSOR_CORRELATION_PATH.exists():
        corr_df = pd.read_csv(SENSOR_CORRELATION_PATH)
        top_corr = corr_df.head(10).to_dict(orient="records")
        sig_count = (
            int(corr_df["is_significant_05"].sum())
            if "is_significant_05" in corr_df
            else 0
        )
        context["correlation_analysis"] = {
            "statistically_significant_count": sig_count,
            "top_correlated_sensors": top_corr,
        }

    # 3. PCA Summary
    pca_file = REPORTS_DIR / "pca_summary.json"
    if pca_file.exists():
        with open(pca_file, "r", encoding="utf-8") as f:
            pca_data = json.load(f)
            context["pca"] = {
                "n_components_retained": pca_data.get("n_components_retained"),
                "variance_explained_pct": round(
                    pca_data.get("total_explained_variance", 0.0) * 100, 2
                ),
                "top_pc1_sensors": [
                    s["sensor"]
                    for s in pca_data.get("top_component_loadings", {}).get(
                        "PC_01", []
                    )[:3]
                ],
            }

    # 4. SPC Drift Events
    if SPC_DRIFT_EVENTS_PATH.exists():
        spc_df = pd.read_csv(SPC_DRIFT_EVENTS_PATH)
        context["spc_monitoring"] = {
            "total_drift_events": len(spc_df),
            "monitored_sensors_with_drift": (
                spc_df["sensor"].nunique() if not spc_df.empty else 0
            ),
            "top_drifting_sensors": (
                spc_df["sensor"].value_counts().head(5).to_dict()
                if not spc_df.empty
                else {}
            ),
        }

    # 5. Rule-Based Recovery Candidates
    if RECOVERY_CANDIDATES_PATH.exists():
        rec_cand_df = pd.read_csv(RECOVERY_CANDIDATES_PATH)
        recoverable = int(rec_cand_df["is_recovery_candidate"].sum())
        context["statistical_recovery"] = {
            "total_failing_lots": len(rec_cand_df),
            "physical_recovery_candidates": recoverable,
            "physical_recovery_rate_pct": (
                round(recoverable / len(rec_cand_df) * 100, 2)
                if len(rec_cand_df) > 0
                else 0.0
            ),
        }

    # 6. ML Model Evaluation
    if MODEL_EVAL_PATH.exists():
        with open(MODEL_EVAL_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
            context["model_evaluation"] = {
                "baseline_logistic_regression": eval_data.get("models", {}).get(
                    "logistic_regression_baseline", {}
                ),
                "xgboost_tuned": eval_data.get("models", {}).get("xgboost_tuned", {}),
                "best_hyperparameters": eval_data.get("best_hyperparameters", {}),
            }

    # 7. ML Recovery Ranking
    if RECOVERY_RANKING_PATH.exists():
        ranking_df = pd.read_csv(RECOVERY_RANKING_PATH)
        cands = int(ranking_df["recovery_candidate"].sum())
        context["ml_recovery_ranking"] = {
            "evaluated_failing_lots": len(ranking_df),
            "ml_recommended_recovery_candidates": cands,
            "ml_recovery_rate_pct": (
                round(cands / len(ranking_df) * 100, 2) if len(ranking_df) > 0 else 0.0
            ),
            "top_recovery_candidates": ranking_df.head(5).to_dict(orient="records"),
        }

    # 8. Single lot specific details if requested
    if lot_id:
        if SHAP_EXPLANATIONS_PATH.exists():
            with open(SHAP_EXPLANATIONS_PATH, "r", encoding="utf-8") as f:
                shap_all = json.load(f)
                context["focused_lot_details"] = shap_all.get(lot_id, {})

    return context
