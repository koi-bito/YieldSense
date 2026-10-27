"""Real-time inference pipeline executing clean -> scale -> predict -> SHAP -> recovery."""

from typing import Dict, List

from src.api.dependencies import ModelManager
from src.api.schemas import (
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    BatchSummary,
    LotAnalysisRequest,
    LotAnalysisResponse,
    SensorContribution,
)
from src.models.explain import explain_single_lot
from src.models.recovery_scorer import calculate_recovery_score


def process_single_lot(
    request: LotAnalysisRequest,
    manager: ModelManager,
    top_n_features: int = 5,
) -> LotAnalysisResponse:
    """Execute complete real-time inference flow for a single production lot."""
    if not manager.is_ready:
        manager.load_artifacts()
    if not manager.is_ready:
        raise RuntimeError("Model artifacts have not been trained or loaded yet.")

    sensor_names = manager.expected_sensors
    input_sensors = request.sensors

    # Check that sufficient features are provided
    provided_sensors = set(input_sensors.keys())
    missing_sensors = set(sensor_names) - provided_sensors

    # If missing sensors exceed 20%, raise validation error
    if len(missing_sensors) > (0.20 * len(sensor_names)):
        raise ValueError(
            f"Too many missing sensors ({len(missing_sensors)} of {len(sensor_names)} required)."
        )

    # Impute missing values with training median (or baseline mean)
    vector_values: Dict[str, float] = {}
    for s in sensor_names:
        if s in input_sensors and input_sensors[s] is not None:
            vector_values[s] = float(input_sensors[s])
        else:
            vector_values[s] = manager.get_sensor_imputation_value(s)

    # Execute SHAP explanation and prediction
    explanation = explain_single_lot(
        model=manager.model,
        scaler=manager.scaler,
        sensor_names=sensor_names,
        sensor_values=vector_values,
        top_n_features=top_n_features,
    )

    pred = explanation["prediction"]
    prob = explanation["failure_probability"]

    # Extract adverse sensor count
    adverse_sensors = [
        s
        for s in explanation["top_contributing_sensors"]
        if s["impact"] == "INCREASES_FAIL_RISK"
    ]

    rec_score = calculate_recovery_score(
        failure_probability=prob,
        adverse_sensor_count=len(adverse_sensors),
    )

    is_recovery_cand = (pred == "FAIL") and (rec_score >= 0.40)

    top_sensors = [
        SensorContribution(
            sensor=s["sensor"],
            shap_value=s["shap_value"],
            scaled_reading=s["scaled_reading"],
            impact=s["impact"],
            direction=s["direction"],
        )
        for s in explanation["top_contributing_sensors"]
    ]

    return LotAnalysisResponse(
        lot_id=request.lot_id,
        prediction=pred,
        failure_probability=prob,
        recovery_score=rec_score,
        recovery_candidate=is_recovery_cand,
        top_sensors=top_sensors,
    )


def process_batch_lots(
    request: BatchAnalysisRequest,
    manager: ModelManager,
) -> BatchAnalysisResponse:
    """Analyze a batch of lots and compile aggregate statistics and ranked candidate list."""
    results: List[LotAnalysisResponse] = []
    for lot_req in request.lots:
        results.append(process_single_lot(lot_req, manager))

    total = len(results)
    pred_pass = sum(1 for r in results if r.prediction == "PASS")
    pred_fail = sum(1 for r in results if r.prediction == "FAIL")
    cands = sum(1 for r in results if r.recovery_candidate)
    recovery_rate = (cands / pred_fail * 100.0) if pred_fail > 0 else 0.0

    # Sort results: recovery candidates first, descending by recovery_score
    sorted_results = sorted(
        results,
        key=lambda r: (r.recovery_candidate, r.recovery_score),
        reverse=True,
    )

    summary = BatchSummary(
        total_lots=total,
        predicted_pass=pred_pass,
        predicted_fail=pred_fail,
        recovery_candidates_count=cands,
        recovery_rate_percent=round(recovery_rate, 2),
    )

    return BatchAnalysisResponse(summary=summary, results=sorted_results)
