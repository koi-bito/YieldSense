"""Pydantic schemas for YieldSense REST API requests and responses."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class SensorContribution(BaseModel):
    """SHAP attribution and direction for a single sensor feature."""

    sensor: str = Field(..., description="Sensor column identifier (e.g., sensor_060)")
    shap_value: float = Field(..., description="SHAP attribution value")
    scaled_reading: float = Field(..., description="Standardized reading (z-score)")
    impact: str = Field(..., description="INCREASES_FAIL_RISK or DECREASES_FAIL_RISK")
    direction: str = Field(..., description="HIGH, LOW, or NOMINAL relative to distribution mean")


class LotAnalysisRequest(BaseModel):
    """Payload for analyzing a single semiconductor production lot."""

    lot_id: str = Field(
        "LOT_INCOMING", description="Unique identifier for the wafer/lot"
    )
    sensors: Dict[str, float] = Field(
        ...,
        description="Key-value mapping of sensor names to measurements",
    )


class LotAnalysisResponse(BaseModel):
    """Analytical verdict and recovery recommendation for a lot."""

    lot_id: str
    prediction: str = Field(..., description="PASS or FAIL")
    failure_probability: float = Field(
        ..., description="Model predicted failure probability [0.0, 1.0]"
    )
    recovery_score: float = Field(
        ..., description="Recovery potential score [0.0, 1.0]"
    )
    recovery_candidate: bool = Field(
        ..., description="True if recommended for salvage or re-binning"
    )
    top_sensors: List[SensorContribution] = Field(
        default_factory=list,
        description="Top driving sensors ranked by |SHAP|",
    )


class BatchAnalysisRequest(BaseModel):
    """Payload for analyzing multiple production lots in batch."""

    lots: List[LotAnalysisRequest]


class BatchSummary(BaseModel):
    """Aggregate statistics for batch lot analysis."""

    total_lots: int
    predicted_pass: int
    predicted_fail: int
    recovery_candidates_count: int
    recovery_rate_percent: float


class BatchAnalysisResponse(BaseModel):
    """Batch analytical output with ranked recovery candidates."""

    summary: BatchSummary
    results: List[LotAnalysisResponse]


class HealthResponse(BaseModel):
    """Service health and model readiness check."""

    status: str
    model_loaded: bool
    scaler_loaded: bool
    expected_sensors_count: int
    version: str


class ModelInfoResponse(BaseModel):
    """Metadata regarding currently active machine learning model."""

    model_type: str
    evaluation_metrics: Dict[str, Any]
    best_hyperparameters: Dict[str, Any]
