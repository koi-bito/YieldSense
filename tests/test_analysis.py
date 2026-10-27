"""Unit tests for Phase 2 statistical analysis modules."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.correlation import compute_sensor_correlations
from src.analysis.pca import perform_pca_analysis
from src.analysis.recovery import identify_recovery_candidates
from src.analysis.spc import detect_spc_drift_events


@pytest.fixture
def sample_clean_data():
    """Generate synthetic semiconductor sensor dataset."""
    np.random.seed(42)
    n_samples = 100
    n_sensors = 20

    sensor_data = np.random.randn(n_samples, n_sensors)
    # Inject signal into sensor_001
    labels = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    sensor_data[:, 0] += labels * 2.0  # sensor_001 positively correlated with fail

    df = pd.DataFrame(
        sensor_data,
        columns=[f"sensor_{i:03d}" for i in range(1, n_sensors + 1)],
    )
    df.insert(0, "lot_id", [f"LOT_{i+1:04d}" for i in range(n_samples)])
    df["label"] = labels
    df["timestamp"] = pd.date_range("2008-01-01", periods=n_samples, freq="1h")
    return df


def test_compute_sensor_correlations(sample_clean_data):
    """Verify correlation values are bounded in [-1, 1] and rank is ordered."""
    corr_df = compute_sensor_correlations(sample_clean_data)

    assert len(corr_df) == 20
    assert "sensor_001" == corr_df["sensor"].iloc[0]  # Injected signal ranked highest
    assert (corr_df["abs_correlation"] >= 0.0).all()
    assert (corr_df["abs_correlation"] <= 1.0).all()
    assert list(corr_df["rank"]) == list(range(1, 21))


def test_perform_pca_analysis(sample_clean_data):
    """Verify PCA reduces dimensionality and captures target variance."""
    pca, scaler, summary = perform_pca_analysis(
        sample_clean_data, variance_threshold=0.90
    )

    assert summary["total_sensors"] == 20
    assert summary["n_components_retained"] < 20
    assert summary["total_explained_variance"] >= 0.90
    assert "PC_01" in summary["top_component_loadings"]
    assert len(summary["top_component_loadings"]["PC_01"]) > 0


def test_detect_spc_drift_events(sample_clean_data):
    """Verify 3-sigma drift event detection."""
    # Inject deliberate 5-sigma outlier in row 50 for sensor_002
    sample_clean_data.loc[50, "sensor_002"] = 15.0

    events_df = detect_spc_drift_events(
        sample_clean_data,
        sensors=["sensor_002"],
        window_size=20,
        min_periods=5,
    )

    assert not events_df.empty
    breached_lots = events_df["lot_id"].tolist()
    assert sample_clean_data.loc[50, "lot_id"] in breached_lots
    outlier_event = events_df[
        events_df["lot_id"] == sample_clean_data.loc[50, "lot_id"]
    ].iloc[0]
    assert outlier_event["drift_direction"] == "HIGH"
    assert outlier_event["deviation_sigmas"] > 3.0


def test_identify_recovery_candidates(sample_clean_data):
    """Verify marginal candidate selection."""
    candidates_df = identify_recovery_candidates(
        sample_clean_data,
        max_oob_sensors=3,
        margin_std_factor=0.5,
    )

    failing_count = (sample_clean_data["label"] == 1).sum()
    assert len(candidates_df) == failing_count
    assert "is_recovery_candidate" in candidates_df.columns
    assert "recovery_status" in candidates_df.columns
