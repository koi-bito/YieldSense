"""Unit tests for Phase 3 ML models, SHAP, and recovery scoring."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.recovery_scorer import calculate_recovery_score
from src.models.train import evaluate_classifier, prepare_train_test_data


@pytest.fixture
def sample_dataset():
    """Create synthetic dataset with class imbalance for ML unit tests."""
    np.random.seed(42)
    n = 100
    n_sensors = 10
    X = np.random.randn(n, n_sensors)
    # 90 passes (0), 10 fails (1)
    y = np.array([0] * 90 + [1] * 10)
    df = pd.DataFrame(X, columns=[f"sensor_{i:03d}" for i in range(1, n_sensors + 1)])
    df["label"] = y
    df.insert(0, "lot_id", [f"LOT_{i+1:04d}" for i in range(n)])
    return df


def test_prepare_train_test_data(sample_dataset):
    """Verify leak-free split and SMOTE balancing on training set only."""
    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        meta_train,
        meta_test,
    ) = prepare_train_test_data(sample_dataset, test_size=0.20, random_state=42)

    # Test set must retain original imbalance
    assert len(X_test) == 20
    assert (y_test == 1).sum() == 2
    assert (y_test == 0).sum() == 18

    # Training set must be balanced by SMOTE
    assert (y_train == 1).sum() == (y_train == 0).sum()

    # Scaler must have fitted on 10 sensors
    assert scaler.n_features_in_ == 10


def test_evaluate_classifier():
    """Verify classification metrics evaluation function."""
    clf = LogisticRegression()
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    clf.fit(X, y)

    metrics = evaluate_classifier(clf, X, y)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_calculate_recovery_score():
    """Verify recovery score behaves monotonically with failure probability."""
    # Lot barely failing (P=0.55) should have higher recovery score than catastrophic fail (P=0.98)
    marginal_score = calculate_recovery_score(
        failure_probability=0.55, adverse_sensor_count=1
    )
    severe_score = calculate_recovery_score(
        failure_probability=0.98, adverse_sensor_count=5
    )

    assert marginal_score > severe_score
    assert 0.0 <= marginal_score <= 1.0
    assert 0.0 <= severe_score <= 1.0


def test_explain_direction_nominal():
    """Verify explain_single_lot categorizes standardized readings into HIGH, LOW, and NOMINAL."""
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    from src.models.explain import explain_single_lot

    X = np.array([[-1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])
    y = np.array([0, 1, 0])
    scaler = StandardScaler().fit(X)
    model = XGBClassifier(n_estimators=2, random_state=42).fit(
        scaler.transform(X), y
    )

    sensor_names = ["sensor_001", "sensor_002"]
    # Sensor 1 at mean (0.0 -> scaled = 0.0 -> NOMINAL)
    # Sensor 2 above mean (2.0 -> scaled > 0.0 -> HIGH)
    explanation = explain_single_lot(
        model=model,
        scaler=scaler,
        sensor_names=sensor_names,
        sensor_values={"sensor_001": 0.0, "sensor_002": 2.0},
        top_n_features=2,
    )
    directions = {
        s["sensor"]: s["direction"]
        for s in explanation["top_contributing_sensors"]
    }
    assert directions["sensor_001"] == "NOMINAL"
    assert directions["sensor_002"] == "HIGH"


def test_recovery_ranking_empty():
    """Verify build_recovery_ranking safely returns empty table without ZeroDivisionError."""
    from src.models.recovery_scorer import build_recovery_ranking

    empty_df = pd.DataFrame(columns=["lot_id", "label"])
    ranking = build_recovery_ranking(
        df=empty_df,
        model=None,
        scaler=None,
        shap_explanations={},
    )
    assert len(ranking) == 0
    assert "recovery_candidate" in ranking.columns
