"""Principal Component Analysis on semiconductor sensor matrix."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import DB_PATH, PCA_PATH, RANDOM_STATE, REPORTS_DIR
from src.ingestion.features import get_sensor_columns
from src.ingestion.loader import load_from_duckdb


def perform_pca_analysis(
    df: pd.DataFrame,
    variance_threshold: float = 0.95,
    top_loadings_per_pc: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[PCA, StandardScaler, Dict[str, Any]]:
    """Fit PCA on standardized sensor data and extract variance metrics and loadings.

    Args:
        df: DataFrame containing sensor columns.
        variance_threshold: Cumulative variance target (e.g. 0.95 for 95%).
        top_loadings_per_pc: Number of top contributing sensors per principal component.
        random_state: Random seed.

    Returns:
        Tuple[PCA, StandardScaler, Dict[str, Any]]:
            Fitted PCA model, fitted scaler, and analysis summary dict.
    """
    sensor_cols = get_sensor_columns(df)
    X = df[sensor_cols].values

    # Standardize features (mean=0, variance=1) prior to PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA with enough components to capture the target cumulative variance
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    pca.fit(X_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = pca.n_components_

    # Extract top sensor loadings for the first min(10, n_components) principal components
    loadings = pca.components_
    top_loadings = {}
    num_pcs_to_report = min(10, n_components)

    for pc_idx in range(num_pcs_to_report):
        pc_name = f"PC_{pc_idx + 1:02d}"
        pc_loadings = loadings[pc_idx]
        top_sensor_indices = np.argsort(np.abs(pc_loadings))[::-1][:top_loadings_per_pc]

        top_loadings[pc_name] = [
            {
                "sensor": sensor_cols[idx],
                "loading": round(float(pc_loadings[idx]), 5),
                "abs_loading": round(abs(float(pc_loadings[idx])), 5),
            }
            for idx in top_sensor_indices
        ]

    summary = {
        "total_sensors": len(sensor_cols),
        "target_variance_threshold": variance_threshold,
        "n_components_retained": int(n_components),
        "dimension_reduction_ratio": round(1.0 - (n_components / len(sensor_cols)), 4),
        "top_5_explained_variance_ratio": [
            round(float(v), 5) for v in pca.explained_variance_ratio_[:5]
        ],
        "top_5_cumulative_variance": [
            round(float(v), 5) for v in cumulative_variance[:5]
        ],
        "total_explained_variance": round(float(cumulative_variance[-1]), 5),
        "top_component_loadings": top_loadings,
    }

    return pca, scaler, summary


def run_pca_pipeline(
    db_path: Path = DB_PATH,
    model_path: Path = PCA_PATH,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute PCA pipeline, serialize model, and save summary report.

    Args:
        db_path: Path to DuckDB database.
        model_path: Path to save fitted PCA model.
        report_path: Path to save variance summary JSON (defaults to reports/pca_summary.json).

    Returns:
        Dict[str, Any]: Analysis summary metrics.
    """
    if report_path is None:
        report_path = REPORTS_DIR / "pca_summary.json"

    df = load_from_duckdb("clean_lots", db_path=db_path)
    pca, scaler, summary = perform_pca_analysis(df)

    # Save PCA model and accompanying scaler to disk
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pca": pca, "scaler": scaler}, model_path)

    # Save summary report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    results = run_pca_pipeline()
    print("PCA completed.")
    print(
        f"Retained {results['n_components_retained']} PCs explaining {results['total_explained_variance']:.1%} variance."
    )
    print(f"Dimensionality reduction: {results['dimension_reduction_ratio']:.1%}")
