"""ML model training pipeline: Stratified split, SMOTE, Optuna tuning, and evaluation."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    DB_PATH,
    MODEL_EVAL_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_SIZE,
    XGB_MODEL_PATH,
    XGB_PARAMS_PATH,
)
from src.ingestion.features import save_scaler, split_features_and_targets
from src.ingestion.loader import load_from_duckdb

# Suppress Optuna logging verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Perform leak-free stratified split, fit scaler on train only, and apply SMOTE.

    Args:
        df: Cleaned lots DataFrame.
        test_size: Proportion of dataset to include in test split (default 0.20).
        random_state: Random state for reproducibility.

    Returns:
        Tuple:
            X_train_resampled, y_train_resampled (post-SMOTE, scaled),
            X_test_scaled, y_test,
            fitted_scaler,
            meta_train, meta_test
    """
    X, y, meta = split_features_and_targets(df)

    # Stratified split preserving 93.4 / 6.6 class ratio
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    X_train_raw = X.iloc[train_idx].values
    y_train = y.iloc[train_idx].values
    meta_train = meta.iloc[train_idx].reset_index(drop=True)

    X_test_raw = X.iloc[test_idx].values
    y_test = y.iloc[test_idx].values
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    # Fit scaler strictly on training split to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Apply SMOTE strictly on training split
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return (
        X_train_resampled,
        y_train_resampled,
        X_test_scaled,
        y_test,
        scaler,
        meta_train,
        meta_test,
    )


def evaluate_classifier(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute standard classification metrics on test set."""
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
    }


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 25,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Tune XGBoost hyperparameters with Optuna using 3-fold stratified cross-validation."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
        }

        scores = []
        for train_fold_idx, val_fold_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_fold_idx], X_train[val_fold_idx]
            y_tr, y_val = y_train[train_fold_idx], y_train[val_fold_idx]

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, probs))

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["random_state"] = random_state
    best_params["eval_metric"] = "logloss"
    best_params["n_jobs"] = -1
    return best_params


def run_training_pipeline(
    db_path: Path = DB_PATH,
    n_trials: int = 25,
) -> Dict[str, Any]:
    """Execute end-to-end training, hyperparameter optimization, evaluation, and artifact saving."""
    df = load_from_duckdb("clean_lots", db_path=db_path)
    (
        X_train,
        y_train,
        X_test,
        y_test,
        scaler,
        meta_train,
        meta_test,
    ) = prepare_train_test_data(df)

    # 1. Save fitted StandardScaler for inference
    save_scaler(scaler, path=SCALER_PATH)

    # 2. Baseline Logistic Regression
    lr_baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_baseline.fit(X_train, y_train)
    lr_metrics = evaluate_classifier(lr_baseline, X_test, y_test)

    # 3. XGBoost Hyperparameter Optimization
    best_xgb_params = tune_xgboost(X_train, y_train, n_trials=n_trials)

    # 4. Train final XGBoost model
    xgb_model = XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    xgb_metrics = evaluate_classifier(xgb_model, X_test, y_test)

    # 5. Persist best params and final model
    XGB_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(XGB_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(best_xgb_params, f, indent=2)

    XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, XGB_MODEL_PATH)

    # 6. Save comparative evaluation report
    eval_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_fail_samples": int((y_test == 1).sum()),
        "test_pass_samples": int((y_test == 0).sum()),
        "models": {
            "logistic_regression_baseline": lr_metrics,
            "xgboost_tuned": xgb_metrics,
        },
        "best_hyperparameters": best_xgb_params,
    }

    MODEL_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    return eval_report


if __name__ == "__main__":
    print("Initiating YieldSense model training pipeline...")
    results = run_training_pipeline()
    print("\nModel Training Complete:")
    print(
        "Baseline Logistic Regression:",
        results["models"]["logistic_regression_baseline"],
    )
    print("Tuned XGBoost:", results["models"]["xgboost_tuned"])
