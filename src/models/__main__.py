"""CLI runner for machine learning training, SHAP attribution, and recovery scoring."""

import sys

from src.models.explain import run_shap_analysis
from src.models.recovery_scorer import run_recovery_scoring
from src.models.train import run_training_pipeline


def main() -> int:
    try:
        print("=== Step 1: Model Training & Hyperparameter Tuning ===")
        eval_res = run_training_pipeline(n_trials=20)
        print("XGBoost Test Performance:")
        for metric, val in eval_res["models"]["xgboost_tuned"].items():
            print(f"  {metric}: {val}")

        print("\n=== Step 2: SHAP Feature Attribution ===")
        shaps = run_shap_analysis()
        print(f"Computed SHAP values and root-cause attributions for {len(shaps)} lots.")

        print("\n=== Step 3: ML Recovery Likelihood Scoring ===")
        ranking = run_recovery_scoring()
        cands = int(ranking["recovery_candidate"].sum()) if len(ranking) > 0 else 0
        total_fail = len(ranking)
        pct_str = f"({cands / total_fail:.1%})" if total_fail > 0 else "(0.0%)"
        print(f"Total evaluated failing lots: {total_fail}")
        print(f"Recommended recovery candidates: {cands} {pct_str}")

        print("\nPhase 3 ML Pipeline completed successfully.")
        return 0
    except Exception as exc:
        print(f"\n❌ Machine learning pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
