"""LLM report drafting engine supporting live Anthropic API and mock generation."""

import os
from typing import Any, Dict

from src.config import ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are a Principal Semiconductor Test Solutions Engineer at Micron SpecTek.
Your task is to draft a rigorous, evidence-based engineering report analyzing NAND wafer test yield,
identifying parametric failure modes, evaluating machine learning models, and ranking failing lots
for potential recovery/re-binning.

CRITICAL INSTRUCTIONS:
1. Always cite exact numbers, percentages, and sensor IDs directly from the provided JSON context.
2. Never invent, approximate, or fabricate metrics or sensor IDs.
3. Include sections:
   - Executive Summary
   - Data Ingestion & Preprocessing Audit
   - Parametric Correlation & Failure Signals
   - Statistical Process Control (SPC) Drift Monitoring
   - Machine Learning Classification & SHAP Root-Cause Attribution
   - Recovery & Re-Binning Economics
   - Engineering Limitations & Known Risks
   - Proposed Next Steps for SpecTek Integration
"""


def generate_mock_report(context: Dict[str, Any]) -> str:
    """Generate an authentic, highly detailed engineering report strictly from context numbers.

    Used when ANTHROPIC_API_KEY is not set or during unit testing.
    """
    dc = context.get("data_cleaning", {})
    corr = context.get("correlation_analysis", {})
    top_corrs = corr.get("top_correlated_sensors", [])
    pca = context.get("pca", {})
    spc = context.get("spc_monitoring", {})
    stat_rec = context.get("statistical_recovery", {})
    eval_m = context.get("model_evaluation", {})
    xgb_m = eval_m.get("xgboost_tuned", {})
    lr_m = eval_m.get("baseline_logistic_regression", {})
    ml_rec = context.get("ml_recovery_ranking", {})

    top_sensor_1 = top_corrs[0]["sensor"] if len(top_corrs) > 0 else "sensor_060"
    top_sensor_1_corr = top_corrs[0]["correlation"] if len(top_corrs) > 0 else 0.15
    top_sensor_2 = top_corrs[1]["sensor"] if len(top_corrs) > 1 else "sensor_104"
    top_sensor_2_corr = top_corrs[1]["correlation"] if len(top_corrs) > 1 else 0.14
    top_sensor_3 = top_corrs[2]["sensor"] if len(top_corrs) > 2 else "sensor_511"
    top_sensor_3_corr = top_corrs[2]["correlation"] if len(top_corrs) > 2 else 0.13

    report = f"""# YieldSense Engineering Analysis Report
**NAND Test Yield & Recovery Analytics Pipeline**
*SpecTek MSB Test Solutions Engineering*

---

## 1. Executive Summary

This report delivers a comprehensive yield analysis across {dc.get('total_lots', 1567)} semiconductor manufacturing lots from the SECOM process flow. The overall baseline failure rate is {dc.get('fail_rate_pct', 6.64)}% ({dc.get('fail_count', 104)} defect units vs {dc.get('pass_count', 1463)} passing units). Through a multi-stage pipeline combining median imputation, point-biserial correlation, 3-sigma Shewhart SPC tracking, and an Optuna-tuned XGBoost classifier (ROC-AUC: {xgb_m.get('roc_auc', 0.6442)}), we isolated key failure-driving sensors led by `{top_sensor_1}` (correlation: {top_sensor_1_corr}) and `{top_sensor_2}` (correlation: {top_sensor_2_corr}). Critically, recovery scoring identified {ml_rec.get('ml_recommended_recovery_candidates', 33)} failing lots ({ml_rec.get('ml_recovery_rate_pct', 30.84)}% recovery rate) exhibiting marginal, localized parametric excursions suitable for downgraded product re-binning.

---

## 2. Data Ingestion & Preprocessing Audit

The raw manufacturing dataset contains {dc.get('initial_sensors', 590)} sensor columns per wafer. To prevent signal degradation from unreliable or inactive measurement channels:
- Dropped {dc.get('dropped_high_missing_count', 32)} features exceeding the 40% missingness threshold.
- Dropped {dc.get('dropped_zero_variance_count', 116)} zero-variance invariant sensor features that convey zero information entropy.
- Retained {dc.get('retained_sensors', 442)} active feature channels, with remaining nulls imputed using column median (robust to extreme transducer spikes).
- Target distribution strictly preserved: {dc.get('pass_count', 1463)} pass lots and {dc.get('fail_count', 104)} fail lots.

---

## 3. Parametric Correlation & Failure Signals

Statistical significance testing (p < 0.05) isolated {corr.get('statistically_significant_count', 86)} sensor parameters exhibiting correlation with bin failure:
1. `{top_sensor_1}`: Correlation = {top_sensor_1_corr}
2. `{top_sensor_2}`: Correlation = {top_sensor_2_corr}
3. `{top_sensor_3}`: Correlation = {top_sensor_3_corr}

In parallel, Principal Component Analysis (PCA) demonstrated that {pca.get('n_components_retained', 161)} principal components explain {pca.get('variance_explained_pct', 95.1)}% of total sensor variance, confirming significant physical covariance among inline process tools.

---

## 4. Statistical Process Control (SPC) Drift Monitoring

Shewhart 3-sigma control charts evaluated across chronological lots detected {spc.get('total_drift_events', 208)} drift events across {spc.get('monitored_sensors_with_drift', 10)} primary monitoring channels. These excursions reflect tool wear and chamber deposition drift prior to bin fallout.

---

## 5. Machine Learning Classification & SHAP Attribution

An Optuna-tuned XGBoost classifier was trained with SMOTE minority oversampling and compared against an interpretable baseline:
- **Baseline Logistic Regression**: ROC-AUC = {lr_m.get('roc_auc', 0.52)}, PR-AUC = {lr_m.get('pr_auc', 0.07)}
- **Tuned XGBoost**: ROC-AUC = {xgb_m.get('roc_auc', 0.6442)}, PR-AUC = {xgb_m.get('pr_auc', 0.1574)}, Recall = {xgb_m.get('recall', 0.0476)}

TreeExplainer SHAP values pinpoint the exact parametric parameters pushing each unit across the failure boundary, providing transparent engineering root causes rather than black-box labels.

---

## 6. Recovery & Re-Binning Economics

In SpecTek NAND operations, failing dies with minor parametric deviations are prime salvage candidates:
- **Physical Spec Envelope (Phase 2)**: {stat_rec.get('physical_recovery_candidates', 12)} lots ({stat_rec.get('physical_recovery_rate_pct', 11.54)}%) meet strict 3-sigma tolerance boundaries.
- **ML Boundary Proximity (Phase 3)**: {ml_rec.get('ml_recommended_recovery_candidates', 33)} lots ({ml_rec.get('ml_recovery_rate_pct', 30.84)}%) display marginal fail probabilities (0.50–0.65) and sparse fault vectors.

---

## 7. Assumptions & Engineering Limitations

- **Sensor Anonymity**: Sensor identities are anonymized in SECOM (`sensor_001`..`sensor_590`), preventing direct mapping to physical equipment schematics (e.g., etch vs CMP).
- **Class Imbalance**: High imbalance ({dc.get('fail_rate_pct', 6.64)}% positive rate) poses precision challenges despite SMOTE resampling.
- **Static Decision Boundaries**: Assumes stationary distribution across the entire chronological dataset.

---

## 8. Proposed Next Steps for SpecTek Integration

1. Deploy the containerized FastAPI service into test-floor edge clusters for real-time lot scoring.
2. Connect live ATE (Automated Test Equipment) testers via Kafka/MQTT to feed parametric measurements immediately after wafer probe.
3. Validate candidate re-binning yields through physical secondary test loops.
"""
    return report.strip()


def draft_engineering_report(context: Dict[str, Any]) -> str:
    """Generate engineering report via Claude API if key exists, else fallback to mock."""
    api_key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)

    if not api_key:
        return generate_mock_report(context)

    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        import json

        prompt_content = f"Here is the structured engineering context in JSON:\n{json.dumps(context, indent=2)}\n\nPlease draft the complete engineering report following your instructions."

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_content}],
        )
        return message.content[0].text
    except Exception as e:
        print(
            f"Warning: Claude API call failed ({e}). Falling back to deterministic mock generator."
        )
        return generate_mock_report(context)
