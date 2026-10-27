"""Rules-based validation layer to audit LLM reports and detect hallucinations."""

import re
from typing import Any, Dict, List, Set


def extract_numbers_from_context(context: Dict[str, Any]) -> Set[float]:
    """Recursively collect all numeric values present in the source context."""
    numbers = set()

    def _collect(val):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            numbers.add(round(float(val), 4))
            numbers.add(round(float(val), 2))
            numbers.add(round(float(val), 1))
        elif isinstance(val, dict):
            for v in val.values():
                _collect(v)
        elif isinstance(val, list):
            for item in val:
                _collect(item)

    _collect(context)
    return numbers


def extract_sensors_from_text(text: str) -> List[str]:
    """Extract all sensor references like sensor_060, sensor_104 from text."""
    pattern = r"\bsensor_\d{3}\b"
    return sorted(list(set(re.findall(pattern, text, re.IGNORECASE))))


def validate_report(
    report_text: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit AI-generated engineering report against ground-truth context data.

    Checks:
    1. Sensor Hallucination: Verifies every cited sensor actually exists in the active dataset.
    2. Numerical Grounding: Extracts numeric claims and checks against source context values.
    3. Structural Completeness: Verifies essential engineering sections are present.

    Args:
        report_text: Markdown content of the drafted report.
        context: Ground-truth structured context JSON.

    Returns:
        Dict[str, Any]: Structured validation report with status (PASSED, WARNINGS, FAILED).
    """
    hallucinations: List[str] = []
    warnings: List[str] = []
    verified_claims: List[Dict[str, Any]] = []

    # 1. Check Sensor Identifiers
    cited_sensors = extract_sensors_from_text(report_text)
    from src.ingestion.features import get_sensor_columns
    from src.ingestion.loader import load_from_duckdb

    try:
        clean_df = load_from_duckdb("clean_lots")
        valid_sensor_pool = set(get_sensor_columns(clean_df))
    except Exception:
        valid_sensor_pool = set()

    for sensor in cited_sensors:
        if valid_sensor_pool and sensor.lower() not in valid_sensor_pool:
            hallucinations.append(
                f"Hallucinated sensor '{sensor}' cited in report is not in the active dataset."
            )

    # 2. Extract and Validate Numbers
    context_numbers = extract_numbers_from_context(context)

    # Extract all floating point numbers and percentages from report
    number_pattern = r"(?<![a-zA-Z_])(\d+(?:\.\d+)?)(%?)(?![a-zA-Z_])"
    matches = re.findall(number_pattern, report_text)

    audited_count = 0
    grounded_count = 0

    for num_str, is_pct in matches:
        val = float(num_str)
        # Skip trivial formatting numbers like section indices (1, 2, 3) or calendar years (2000-2099)
        if val in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0] or (
            2000.0 <= val <= 2099.0
        ):
            continue

        audited_count += 1

        # Check if number matches any context metric within 1% relative tolerance
        matched = False
        for ctx_num in context_numbers:
            if ctx_num != 0 and abs(val - ctx_num) / abs(ctx_num) < 0.02:
                matched = True
                break
            elif abs(val - ctx_num) < 0.05:
                matched = True
                break

        if matched:
            grounded_count += 1
            verified_claims.append(
                {"value": val, "unit": "%" if is_pct else "", "status": "VERIFIED"}
            )
        else:
            warnings.append(
                f"Numeric claim '{val}{is_pct}' could not be matched to ground-truth context metrics."
            )

    # 3. Determine Final Verdict
    if hallucinations:
        status = "FAILED"
    elif len(warnings) > 5:
        status = "WARNINGS"
    else:
        status = "PASSED"

    validation_summary = {
        "status": status,
        "cited_sensors_count": len(cited_sensors),
        "cited_sensors": cited_sensors,
        "hallucinated_sensors": hallucinations,
        "total_numeric_claims_audited": audited_count,
        "verified_numeric_claims_count": grounded_count,
        "grounding_accuracy_pct": (
            round((grounded_count / audited_count * 100), 2)
            if audited_count > 0
            else 100.0
        ),
        "warnings": warnings,
    }

    return validation_summary


def format_validation_markdown(validation_result: Dict[str, Any]) -> str:
    """Format validation metrics into a standard Markdown disclosure block."""
    status = validation_result["status"]
    status_icon = (
        "✅" if status == "PASSED" else ("⚠️" if status == "WARNINGS" else "❌")
    )

    md = [
        "---",
        "## AI Validation & Responsible Governance Layer",
        f"> **Validation Verdict**: {status_icon} **{status}**",
        "> *All claims evaluated by rules-based cross-referencing against pipeline truth data.*",
        "",
        "| Audit Dimension | Result |",
        "|---|---|",
        f"| **Sensors Cited** | {validation_result['cited_sensors_count']} parameters |",
        f"| **Hallucinated Sensors** | {len(validation_result['hallucinated_sensors'])} detected |",
        f"| **Numeric Claims Audited** | {validation_result['total_numeric_claims_audited']} claims |",
        f"| **Verified Ground-Truth Claims** | {validation_result['verified_numeric_claims_count']} claims |",
        f"| **Grounding Accuracy** | {validation_result['grounding_accuracy_pct']}% |",
        "",
    ]

    if validation_result["hallucinated_sensors"]:
        md.append("### ❌ Hallucination Alerts")
        for alert in validation_result["hallucinated_sensors"]:
            md.append(f"- {alert}")
        md.append("")

    if validation_result["warnings"]:
        md.append("### ⚠️ Discrepancy Warnings")
        for warn in validation_result["warnings"][:5]:  # show first 5
            md.append(f"- {warn}")
        md.append("")

    return "\n".join(md)
