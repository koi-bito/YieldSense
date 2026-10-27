from src.agent.assembler import assemble_engineering_context
from src.agent.drafter import generate_mock_report
from src.agent.validator import validate_report
from src.agent.writer import write_validated_report


def test_assemble_engineering_context():
    """Verify context aggregator compiles all pipeline outputs."""
    context = assemble_engineering_context()
    assert "report_metadata" in context
    assert "data_cleaning" in context
    assert "correlation_analysis" in context
    assert "pca" in context
    assert "spc_monitoring" in context
    assert "statistical_recovery" in context
    assert "model_evaluation" in context
    assert "ml_recovery_ranking" in context


def test_generate_mock_report():
    """Verify mock drafter produces structured markdown containing all key sections."""
    context = assemble_engineering_context()
    report = generate_mock_report(context)

    assert "# YieldSense Engineering Analysis Report" in report
    assert "Executive Summary" in report
    assert "Data Ingestion & Preprocessing Audit" in report
    assert "Parametric Correlation & Failure Signals" in report
    assert "Statistical Process Control (SPC) Drift Monitoring" in report
    assert "Machine Learning Classification & SHAP Attribution" in report
    assert "Recovery & Re-Binning Economics" in report


def test_validator_passes_on_grounded_report():
    """Verify validation layer passes valid report with zero hallucinations."""
    context = assemble_engineering_context()
    report = generate_mock_report(context)

    summary = validate_report(report, context)
    assert summary["status"] in ["PASSED", "WARNINGS"]
    assert len(summary["hallucinated_sensors"]) == 0
    assert summary["grounding_accuracy_pct"] >= 75.0


def test_validator_catches_hallucinated_sensor():
    """Verify validator flags fictitious sensor references as FAILED."""
    context = assemble_engineering_context()
    # Inject nonexistent sensor
    hallucinated_report = "Critical finding: sensor_999 is highly predictive of catastrophic tool failure."

    summary = validate_report(hallucinated_report, context)
    assert summary["status"] == "FAILED"
    assert any("sensor_999" in alert for alert in summary["hallucinated_sensors"])


def test_write_validated_report(tmp_path, monkeypatch):
    """Verify writer appends validation disclosures and saves markdown file."""
    context = assemble_engineering_context()
    report = generate_mock_report(context)
    summary = validate_report(report, context)

    monkeypatch.setattr("src.agent.writer.REPORTS_DIR", tmp_path)

    report_path, full_text = write_validated_report(
        report_body=report,
        validation_summary=summary,
        output_filename="test_report.md",
        save_as_sample=False,
    )

    assert report_path.exists()
    assert "## AI Validation & Responsible Governance Layer" in full_text
    assert f"**{summary['status']}**" in full_text


def test_validator_year_range_exclusion():
    """Verify calendar years in the range 2000-2099 are excluded from strict numeric claim auditing."""
    context = {"context_numbers": [10.0]}
    year_text = (
        "In 2027 and 2035, the tool recipe remained stable following the 2008 baseline."
    )
    summary = validate_report(year_text, context)
    # The years 2027, 2035, 2008 should all be skipped as calendar years
    assert summary["total_numeric_claims_audited"] == 0
