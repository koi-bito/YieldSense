"""Report writer combining drafted text, validation blocks, and persisting to disk."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.agent.validator import format_validation_markdown
from src.config import REPORTS_DIR


def write_validated_report(
    report_body: str,
    validation_summary: Dict[str, Any],
    output_filename: Optional[str] = None,
    save_as_sample: bool = True,
) -> Tuple[Path, str]:
    """Combine report body and validation audit block, and save to disk."""
    validation_block = format_validation_markdown(validation_summary)

    header = f"""<!--
YIELDSENSE AUTOMATED ENGINEERING REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Validation Status: {validation_summary['status']}
-->

"""
    full_report = header + report_body + "\n\n" + validation_block

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"engineering_report_{timestamp_str}.md"

    report_path = REPORTS_DIR / output_filename
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    if save_as_sample:
        sample_path = REPORTS_DIR / "sample_engineering_report.md"
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(full_report)

    return report_path, full_report
