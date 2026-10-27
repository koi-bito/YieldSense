"""CLI entry point for YieldSense automated engineering report generator."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.assembler import assemble_engineering_context  # noqa: E402
from src.agent.drafter import draft_engineering_report  # noqa: E402
from src.agent.validator import validate_report  # noqa: E402
from src.agent.writer import write_validated_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="YieldSense: AI-Assisted NAND Yield & Recovery Report Generator"
    )
    parser.add_argument(
        "--lot-id",
        type=str,
        default=None,
        help="Specific lot ID (e.g. LOT_0003) to generate focused recovery attribution for.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output filename in reports/ directory.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  YieldSense AI-Assisted Engineering Report Generator")
    print("=" * 65)

    print("\n[1/4] Assembling structured engineering context from pipeline...")
    context = assemble_engineering_context(lot_id=args.lot_id)
    print(f"  Context compiled across {len(context)} analytical dimensions.")

    print("\n[2/4] Drafting engineering analysis report...")
    report_draft = draft_engineering_report(context)
    print(f"  Draft generated ({len(report_draft.split())} words).")

    print("\n[3/4] Running rules-based validation and hallucination audit layer...")
    validation_summary = validate_report(report_draft, context)
    print(f"  Verdict: {validation_summary['status']}")
    print(f"  Sensors audited: {validation_summary['cited_sensors_count']}")
    print(f"  Hallucinations: {len(validation_summary['hallucinated_sensors'])}")
    print(
        f"  Grounded numeric claims: {validation_summary['verified_numeric_claims_count']}/{validation_summary['total_numeric_claims_audited']} ({validation_summary['grounding_accuracy_pct']}%)"
    )

    print("\n[4/4] Writing validated report artifact...")
    report_path, _ = write_validated_report(
        report_body=report_draft,
        validation_summary=validation_summary,
        output_filename=args.output,
        save_as_sample=True,
    )
    print(f"  Saved to: {report_path.resolve()}")
    print("  Updated sample report: reports/sample_engineering_report.md")

    print("\n" + "=" * 65)
    print("  REPORT GENERATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
