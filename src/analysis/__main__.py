"""CLI entrypoint to execute all statistical analyses."""

import sys

from src.analysis.correlation import run_correlation_analysis
from src.analysis.pca import run_pca_pipeline
from src.analysis.recovery import run_recovery_analysis
from src.analysis.spc import run_spc_analysis


def main() -> int:
    try:
        print("=== Statistical Analysis 1: Sensor Correlation Analysis ===")
        corr_df = run_correlation_analysis()
        sig_count = corr_df["is_significant_05"].sum()
        print(f"Calculated correlations for {len(corr_df)} sensors.")
        print(f"Statistically significant sensors (p < 0.05): {sig_count}")
        print(f"Top 3 correlated: {list(corr_df['sensor'].head(3))}")

        print("\n=== Statistical Analysis 2: Principal Component Analysis (PCA) ===")
        pca_summary = run_pca_pipeline()
        print(
            f"Dimensions reduced from {pca_summary['total_sensors']} to {pca_summary['n_components_retained']} PCs"
        )
        print(
            f"Capturing {pca_summary['total_explained_variance']:.1%} of overall variance."
        )

        print("\n=== Statistical Analysis 3: Statistical Process Control (SPC) ===")
        spc_df = run_spc_analysis()
        print(f"Detected {len(spc_df)} 3-sigma drift events across top 10 sensors.")

        print("\n=== Statistical Analysis 4: Marginal Recovery Candidates ===")
        rec_df = run_recovery_analysis()
        recoverable = rec_df["is_recovery_candidate"].sum()
        print(
            f"Identified {recoverable} recoverable candidate lots out of {len(rec_df)} failing lots."
        )

        print("\nAll statistical analysis modules completed successfully.")
        return 0
    except Exception as exc:
        print(f"\n❌ Statistical analysis pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
