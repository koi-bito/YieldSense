"""End-to-end ingestion pipeline runner."""

import sys

from src.ingestion.cleaner import run_cleaner
from src.ingestion.loader import run_loader


def main() -> int:
    try:
        print("=== Step 1: Loading raw SECOM dataset into DuckDB ===")
        raw_df = run_loader()
        print(f"Raw data loaded: {raw_df.shape[0]} lots, {raw_df.shape[1]} columns.")

        print("\n=== Step 2: Cleaning and auditing sensor features ===")
        clean_df, log_data = run_cleaner()
        print(f"Cleaned dataset: {clean_df.shape[0]} lots, {clean_df.shape[1]} columns.")
        print(f"Initial sensors: {log_data['initial_sensor_count']}")
        print(f"Dropped (>40% missing): {log_data['dropped_high_missing_count']}")
        print(f"Dropped (zero variance): {log_data['dropped_zero_variance_count']}")
        print(f"Final active sensors: {log_data['final_sensor_count']}")
        print(f"Class distribution: {log_data['label_distribution']}")
        print("\nIngestion pipeline completed successfully.")
        return 0
    except Exception as exc:
        print(f"\n❌ Ingestion pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
