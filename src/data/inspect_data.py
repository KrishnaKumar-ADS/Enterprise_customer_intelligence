
"""Inspect downloaded source data and write a data profile."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.schema import make_profile, normalize_transactions

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "online_retail_ii.csv"
REPORT_PATH = ROOT / "reports" / "source_profile.json"


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_PATH}\n"
            "Run: python -m src.data.download_data"
        )

    print(f"Loading raw data: {RAW_PATH}")
    raw = pd.read_csv(RAW_PATH, low_memory=False)

    print("Normalizing an in-memory copy for profiling...")
    normalized = normalize_transactions(raw)
    profile = make_profile(normalized)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDATA PROFILE")
    print(f"Rows: {profile['row_count']:,}")
    print(f"Columns: {profile['column_count']}")
    print(f"Unique customers: {profile.get('unique_customers_non_missing', 0):,}")
    print(f"Missing customer IDs: {profile.get('missing_customer_id_count', 0):,}")
    print(f"Exact duplicate rows: {profile['exact_duplicate_rows']:,}")
    print(f"Invoice date range: {profile.get('invoice_date_min')} to "
          f"{profile.get('invoice_date_max')}")
    print(f"Report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
