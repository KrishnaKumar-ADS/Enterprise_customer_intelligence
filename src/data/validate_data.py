from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.schema import REQUIRED_COLUMNS, make_profile, normalize_transactions

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "project.yaml"
RAW_PATH = ROOT / "data" / "raw" / "online_retail_ii.csv"
NORMALIZED_PATH = ROOT / "data" / "processed" / "transactions_normalized.csv"
PROFILE_PATH = ROOT / "reports" / "data_profile.json"
VALIDATION_PATH = ROOT / "reports" / "data_validation.json"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def main() -> None:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration not found: {CONFIG_PATH}")
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}\n"
            "Run: python -m src.data.download_data"
        )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rules = config["validation"]

    print(f"Reading source data: {RAW_PATH}")
    raw = pd.read_csv(RAW_PATH, low_memory=False)

    errors: list[str] = []
    warnings: list[str] = []

    if len(raw) < int(rules["minimum_rows"]):
        errors.append(
            f"Row count {len(raw)} is below the required minimum "
            f"{rules['minimum_rows']}."
        )

    normalized = normalize_transactions(raw)

    missing_after_normalization = [
        col for col in REQUIRED_COLUMNS if col not in normalized.columns
    ]
    if missing_after_normalization:
        errors.append(
            f"Required fields missing: {missing_after_normalization}"
        )

    if len(normalized) != len(raw):
        errors.append(
            f"Row preservation failed: raw={len(raw)}, "
            f"normalized={len(normalized)}."
        )

    # Evaluate parsing coverage before writing normalized output.
    date_ratio = float(normalized["invoice_date"].notna().mean())
    numeric_ratios = {
        col: float(normalized[col].notna().mean())
        for col in ("quantity", "unit_price")
    }

    date_threshold = float(
        rules["minimum_valid_invoice_date_ratio"]
    )
    numeric_threshold = float(
        rules["minimum_numeric_parse_ratio"]
    )

    if date_ratio < date_threshold:
        errors.append(
            f"Valid invoice date ratio {date_ratio:.4%} is below "
            f"the minimum {date_threshold:.2%}."
        )

    for col, ratio in numeric_ratios.items():
        if ratio < numeric_threshold:
            errors.append(
                f"Valid {col} ratio {ratio:.4%} is below "
                f"the minimum {numeric_threshold:.2%}."
            )

    required_missing = {
        col: int(normalized[col].isna().sum())
        for col in ("invoice_no", "stock_code")
    }
    for col, count in required_missing.items():
        if count:
            errors.append(
                f"{col} contains {count:,} missing values."
            )

    profile = make_profile(normalized)

    if profile.get("missing_customer_id_count", 0):
        warnings.append(
            "Some transactions have missing customer IDs; "
            "define a customer-level exclusion/handling rule later."
        )

    if profile["exact_duplicate_rows"]:
        warnings.append(
            "Exact duplicate rows exist. They are reported but were not "
            "automatically removed."
        )

    if profile.get("cancellation_invoice_line_count", 0):
        warnings.append(
            "Cancellation invoice lines exist and have been retained."
        )

    if profile.get("negative_quantity_line_count", 0):
        warnings.append(
            "Negative quantities exist and have been retained."
        )

    if profile.get("nonpositive_unit_price_line_count", 0):
        warnings.append(
            "Nonpositive unit prices exist and need a documented "
            "business-rule decision before revenue analysis."
        )

    NORMALIZED_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(
        NORMALIZED_PATH,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S",
    )

    write_json(PROFILE_PATH, profile)

    result = {
        "status": "PASSED" if not errors else "FAILED",
        "raw_path": str(RAW_PATH.relative_to(ROOT)),
        "normalized_path": str(NORMALIZED_PATH.relative_to(ROOT)),
        "raw_rows": int(len(raw)),
        "normalized_rows": int(len(normalized)),
        "row_count_preserved": bool(len(raw) == len(normalized)),
        "valid_invoice_date_ratio": round(date_ratio, 6),
        "numeric_parse_ratios": {
            key: round(value, 6) for key, value in numeric_ratios.items()
        },
        "errors": errors,
        "warnings": warnings,
    }
    write_json(VALIDATION_PATH, result)

    print("\nVALIDATION RESULT")
    print(f"Status: {result['status']}")
    print(f"Source rows: {len(raw):,}")
    print(f"Normalized rows: {len(normalized):,}")
    print(f"Valid date ratio: {date_ratio:.4%}")

    for col, ratio in numeric_ratios.items():
        print(f"Valid {col} ratio: {ratio:.4%}")

    if errors:
        print("\nERRORS")
        for item in errors:
            print(f"- {item}")

    if warnings:
        print("\nWARNINGS")
        for item in warnings:
            print(f"- {item}")

    print(f"\nProfile report: {PROFILE_PATH}")
    print(f"Validation report: {VALIDATION_PATH}")
    print(f"Normalized data: {NORMALIZED_PATH}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()