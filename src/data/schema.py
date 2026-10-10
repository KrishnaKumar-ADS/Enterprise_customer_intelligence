
"""Column normalization, record flags, and data profiling."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

ALIASES = {
    "invoice_no": {
        "invoiceno", "invoice", "invoicenumber", "invoiceid", "invoice_no"
    },
    "stock_code": {
        "stockcode", "stock", "itemcode", "productcode", "stock_code"
    },
    "description": {
        "description", "itemdescription", "productdescription", "desc"
    },
    "quantity": {"quantity", "qty", "units"},
    "invoice_date": {
        "invoicedate", "invoicedatetime", "transactiondate", "date",
        "invoice_date"
    },
    "unit_price": {
        "unitprice", "price", "unit_price", "itemprice"
    },
    "customer_id": {
        "customerid", "customer", "customer_id", "clientid"
    },
    "country": {"country", "customer_country"},
}

REQUIRED_COLUMNS = [
    "invoice_no",
    "stock_code",
    "quantity",
    "invoice_date",
    "unit_price",
]


def _key(value: Any) -> str:
    """Convert a column label to a compact lowercase matching key."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map recognized source headers to canonical field names."""
    reverse = {
        _key(alias): canonical
        for canonical, aliases in ALIASES.items()
        for alias in aliases
    }

    rename: dict[Any, str] = {}
    for col in frame.columns:
        target = reverse.get(_key(col))
        if target is not None:
            rename[col] = target

    target_names = list(rename.values())
    duplicates = sorted({
        name for name in target_names if target_names.count(name) > 1
    })

    if duplicates:
        raise ValueError(
            "Multiple source columns map to the same canonical field: "
            f"{duplicates}. Inspect the raw columns."
        )

    return frame.rename(columns=rename).copy()


def _clean_customer_id(series: pd.Series) -> pd.Series:
    """Preserve customer IDs as strings while removing numeric '.0' suffixes."""
    original = series.astype("string").str.strip()
    numeric = pd.to_numeric(original, errors="coerce")

    integer_mask = numeric.notna() & ((numeric % 1) == 0)

    output = original.copy()
    if integer_mask.any():
        output.loc[integer_mask] = (
            numeric.loc[integer_mask]
            .astype("int64")
            .astype("string")
        )

    blank_mask = output.str.lower().isin(["", "nan", "none", "<na>"])
    output.loc[blank_mask.fillna(False)] = pd.NA
    return output


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize fields while preserving every source row."""
    out = canonicalize_columns(frame)

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after mapping: {missing}. "
            f"Actual source columns: {list(frame.columns)}"
        )

    out["invoice_no"] = out["invoice_no"].astype("string").str.strip()
    out["stock_code"] = out["stock_code"].astype("string").str.strip()

    if "description" not in out.columns:
        out["description"] = pd.NA
    if "customer_id" not in out.columns:
        out["customer_id"] = pd.NA
    if "country" not in out.columns:
        out["country"] = pd.NA

    out["description"] = out["description"].astype("string").str.strip()
    out["country"] = out["country"].astype("string").str.strip()
    out["customer_id"] = _clean_customer_id(out["customer_id"])

    for col in ("description", "country"):
        blank = out[col].eq("").fillna(False)
        out.loc[blank, col] = pd.NA

    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["unit_price"] = pd.to_numeric(
        out["unit_price"], errors="coerce"
    )

    # Invoice dates in this source use day-first date formatting.
    out["invoice_date"] = pd.to_datetime(
        out["invoice_date"],
        errors="coerce",
        dayfirst=True,
    )

    invoice_text = out["invoice_no"].astype("string")
    out["is_cancellation"] = (
        invoice_text.str.upper().str.startswith("C", na=False)
    )
    out["is_negative_quantity"] = (
        out["quantity"].lt(0).fillna(False)
    )
    out["line_value"] = out["quantity"] * out["unit_price"]

    return out


def make_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize row counts, missingness, date range, and review conditions."""
    missing_counts = frame.isna().sum()

    profile: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "columns": [str(c) for c in frame.columns],
        "dtypes": {
            str(c): str(dtype) for c, dtype in frame.dtypes.items()
        },
        "missing_count": {
            str(c): int(n) for c, n in missing_counts.items()
        },
        "missing_pct": {
            str(c): round(float(n / len(frame) * 100), 4)
            if len(frame) else 0.0
            for c, n in missing_counts.items()
        },
        "exact_duplicate_rows": int(frame.duplicated().sum()),
    }

    if "invoice_date" in frame.columns:
        dates = pd.to_datetime(frame["invoice_date"], errors="coerce")
        profile["invoice_date_min"] = (
            dates.min().isoformat() if dates.notna().any() else None
        )
        profile["invoice_date_max"] = (
            dates.max().isoformat() if dates.notna().any() else None
        )
        profile["valid_invoice_date_count"] = int(dates.notna().sum())
        profile["invalid_invoice_date_count"] = int(dates.isna().sum())

    if "customer_id" in frame.columns:
        profile["unique_customers_non_missing"] = int(
            frame["customer_id"].nunique(dropna=True)
        )
        profile["missing_customer_id_count"] = int(
            frame["customer_id"].isna().sum()
        )

    if "invoice_no" in frame.columns:
        invoice = frame["invoice_no"].astype("string")
        profile["unique_invoices_non_missing"] = int(
            invoice.nunique(dropna=True)
        )
        profile["cancellation_invoice_line_count"] = int(
            invoice.str.upper().str.startswith("C", na=False).sum()
        )
        profile["missing_invoice_no_count"] = int(
            frame["invoice_no"].isna().sum()
        )

    if "quantity" in frame.columns:
        quantity = pd.to_numeric(frame["quantity"], errors="coerce")
        profile["negative_quantity_line_count"] = int(quantity.lt(0).sum())
        profile["invalid_quantity_count"] = int(quantity.isna().sum())

    if "unit_price" in frame.columns:
        price = pd.to_numeric(frame["unit_price"], errors="coerce")
        profile["nonpositive_unit_price_line_count"] = int(
            price.le(0).sum()
        )
        profile["invalid_unit_price_count"] = int(price.isna().sum())

    if "line_value" in frame.columns:
        values = pd.to_numeric(frame["line_value"], errors="coerce")
        profile["total_line_value_including_negative_lines"] = round(
            float(values.sum()), 2
        )

    return profile
