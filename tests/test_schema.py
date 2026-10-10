
"""Unit tests for transaction schema normalization."""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.schema import (
    canonicalize_columns,
    normalize_transactions,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame({
        "InvoiceNo": ["536365", "C536366", "536367"],
        "StockCode": ["85123A", "22423", "85099B"],
        "Description": ["Product A", "Product B", "Product C"],
        "Quantity": [6, -2, 3],
        "InvoiceDate": [
            "01/12/2009 08:26",
            "01/12/2009 08:27",
            "02/12/2009 09:00",
        ],
        "UnitPrice": [2.55, 4.00, 1.50],
        "CustomerID": [17850.0, None, 17850.0],
        "Country": ["United Kingdom", "United Kingdom", "France"],
    })


def test_column_mapping() -> None:
    frame = pd.DataFrame({
        "InvoiceNo": ["100"],
        "StockCode": ["A"],
        "Quantity": [1],
        "InvoiceDate": ["01/12/2009"],
        "UnitPrice": [2.0],
    })

    result = canonicalize_columns(frame)

    assert "invoice_no" in result.columns
    assert "stock_code" in result.columns
    assert "invoice_date" in result.columns
    assert "unit_price" in result.columns


def test_normalization_preserves_rows(sample_data: pd.DataFrame) -> None:
    result = normalize_transactions(sample_data)

    assert len(result) == len(sample_data)
    assert "line_value" in result.columns
    assert result.loc[0, "line_value"] == pytest.approx(15.30)


def test_cancellations_and_returns_are_flagged(
    sample_data: pd.DataFrame,
) -> None:
    result = normalize_transactions(sample_data)

    assert bool(result.loc[1, "is_cancellation"])
    assert bool(result.loc[1, "is_negative_quantity"])
    assert result.loc[1, "quantity"] == -2


def test_dates_use_day_first_convention(
    sample_data: pd.DataFrame,
) -> None:
    result = normalize_transactions(sample_data)

    assert result.loc[0, "invoice_date"] == pd.Timestamp(
        "2009-12-01 08:26"
    )


def test_missing_customer_id_is_preserved(
    sample_data: pd.DataFrame,
) -> None:
    result = normalize_transactions(sample_data)

    assert pd.isna(result.loc[1, "customer_id"])
    assert result.loc[0, "customer_id"] == "17850"


def test_missing_required_column_raises_error() -> None:
    frame = pd.DataFrame({
        "InvoiceNo": ["100"],
        "StockCode": ["A"],
        "Quantity": [1],
        "InvoiceDate": ["01/12/2009"],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        normalize_transactions(frame)
