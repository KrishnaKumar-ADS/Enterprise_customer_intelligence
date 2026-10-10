
"""Typed contract for one normalized transaction record."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TransactionRecord(BaseModel):
    """Canonical structure for a single invoice line."""

    model_config = ConfigDict(extra="allow")

    invoice_no: str = Field(min_length=1)
    stock_code: str = Field(min_length=1)
    description: str | None = None
    quantity: float
    invoice_date: datetime
    unit_price: float
    customer_id: str | None = None
    country: str | None = None
    is_cancellation: bool = False
    is_negative_quantity: bool = False
    line_value: float | None = None
