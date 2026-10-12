from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def canonical_column(name: object) -> str:
    value = str(name).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    aliases = {
        "invoice": "invoice_no",
        "invoiceno": "invoice_no",
        "invoice_no": "invoice_no",
        "stockcode": "stock_code",
        "stock_code": "stock_code",
        "invoicedate": "invoice_date",
        "invoice_date": "invoice_date",
        "price": "unit_price",
        "unit_price": "unit_price",
        "customer_id": "customer_id",
        "customerid": "customer_id",
        "quantity": "quantity",
        "description": "description",
        "country": "country",
    }
    return aliases.get(value, value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild transactions_normalized.csv from every sheet in the official "
            "Online Retail II Excel workbook."
        )
    )
    parser.add_argument(
        "workbook",
        type=Path,
        help=r"Path to the full Online Retail II workbook, e.g. data\raw\online_retail_II.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/transactions_normalized.csv"),
        help="Output CSV path (default: data/processed/transactions_normalized.csv)",
    )
    args = parser.parse_args()

    if not args.workbook.is_file():
        raise FileNotFoundError(f"Workbook not found: {args.workbook.resolve()}")

    # sheet_name=None is deliberate: both Year 2009-2010 and Year 2010-2011
    # must be included, not just the first worksheet.
    sheets = pd.read_excel(args.workbook, sheet_name=None, engine="openpyxl")
    if len(sheets) < 2:
        raise ValueError(
            f"Expected the full two-sheet Online Retail II workbook; found {len(sheets)} sheet(s): "
            f"{list(sheets)}. Do not continue with a one-year extract."
        )

    frames: list[pd.DataFrame] = []
    for sheet_name, sheet in sheets.items():
        data = sheet.copy()
        data.columns = [canonical_column(c) for c in data.columns]
        required = {"invoice_no", "invoice_date", "quantity", "unit_price", "customer_id"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Sheet {sheet_name!r} is missing expected columns: {sorted(missing)}")
        data["source_sheet"] = sheet_name
        frames.append(data)

    tx = pd.concat(frames, ignore_index=True, sort=False)
    tx["invoice_no"] = tx["invoice_no"].astype("string").str.strip()
    tx["invoice_date"] = pd.to_datetime(tx["invoice_date"], errors="coerce", dayfirst=True)
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    tx["unit_price"] = pd.to_numeric(tx["unit_price"], errors="coerce")
    # A nullable integer key makes the ID type consistent while retaining missing IDs.
    tx["customer_id"] = pd.to_numeric(tx["customer_id"], errors="coerce").round().astype("Int64")

    tx = tx.loc[tx["invoice_date"].notna()].copy()
    if tx.empty:
        raise ValueError("No rows with valid invoice dates were read from the workbook.")

    min_date = tx["invoice_date"].min().normalize()
    max_date = tx["invoice_date"].max().normalize()
    print("Worksheets included:", ", ".join(sheets.keys()))
    print(f"Rows after date parsing: {len(tx):,}")
    print(f"Transaction date range: {min_date.date()} through {max_date.date()}")
    if min_date > pd.Timestamp("2010-06-01"):
        raise ValueError(
            "The combined transaction file starts after the project's June 2010 snapshots. "
            "This is not the complete Online Retail II history; verify that the workbook has both year sheets."
        )
    if max_date < pd.Timestamp("2011-09-30"):
        raise ValueError(
            "The transaction file ends before the planned validation/test history. "
            "Verify that the full workbook was supplied."
        )

    preferred = [
        "customer_id", "invoice_no", "invoice_date", "quantity", "unit_price",
        "stock_code", "description", "country", "source_sheet",
    ]
    columns = [c for c in preferred if c in tx.columns] + [c for c in tx.columns if c not in preferred]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tx[columns].to_csv(args.output, index=False)
    print(f"Wrote: {args.output.resolve()}")
    print("Note: rerun the Phase 2 snapshot/feature builder and Phase 3 target preparation after replacing this source.")


if __name__ == "__main__":
    main()
