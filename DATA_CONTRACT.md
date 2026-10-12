
# Phase 1 — Problem and Data Contract

## Planned Business Capabilities

### 1. Churn Risk Classification

Estimate whether a customer becomes inactive during a defined future window.
The dataset provides transaction history, not official cancellation labels.

### 2. Customer Value Regression

Predict net purchase value in the next 90 days from information available
at the prediction timestamp.

### 3. Customer Segmentation

Group customers using past purchasing behaviour, including recency,
frequency, and monetary value.

### 4. Anomaly Detection

Rank unusual transactions or purchasing patterns for investigation.
An anomaly score is not evidence of fraud.

## Source Grain

One row per invoice line item.

This is not a customer-level table.

## Required Source Fields

- InvoiceNo
- StockCode
- Quantity
- InvoiceDate
- UnitPrice

Expected additional fields:

- Description
- CustomerID
- Country

## Canonical Fields

- invoice_no
- stock_code
- description
- quantity
- invoice_date
- unit_price
- customer_id
- country
- is_cancellation
- is_negative_quantity
- line_value

## Validation Rules

- Missing required columns are blocking errors.
- Unparseable dates and quantities must be measured.
- Unparseable prices must be measured.
- Missing customer IDs are allowed in source data.
- Duplicate rows are reported before any removal decision.
- Negative quantities and cancellations are retained and flagged.
- Nonpositive unit prices are reported.
- Source rows must not silently disappear during normalization.

## Time-Safety Rules

- Do not use future information to build features.
- Do not create model targets in Phase 1.
- Do not randomly split time-dependent data in later phases.
- Do not treat the latest incomplete outcome window as a mature label.

## Phase 1 Completion Criteria

- The Python environment works.
- The real dataset has been downloaded.
- Source column names are normalized consistently.
- A data profile and a validation report are produced.
- Automated tests pass.
- Source limitations and processing rules are documented.
