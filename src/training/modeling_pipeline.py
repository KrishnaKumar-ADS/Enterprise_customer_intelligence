"""Consolidated customer modeling pipeline and model tuning workflows."""
from __future__ import annotations


# --- Consolidated from prepare_phase_3_data.py ---

"""Derive the PDF-aligned 30-day inactivity target without editing Phase 2 data."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.training.common import ROOT, load_config, project_path, json_dump, sha256_file


def _normalize_ids(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    result = result.str.replace(r"\.0$", "", regex=True)
    blank = result.isin(["", "nan", "None", "<NA>"])
    result.loc[blank.fillna(False)] = pd.NA
    return result



def build_churn_30d_labels(
    snapshots: pd.DataFrame, transactions: pd.DataFrame, horizon_days: int = 30
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return copied snapshots with mature 30-day inactivity labels.

    The rule is: target=1 when there is no qualifying purchase invoice in
    [snapshot_date, snapshot_date + horizon_days).
    """
    if horizon_days != 30:
        raise ValueError("This PDF-alignment helper is configured for a 30-day churn horizon.")
    result = snapshots.copy()
    tx = transactions.copy()
    needed_snapshot = {"customer_id", "snapshot_date"}
    needed_tx = {"customer_id", "invoice_no", "invoice_date", "quantity", "unit_price", "is_cancellation"}
    if not needed_snapshot.issubset(result.columns):
        raise ValueError(f"Snapshot input is missing fields: {sorted(needed_snapshot - set(result.columns))}")
    if not needed_tx.issubset(tx.columns):
        raise ValueError(f"Transaction input is missing fields: {sorted(needed_tx - set(tx.columns))}")

    result["customer_id"] = _normalize_ids(result["customer_id"])
    result["snapshot_date"] = pd.to_datetime(result["snapshot_date"], errors="coerce").dt.normalize()
    if result["snapshot_date"].isna().any():
        raise ValueError("Snapshot date contains missing or invalid values")
    tx["customer_id"] = _normalize_ids(tx["customer_id"])
    tx["invoice_date"] = pd.to_datetime(tx["invoice_date"], errors="coerce")
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce")
    tx["unit_price"] = pd.to_numeric(tx["unit_price"], errors="coerce")
    tx["invoice_no"] = tx["invoice_no"].astype("string").str.strip()
    cancellation = tx["is_cancellation"].astype("string").str.lower().isin(["true", "1", "yes"])

    observed = tx.loc[tx["invoice_date"].notna()].copy()
    if observed.empty:
        raise ValueError("No valid transaction timestamps were found")
    observed_end_exclusive = observed["invoice_date"].max().normalize() + pd.Timedelta(days=1)
    sales_mask = (
        observed["customer_id"].notna()
        & ~cancellation.loc[observed.index]
        & observed["quantity"].gt(0)
        & observed["unit_price"].gt(0)
    )
    sales = observed.loc[sales_mask, ["customer_id", "invoice_no", "invoice_date"]].copy()
    if sales.empty:
        raise ValueError("No qualifying purchases are available to create the churn target")
    observed_start = observed["invoice_date"].min().normalize()
    result["target_churn_30d"] = np.nan
    result["target_churn_30d_matured"] = False
    result["churn_30d_outcome_end_exclusive"] = result["snapshot_date"] + pd.Timedelta(days=horizon_days)
    for raw_date in sorted(result["snapshot_date"].unique()):
        snapshot_date = pd.Timestamp(raw_date).normalize()
        end_date = snapshot_date + pd.Timedelta(days=horizon_days)
        mask = result["snapshot_date"].eq(snapshot_date)
        matured = (
            snapshot_date >= observed_start and end_date <= observed_end_exclusive
        )
        result.loc[mask, "target_churn_30d_matured"] = bool(matured)
        if not matured:
            continue
        future = sales.loc[
            sales["invoice_date"].ge(snapshot_date)
            & sales["invoice_date"].lt(end_date)
        ]
        future_orders = future.groupby("customer_id")["invoice_no"].nunique()
        counts = result.loc[mask, "customer_id"].map(future_orders).fillna(0)
        result.loc[mask, "target_churn_30d"] = counts.eq(0).astype("int8").to_numpy()

    if result.duplicated(["customer_id", "snapshot_date"]).any():
        raise ValueError("Duplicate customer/snapshot keys found in Phase 2 input")
    return result, observed_end_exclusive

def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    source_snapshot_path = project_path(paths.get("customer_snapshots_source", "data/processed/customer_snapshots.csv"))
    destination_path = project_path(paths["customer_snapshots"])
    tx_path = project_path("data/processed/transactions_normalized.csv")
    validation_path = project_path("reports/data_validation.json")

    if not validation_path.exists():
        raise FileNotFoundError("Phase 1 validation report is missing; complete Phase 1 first.")
    import json
    phase1 = json.loads(validation_path.read_text(encoding="utf-8"))
    if phase1.get("status") != "PASSED":
        raise RuntimeError("Phase 1 validation has not passed. Refusing to prepare Phase 3 targets.")
    for path in (source_snapshot_path, tx_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 1/2 input is missing: {path}")

    print(f"Reading Phase 2 snapshots: {source_snapshot_path}")
    snapshots = pd.read_csv(source_snapshot_path, low_memory=False, dtype={"customer_id": "string"})
    print(f"Reading normalized transaction history: {tx_path}")
    tx = pd.read_csv(tx_path, low_memory=False, dtype={"customer_id": "string"})

    required_snapshot = {"customer_id", "snapshot_date", "split", "target_net_value_90d_gbp"}
    required_tx = {"customer_id", "invoice_no", "invoice_date", "quantity", "unit_price", "is_cancellation"}
    if not required_snapshot.issubset(snapshots.columns):
        raise ValueError(f"Phase 2 snapshot file lacks fields: {sorted(required_snapshot - set(snapshots.columns))}")
    if not required_tx.issubset(tx.columns):
        raise ValueError(f"Normalized transaction file lacks fields: {sorted(required_tx - set(tx.columns))}")

    horizon = int(cfg["supervised"].get("churn_horizon_days", 30))
    snapshots, observed_end_exclusive = build_churn_30d_labels(snapshots, tx, horizon)

    # All supervised partitions require a fully observed label. The value task
    # also requires mature 90-day labels from the Phase 2 pipeline.
    unlabelled_development = snapshots["split"].isin(["train", "validation", "test"]) & snapshots["target_churn_30d"].isna()
    if unlabelled_development.any():
        raise ValueError(
            f"{int(unlabelled_development.sum())} train/validation/test rows do not have mature 30-day churn labels. "
            "Regenerate Phase 2 with mature target windows or revise split dates."
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(destination_path, index=False)
    report = {
        "status": "PASSED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_file": str(source_snapshot_path.relative_to(ROOT)),
        "source_snapshot_sha256": sha256_file(source_snapshot_path),
        "transaction_file": str(tx_path.relative_to(ROOT)),
        "transaction_sha256": sha256_file(tx_path),
        "output_snapshot_file": str(destination_path.relative_to(ROOT)),
        "output_snapshot_sha256": sha256_file(destination_path),
        "row_count": int(len(snapshots)),
        "unique_snapshot_dates": int(snapshots["snapshot_date"].nunique()),
        "matured_30d_rows": int(snapshots["target_churn_30d_matured"].sum()),
        "target_churn_30d_positive_rate": float(snapshots["target_churn_30d"].mean()),
        "target_churn_30d_positive_count": int(snapshots["target_churn_30d"].sum()),
        "churn_horizon_days": horizon,
        "value_horizon_days": int(cfg["supervised"].get("value_horizon_days", 90)),
        "split_counts": {str(k): int(v) for k, v in snapshots["split"].value_counts().items()},
        "definition": "1 if no qualifying purchase occurs in [snapshot_date, snapshot_date + 30 days); 0 otherwise.",
        "limitations": [
            "This is an inactivity proxy, not verified cancellation.",
            "Phase 2 source snapshot is preserved; this script writes a Phase 3 copy.",
        ],
    }
    json_dump(project_path("reports/phase3/churn_target_manifest.json"), report)
    print(f"Created: {destination_path}")
    print(f"Rows: {len(snapshots):,}; snapshots: {snapshots.snapshot_date.nunique()}; target-positive rate: {snapshots.target_churn_30d.mean():.3%}")
    print("Churn horizon: 30 days (aligned with the PDF business specification). Value horizon remains 90 days.")


# --- Consolidated from phase_4_supervised_model_comparison.py ---

"""Phase 4: supervised model comparison, validation error analysis, and latency.

Consumes Phase 3 data manifests, model-comparison tables, CV folds, and model
bundles. Uses train and model-selection validation rows only. The test split is
never used for fitting, scoring, model selection, or report generation.
"""

import argparse
import io
import json
import math
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.training.common import (
    ROOT,
    build_pipeline,
    feature_columns,
    load_config,
    load_snapshots,
    process_rss_mb,
    project_path,
    sha256_file,
    split_validation_roles,
)

CONFIG_PATH = ROOT / "configs" / "project.yaml"
PHASE4_REPORT_DIR = ROOT / "reports" / "phase4"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Phase 3 file is missing: {path.relative_to(ROOT)}.\n"
            "Complete data preparation first with the dataset pipeline."
        )
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"Required Phase 3 file is empty: {path}")
    return frame


def load_phase4_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Phase 4 config missing: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["comparison"]


def parse_params(value: Any) -> dict[str, Any]:
    """Parse model params saved as a JSON string in Phase 3 tables."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    try:
        loaded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return proba[:, classes.index(1)]
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, -1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float).ravel()
        # Monotonic conversion for ranking diagnostics only. This is not a
        # probability calibration step.
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -35, 35)))
    raise TypeError(f"{type(model).__name__} cannot produce a churn ranking score")


def _top_capacity_metrics(y: np.ndarray, score: np.ndarray, fraction: float) -> tuple[float, float, int]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), 0
    k = max(1, min(n, int(math.ceil(n * fraction))))
    selected = np.argsort(-score, kind="stable")[:k]
    tp = int(y[selected].sum())
    positive = int(y.sum())
    return (tp / positive if positive else float("nan"), tp / k, k)


def comparison_classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float, capacity_fraction: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    prediction = (score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel().tolist()
    recall_at_capacity, precision_at_capacity, capacity_n = _top_capacity_metrics(
        y, score, capacity_fraction
    )
    return {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(np.mean(y)) if len(y) else None,
        "pr_auc_average_precision": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else None,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "recall_at_capacity": recall_at_capacity,
        "precision_among_contacted_at_capacity": precision_at_capacity,
        "capacity_selected_count": int(capacity_n),
    }


def comparison_regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = y - pred
    result: dict[str, Any] = {
        "n": int(len(y)),
        "mae_gbp": float(mean_absolute_error(y, pred)),
        "mse_gbp_squared": float(mean_squared_error(y, pred)),
        "rmse_gbp": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None,
        "mean_error_actual_minus_predicted_gbp": float(np.mean(residual)),
        "median_absolute_error_gbp": float(np.median(np.abs(residual))),
        "p90_absolute_error_gbp": float(np.quantile(np.abs(residual), 0.90)),
    }
    total = float(y.sum())
    if len(y) and total > 0:
        k = max(1, int(math.ceil(len(y) * 0.10)))
        chosen = np.argsort(-pred, kind="stable")[:k]
        result["top_predicted_decile_actual_value_gbp"] = float(y[chosen].sum())
        result["top_predicted_decile_fraction_total_value"] = float(y[chosen].sum() / total)
    else:
        result["top_predicted_decile_actual_value_gbp"] = None
        result["top_predicted_decile_fraction_total_value"] = None
    return result


def _safe_metric_group(group: pd.DataFrame, task: str) -> dict[str, Any]:
    if task == "churn":
        return comparison_classification_metrics(
            group["actual"].astype(int).to_numpy(),
            group["score"].astype(float).to_numpy(),
            float(group["threshold"].iloc[0]),
            0.10,
        )
    return comparison_regression_metrics(
        group["actual"].astype(float).to_numpy(),
        group["score"].astype(float).to_numpy(),
    )


def _latency_benchmark(model: Any, X: pd.DataFrame, task: str, rows: int, batch_rows: int) -> dict[str, Any]:
    if X.empty:
        return {"latency_status": "NO_VALIDATION_ROWS"}
    sample = X.iloc[: min(len(X), max(1, rows))]
    fn = (lambda frame: _positive_probability(model, frame)) if task == "churn" else (lambda frame: model.predict(frame))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(sample.iloc[:1])
            elapsed_ms = []
            for index in range(len(sample)):
                row = sample.iloc[[index]]
                started = time.perf_counter()
                fn(row)
                elapsed_ms.append((time.perf_counter() - started) * 1000.0)
            batch = X.iloc[: min(len(X), max(1, batch_rows))]
            started = time.perf_counter()
            fn(batch)
            batch_elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "latency_status": "PASSED",
            "p50_ms_per_row": float(np.percentile(elapsed_ms, 50)),
            "p95_ms_per_row": float(np.percentile(elapsed_ms, 95)),
            "p99_ms_per_row": float(np.percentile(elapsed_ms, 99)),
            "batch_rows": int(len(batch)),
            "batch_elapsed_ms": float(batch_elapsed * 1000),
            "batch_rows_per_second": float(len(batch) / batch_elapsed),
        }
    except Exception as exc:
        return {"latency_status": "FAILED", "latency_error": f"{type(exc).__name__}: {exc}"}


def _serialization_roundtrip(model: Any, task: str, X: pd.DataFrame) -> dict[str, Any]:
    if X.empty:
        return {"serialization_status": "SKIPPED_NO_ROWS"}
    sample = X.iloc[: min(10, len(X))]
    try:
        before = _positive_probability(model, sample) if task == "churn" else np.asarray(model.predict(sample), dtype=float)
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        size = buffer.tell()
        buffer.seek(0)
        restored = joblib.load(buffer)
        after = _positive_probability(restored, sample) if task == "churn" else np.asarray(restored.predict(sample), dtype=float)
        maximum_difference = float(np.max(np.abs(np.asarray(before) - np.asarray(after))))
        return {
            "serialization_status": "PASSED" if np.allclose(before, after, rtol=1e-12, atol=1e-12) else "FAILED",
            "serialized_size_bytes": int(size),
            "roundtrip_max_absolute_difference": maximum_difference,
        }
    except Exception as exc:
        return {"serialization_status": "FAILED", "serialization_error": f"{type(exc).__name__}: {exc}"}


def _validation_partition(frame: pd.DataFrame, target: str, task: str, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subset = frame.loc[frame["split"].isin(["train", "validation"])].copy()
    subset[target] = pd.to_numeric(subset[target], errors="coerce")
    subset = subset.loc[np.isfinite(subset[target].astype(float))].copy()
    if task == "churn":
        subset = subset.loc[subset[target].isin([0, 1])].copy()
        subset[target] = subset[target].astype(int)
    elif "target_value_valid" in subset.columns:
        subset = subset.loc[subset["target_value_valid"].fillna(False).astype(bool)].copy()
    train = subset.loc[subset["split"].eq("train")].copy()
    validation = subset.loc[subset["split"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise ValueError(f"{task}: non-empty train and validation sets are required")
    selection, _, _, roles = split_validation_roles(validation, config["supervised"])
    if selection.empty:
        raise ValueError(f"{task}: no rows in validation model-selection window; role config={roles}")
    return train, validation, selection


def _source_preflight(phase3_report_dir: Path, phase3_model_dir: Path) -> dict[str, Any]:
    required = [
        phase3_report_dir / "churn_model_comparison.csv",
        phase3_report_dir / "value_model_comparison.csv",
        phase3_report_dir / "churn_cv_fold_metrics.csv",
        phase3_report_dir / "value_cv_fold_metrics.csv",
        phase3_report_dir / "churn_model_manifest.json",
        phase3_report_dir / "value_model_manifest.json",
        phase3_model_dir / "churn_model.joblib",
        phase3_model_dir / "value_model.joblib",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 4 depends on completed Phase 3 outputs. Missing:\n- "
            + "\n- ".join(missing)
            + "\nRun the dataset preparation pipeline before model comparison."
        )
    return {
        "status": "PASSED",
        "required_files": [str(path.relative_to(ROOT)) for path in required],
        "input_file_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in required
        },
    }


def _fold_summary(fold_file: Path, task: str) -> pd.DataFrame:
    folds = _read_csv_required(fold_file)
    if "model" not in folds.columns:
        raise ValueError(f"Expected a 'model' column in {fold_file}")
    if "status" in folds.columns:
        folds = folds.loc[folds["status"].astype(str).eq("PASSED")].copy()
    metric_cols = ["pr_auc", "roc_auc"] if task == "churn" else ["mae_gbp", "rmse_gbp"]
    rows: list[dict[str, Any]] = []
    for model_name, group in folds.groupby("model", dropna=False):
        row: dict[str, Any] = {"task": task, "model": str(model_name), "fold_rows": int(len(group))}
        for col in metric_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                row[f"fold_{col}_mean"] = float(values.mean()) if len(values) else None
                row[f"fold_{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) == 1 else None)
                row[f"fold_{col}_min"] = float(values.min()) if len(values) else None
                row[f"fold_{col}_max"] = float(values.max()) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _run_task_diagnostics(
    task: str,
    config: dict[str, Any],
    phase4_cfg: dict[str, Any],
    snapshots: pd.DataFrame,
    source_path: Path,
    phase3_report_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = config["supervised"]["churn_target" if task == "churn" else "value_target"]
    train, _validation, selection_val = _validation_partition(snapshots, target, task, config)
    numeric, categorical, excluded = feature_columns(snapshots)
    features = numeric + categorical
    if not features:
        raise ValueError(f"No approved {task} features found")
    if target in features or any(col.startswith("target_") for col in features):
        raise RuntimeError("Target leakage: target-derived fields entered feature list")

    if task == "churn":
        table_path = phase3_report_dir / "churn_model_comparison.csv"
        folds_path = phase3_report_dir / "churn_cv_fold_metrics.csv"
        manifest_path = phase3_report_dir / "churn_model_manifest.json"
    else:
        table_path = phase3_report_dir / "value_model_comparison.csv"
        folds_path = phase3_report_dir / "value_cv_fold_metrics.csv"
        manifest_path = phase3_report_dir / "value_model_manifest.json"
    table = _read_csv_required(table_path)
    if not {"model", "status"}.issubset(table.columns):
        raise ValueError(f"Unexpected columns in {table_path}")
    table["model"] = table["model"].astype(str)
    passed = table.loc[table["status"].astype(str).eq("PASSED")].copy()
    if passed.empty:
        raise RuntimeError(f"No successful Phase 3 {task} candidates to compare")
    manifest = _read_json(manifest_path)
    target_artifact_name = manifest.get("selected_model")
    current_bundle = joblib.load(project_path(config["paths"]["models_dir"]) / ("churn_model.joblib" if task == "churn" else "value_model.joblib"))
    if current_bundle.get("target") != target:
        raise RuntimeError(
            f"Phase 3 {task} artifact targets {current_bundle.get('target')}, expected {target}. "
            "Re-run Phase 3 before Phase 4."
        )
    if current_bundle.get("feature_names") != features:
        # Preserve the actual feature order stored by Phase 3 if the input schema
        # has not changed; otherwise force a rebuild rather than risk column drift.
        bundle_feature_names = current_bundle.get("feature_names", [])
        if set(bundle_feature_names) != set(features):
            raise RuntimeError(
                f"Phase 3 {task} artifact feature schema differs from current snapshots. "
                "Re-run Phase 3 before Phase 4."
            )
        features = bundle_feature_names

    train = train.copy()
    selection_val = selection_val.copy()
    fold_summary = _fold_summary(folds_path, task)
    row_predictions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    primary_capacity = float(config["supervised"].get("retention_capacity_fraction", 0.10))
    seed = int(config["project"].get("random_seed", phase4_cfg.get("random_seed", 42)))
    chosen_row = None
    if target_artifact_name:
        found = passed.loc[passed["model"].eq(str(target_artifact_name))]
        if not found.empty:
            chosen_row = found.iloc[0]

    for _, candidate in passed.iterrows():
        model_name = str(candidate["model"])
        params = parse_params(candidate.get("params"))
        if model_name == str(target_artifact_name) and isinstance(manifest.get("selected_params"), dict):
            # Respect the latest Phase 3 selected artifact parameters, which may
            # have come from a completed Optuna search and tuned-candidate refit.
            params = dict(manifest.get("selected_params") or {})
        # Each candidate sees the same training rows and the same model-selection
        # validation rows. Learned preprocessing is fitted only on the train set.
        X_train = train[features]
        X_val = selection_val[features]
        y_train = train[target]
        y_val = selection_val[target]
        result: dict[str, Any] = {
            "task": task,
            "model": model_name,
            "phase3_status": "PASSED",
            "is_phase3_artifact_candidate": model_name == str(target_artifact_name),
            "diagnostic_validation_role": "model_selection_window",
            "diagnostic_is_unbiased_final_estimate": False,
            "params": json.dumps(params, sort_keys=True, default=str),
            "train_rows": int(len(train)),
            "validation_rows": int(len(selection_val)),
        }
        try:
            model = build_pipeline(task, model_name, numeric, categorical, seed, params)
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if task == "churn" and y_train.nunique() < 2:
                    raise ValueError("Training partition contains only one churn class")
                model.fit(X_train, y_train)
            result["fit_ms"] = float((time.perf_counter() - started) * 1000.0)
            result["process_rss_mb_after_fit"] = process_rss_mb()

            if task == "churn":
                y_true = y_val.astype(int).to_numpy()
                score = _positive_probability(model, X_val)
                threshold = 0.5  # common comparison threshold; not a deployed policy
                metrics = comparison_classification_metrics(y_true, score, threshold, primary_capacity)
                prediction = (score >= threshold).astype(int)
                details = selection_val[["customer_id", "snapshot_date"]].copy()
                details["task"] = task
                details["model"] = model_name
                details["actual"] = y_true
                details["score"] = score
                details["threshold"] = threshold
                details["predicted"] = prediction
                details["error_type"] = np.select(
                    [(y_true == 1) & (prediction == 1), (y_true == 0) & (prediction == 1), (y_true == 1) & (prediction == 0)],
                    ["TP", "FP", "FN"], default="TN",
                )
                metrics["threshold_note"] = "Common raw-score threshold 0.5 for diagnostic comparability only; production threshold/calibration remain in the Phase 3 manifest."
                result.update({f"validation_{key}": value for key, value in metrics.items()})
                details["country"] = selection_val["country"].astype("string").to_numpy() if "country" in selection_val.columns else "__UNKNOWN__"
                if "recency_days" in selection_val.columns:
                    details["recency_tier"] = pd.cut(
                        pd.to_numeric(selection_val["recency_days"], errors="coerce"),
                        bins=[-np.inf, 30, 90, 180, np.inf],
                        labels=["0-30d", "31-90d", "91-180d", "181d+"],
                    ).astype("string").fillna("missing")
                row_predictions.append(details)
            else:
                y_true = y_val.astype(float).to_numpy()
                pred = np.asarray(model.predict(X_val), dtype=float)
                metrics = comparison_regression_metrics(y_true, pred)
                details = selection_val[["customer_id", "snapshot_date"]].copy()
                details["task"] = task
                details["model"] = model_name
                details["actual"] = y_true
                details["score"] = pred
                details["residual_actual_minus_predicted_gbp"] = y_true - pred
                details["absolute_error_gbp"] = np.abs(y_true - pred)
                # Value tiers use training outcomes, never validation or test targets.
                train_y = train[target].astype(float)
                med = float(train_y.quantile(0.50))
                q90 = float(train_y.quantile(0.90))
                details["training_defined_value_tier"] = np.select(
                    [details["actual"] <= med, details["actual"] <= q90],
                    ["low", "medium"], default="high",
                )
                # Target-based tier is for validation error analysis, not a feature.
                result.update({f"validation_{key}": value for key, value in metrics.items()})
                details["country"] = selection_val["country"].astype("string").to_numpy() if "country" in selection_val.columns else "__UNKNOWN__"
                if "recency_days" in selection_val.columns:
                    details["recency_tier"] = pd.cut(
                        pd.to_numeric(selection_val["recency_days"], errors="coerce"),
                        bins=[-np.inf, 30, 90, 180, np.inf],
                        labels=["0-30d", "31-90d", "91-180d", "181d+"],
                    ).astype("string").fillna("missing")
                row_predictions.append(details)

            latency = _latency_benchmark(
                model, X_val,
                task,
                int(phase4_cfg.get("latency_rows", 100)),
                int(phase4_cfg.get("batch_latency_rows", 512)),
            )
            serialization = _serialization_roundtrip(model, task, X_val)
            result.update(latency)
            result.update(serialization)
            latency_rows.append({"task": task, "model": model_name, **latency, **serialization,
                                 "fit_ms": result.get("fit_ms"),
                                 "process_rss_mb_after_fit": result.get("process_rss_mb_after_fit")})
            result["status"] = "PASSED"
        except Exception as exc:
            result["status"] = "FAILED"
            result["error"] = f"{type(exc).__name__}: {exc}"
            metric_rows.append(result)
            latency_rows.append({"task": task, "model": model_name, "latency_status": "NOT_RUN_OR_FAILED", "error": result["error"]})
            print(f"[{task}] {model_name}: ERROR {result['error']}")
            continue
        metric_rows.append(result)
        print(f"[{task}] {model_name}: validation diagnostic complete")

    if not metric_rows:
        raise RuntimeError(f"No Phase 4 diagnostic rows were generated for {task}")

    pred_frame = pd.concat(row_predictions, ignore_index=True) if row_predictions else pd.DataFrame()
    if not pred_frame.empty:
        # Use a top-error file to keep outputs practical while retaining row-level
        # diagnostic predictions in their own complete file.
        if task == "churn":
            pred_frame["error_priority"] = np.where(
                pred_frame["error_type"].eq("FN"), 2,
                np.where(pred_frame["error_type"].eq("FP"), 1, 0),
            )
            pred_frame = pred_frame.sort_values(
                ["error_priority", "score"], ascending=[False, False]
            )
        else:
            pred_frame = pred_frame.sort_values("absolute_error_gbp", ascending=False)

    summary = pd.DataFrame(metric_rows)
    folds = fold_summary.copy()
    if not folds.empty:
        summary = summary.merge(folds, on=["task", "model"], how="left")
    return summary, pred_frame, pd.DataFrame(latency_rows)


def _subgroup_reports(predictions: pd.DataFrame, task: str, min_rows: int) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    group_cols = [col for col in ("model", "country", "recency_tier", "training_defined_value_tier") if col in predictions.columns]
    rows: list[dict[str, Any]] = []
    for col in [c for c in ("country", "recency_tier", "training_defined_value_tier") if c in group_cols]:
        for (model, value), group in predictions.groupby(["model", col], dropna=False, observed=True):
            row = {"task": task, "model": model, "group_feature": col, "group_value": str(value), "n": int(len(group)), "min_group_rows_for_interpretation": int(min_rows)}
            if len(group) < min_rows:
                row["status"] = "TOO_SMALL_INTERPRET_CAUTIOUSLY"
                rows.append(row)
                continue
            if task == "churn":
                metric = comparison_classification_metrics(
                    group["actual"].astype(int).to_numpy(),
                    group["score"].astype(float).to_numpy(),
                    float(group["threshold"].iloc[0]),
                    0.10,
                )
                for key in ("positive_rate", "pr_auc_average_precision", "precision", "recall", "f1", "recall_at_capacity"):
                    row[key] = metric.get(key)
            else:
                metric = comparison_regression_metrics(group["actual"].astype(float).to_numpy(), group["score"].astype(float).to_numpy())
                for key in ("mae_gbp", "rmse_gbp", "r2", "mean_error_actual_minus_predicted_gbp"):
                    row[key] = metric.get(key)
            row["status"] = "REPORTED"
            rows.append(row)
    return pd.DataFrame(rows)


def _standardize_comparison(task: str, phase3_table: pd.DataFrame, phase4_summary: pd.DataFrame, current_manifest: dict[str, Any]) -> pd.DataFrame:
    # Only allow-listed fields are carried forward. Any test_* columns in a prior
    # Phase 3 table are intentionally excluded from Phase 4 selection reports.
    keep_common = [
        "model", "status", "params", "cv_folds", "training_fit_ms",
        "process_rss_mb_after_fit", "p50_ms", "p95_ms", "p99_ms",
        "batch_rows_per_second", "artifact_size_estimate_bytes",
    ]
    if task == "churn":
        keep_metrics = [
            "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
            "selection_pr_auc", "selection_roc_auc", "selection_recall_at_capacity",
            "selection_precision_among_contacted", "selection_brier_uncalibrated",
            "selection_ece_10_bins", "selected_champion",
            "champion_beats_dummy_on_selection_window",
        ]
        primary = "validation_pr_auc_average_precision"
        ascending = False
    else:
        keep_metrics = [
            "cv_mae_mean_gbp", "cv_mae_std_gbp", "cv_rmse_mean_gbp",
            "selection_mae_gbp", "selection_rmse_gbp", "selection_r2",
            "selection_top_decile_actual_value_gbp", "selection_top_decile_fraction_total_value",
            "selected_champion", "champion_beats_dummy_on_selection_window",
        ]
        primary = "validation_mae_gbp"
        ascending = True
    cols = [col for col in keep_common + keep_metrics if col in phase3_table.columns]
    out = phase3_table[cols].copy()
    out = out.rename(columns={
        "selected_champion": "phase3_initial_selected_champion",
        "champion_beats_dummy_on_selection_window": "phase3_initial_champion_beats_dummy",
    })
    if "model" not in out:
        raise ValueError(f"{task} Phase 3 comparison table lacks model column")
    out["model"] = out["model"].astype(str)
    out["task"] = task
    out["phase3_manifest_selected_model"] = current_manifest.get("selected_model")
    out["matches_current_phase3_artifact"] = out["model"].eq(str(current_manifest.get("selected_model")))

    diagnostic_cols = [
        "model", "status", "params", primary,
        "validation_pr_auc_average_precision", "validation_roc_auc",
        "validation_mae_gbp", "validation_rmse_gbp", "validation_r2",
        "validation_recall_at_capacity", "validation_precision_among_contacted_at_capacity",
        "p50_ms_per_row", "p95_ms_per_row", "p99_ms_per_row",
        "batch_rows_per_second", "serialized_size_bytes",
        "serialization_status", "roundtrip_max_absolute_difference",
        "fold_pr_auc_mean", "fold_pr_auc_std", "fold_mae_gbp_mean", "fold_mae_gbp_std",
    ]
    diagnostic_cols = list(dict.fromkeys(diagnostic_cols))
    diags = phase4_summary[[c for c in diagnostic_cols if c in phase4_summary.columns]].copy()
    if "status" in diags.columns:
        diags = diags.rename(columns={"status": "phase4_diagnostic_status"})
    out = out.merge(diags, on="model", how="left", suffixes=("", "_phase4"))
    if "params_phase4" in out.columns:
        out = out.rename(columns={"params_phase4": "diagnostic_refit_params"})
    out["primary_validation_metric"] = out[primary] if primary in out else np.nan
    out["primary_metric_direction"] = "higher_is_better" if task == "churn" else "lower_is_better"
    out["test_metrics_used_for_selection"] = False
    out["phase4_validation_role"] = "model_selection_window; diagnostic only and not an unbiased final estimate"
    metric_numeric = pd.to_numeric(out["primary_validation_metric"], errors="coerce")
    out["phase4_validation_rank"] = metric_numeric.rank(method="min", ascending=ascending).astype("Int64")

    baseline_name = "dummy" if task == "churn" else "dummy_median"
    baseline = out.loc[out["model"].eq(baseline_name), "primary_validation_metric"]
    baseline_value = float(baseline.iloc[0]) if len(baseline) and pd.notna(baseline.iloc[0]) else None
    out["dummy_validation_metric"] = baseline_value
    if baseline_value is None:
        out["beats_dummy_on_phase4_diagnostic"] = pd.NA
    elif task == "churn":
        out["beats_dummy_on_phase4_diagnostic"] = out["primary_validation_metric"] > baseline_value
    else:
        out["beats_dummy_on_phase4_diagnostic"] = out["primary_validation_metric"] < baseline_value
    return out.sort_values(["phase4_validation_rank", "model"], na_position="last").reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small markdown table without requiring the optional tabulate package."""
    if frame.empty:
        return "_No rows available._"
    columns = [str(c) for c in frame.columns]
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *body])


def _write_markdown_report(
    path: Path,
    comparisons: dict[str, pd.DataFrame],
    manifests: dict[str, dict[str, Any]],
    preflight: dict[str, Any],
) -> None:
    lines = [
        "# Phase 4 — Supervised Model Comparison",
        "",
        "## Connection to Phase 3",
        "This report consumes Phase 3 comparison tables, fold metrics, manifests, and saved candidate bundles. It re-fits candidate pipelines on the Phase 3 training partition only to create comparable validation error diagnostics and measured local latency. It does not use test rows for fitting, scoring, ranking, or selection.",
        "",
        "## Data lineage",
        f"- Phase 3 preflight: **{preflight.get('status')}**",
        f"- Generated at UTC: {datetime.now(timezone.utc).isoformat()}",
        "- Candidate ranking uses the validation model-selection window and is therefore a development diagnostic, not an unbiased final generalization estimate.",
        "- Phase 3 test metrics, if already present in older comparison files, are deliberately excluded from this report.",
        "",
    ]
    for task in ("churn", "value"):
        table = comparisons.get(task, pd.DataFrame())
        manifest = manifests.get(task, {})
        lines.extend([f"## {task.title()}", ""])
        lines.append(f"- Existing Phase 3 artifact candidate: `{manifest.get('selected_model')}`")
        lines.append(f"- Phase 3 candidate status: `{manifest.get('status')}`")
        if table.empty:
            lines.extend(["No comparison rows were produced.", ""])
            continue
        candidate = table.loc[table["phase4_validation_rank"].eq(1)]
        if candidate.empty:
            lines.extend(["No valid Phase 4 validation rank is available.", ""])
        else:
            row = candidate.iloc[0]
            primary = "primary_validation_metric"
            lines.append(f"- Best Phase 4 validation diagnostic: `{row['model']}` ({row[primary]})")
            lines.append(f"- Beats the dummy on this validation diagnostic: `{row.get('beats_dummy_on_phase4_diagnostic')}`")
        if task == "churn":
            lines.extend([
                "- Primary measure: Average Precision / PR-AUC. Also inspect precision, recall, F1, recall at the configured selection capacity, and validation subgroup sizes.",
                "- The common 0.5 threshold in Phase 4 candidate diagnostics is for like-for-like error analysis only. The frozen calibrated policy stays in the Phase 3 model manifest.",
            ])
        else:
            lines.extend([
                "- Primary measure: MAE in GBP. Also inspect RMSE, R², mean signed error, value-tier error, and top-decile allocation diagnostic.",
                "- The target is net recorded line value, not profit or contribution margin.",
            ])
        lines.extend(["", "Top validation diagnostic rows:", ""])
        show_cols = [
            "phase4_validation_rank", "model", "phase3_status", "primary_validation_metric",
            "cv_folds", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_mae_mean_gbp", "cv_mae_std_gbp",
            "p50_ms_per_row", "p95_ms_per_row", "batch_rows_per_second",
            "beats_dummy_on_phase4_diagnostic", "matches_current_phase3_artifact",
        ]
        show_cols = [col for col in show_cols if col in table.columns]
        lines.append(_markdown_table(table[show_cols].head(10)))
        lines.append("")
    lines.extend([
        "## Required review before approval",
        "1. Confirm every required model family has a successful Phase 3 row or a documented exclusion.",
        "2. Review fold means and standard deviations; do not interpret small metric differences beyond fold variability as conclusive.",
        "3. Review false positives and false negatives from the model-selection validation window. This set was already involved in model selection, so the diagnostics are not unbiased final estimates.",
        "4. Compare p50/p95/p99 latency, batch throughput, fit time, memory, and serialized size against a business-owned SLA before production selection.",
        "5. Keep test data untouched until the model, features, tuning, threshold, and calibration are frozen.",
        "",
        "## Limitations",
        "- Churn is a 30-day inactivity proxy, not verified cancellation.",
        "- Customer value is 90-day net transaction-line value in GBP.",
        "- No retention experiment or investigator-confirmed anomaly labels are present in Online Retail II.",
        "- Local latency figures are machine/environment-specific and are not a production load test.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model_comparison() -> dict[str, Any]:
    config = load_config()
    phase4_cfg = load_phase4_config()
    report_dir = project_path(config["paths"]["reports_dir"])
    model_dir = project_path(config["paths"]["models_dir"])
    output_dir = ROOT / phase4_cfg.get("output_dir", "reports/phase4")
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = _source_preflight(report_dir, model_dir)
    snapshots, source_path = load_snapshots()
    if snapshots["snapshot_date"].isna().any():
        raise ValueError("snapshot_date contains invalid timestamps")
    if snapshots.duplicated(["customer_id", "snapshot_date"]).any():
        raise ValueError("Duplicate customer/snapshot key in Phase 3 snapshots")
    if set(snapshots["split"].dropna().astype(str)) - {"train", "validation", "test", "purged_gap"}:
        raise ValueError("Unexpected split label found in Phase 3 snapshots")

    # Guard against data/schema changing after Phase 3. The actual candidate
    # comparisons are reported against the exact source hash recorded in each
    # Phase 3 manifest.
    manifests = {
        task: _read_json(report_dir / f"{task}_model_manifest.json")
        for task in ("churn", "value")
    }
    current_hash = sha256_file(source_path)
    for task, manifest in manifests.items():
        if manifest.get("data_sha256") and manifest["data_sha256"] != current_hash:
            raise RuntimeError(
                f"Phase 3 {task} manifest data hash does not match current snapshots. "
                "Do not compare different data versions; rerun Phase 3 first."
            )
        if manifest.get("test_evaluated"):
            preflight.setdefault("warnings", []).append(
                f"Phase 3 {task} test was previously evaluated. Phase 4 will not read or use those test metrics."
            )

    all_comparison_rows: list[pd.DataFrame] = []
    all_latency_rows: list[pd.DataFrame] = []
    all_subgroup_rows: list[pd.DataFrame] = []
    output_prediction_paths: dict[str, str] = {}
    diagnostic_summaries: dict[str, pd.DataFrame] = {}
    comparisons: dict[str, pd.DataFrame] = {}

    for task in ("churn", "value"):
        summary, prediction_rows, latency = _run_task_diagnostics(
            task, config, phase4_cfg, snapshots, source_path, report_dir
        )
        diagnostic_summaries[task] = summary
        phase3_table = _read_csv_required(report_dir / f"{task}_model_comparison.csv")
        comparison = _standardize_comparison(task, phase3_table, summary, manifests[task])
        comparisons[task] = comparison
        all_comparison_rows.append(comparison)
        all_latency_rows.append(latency)

        metrics_path = output_dir / f"{task}_validation_error_summary.csv"
        summary.to_csv(metrics_path, index=False)
        prediction_path = output_dir / f"{task}_validation_error_rows.csv"
        prediction_rows.to_csv(prediction_path, index=False)
        top_n = int(phase4_cfg.get("top_error_rows_per_model", 250))
        if not prediction_rows.empty:
            if task == "churn":
                false_rows = prediction_rows.loc[prediction_rows["error_type"].isin(["FP", "FN"])].copy()
                top_errors = false_rows.groupby("model", group_keys=False).head(top_n)
            else:
                top_errors = prediction_rows.groupby("model", group_keys=False).head(top_n)
        else:
            top_errors = prediction_rows.copy()
        top_errors_path = output_dir / f"{task}_validation_top_errors.csv"
        top_errors.to_csv(top_errors_path, index=False)
        output_prediction_paths[task] = str(prediction_path.relative_to(ROOT))
        output_prediction_paths[f"{task}_top_errors"] = str(top_errors_path.relative_to(ROOT))
        subgroups = _subgroup_reports(
            prediction_rows, task, int(phase4_cfg.get("subgroup_min_rows", 25))
        )
        subgroup_path = output_dir / f"{task}_validation_subgroup_metrics.csv"
        subgroups.to_csv(subgroup_path, index=False)
        all_subgroup_rows.append(subgroups)
        print(f"[{task}] generated comparison for {len(comparison)} candidate(s)")

    combined = pd.concat(all_comparison_rows, ignore_index=True, sort=False)
    combined_path = output_dir / "supervised_model_comparison.csv"
    combined.to_csv(combined_path, index=False)
    latency_table = pd.concat(all_latency_rows, ignore_index=True, sort=False)
    latency_path = output_dir / "latency_comparison.csv"
    latency_table.to_csv(latency_path, index=False)
    subgroup_table = pd.concat([x for x in all_subgroup_rows if not x.empty], ignore_index=True, sort=False) if any(not x.empty for x in all_subgroup_rows) else pd.DataFrame()
    subgroup_table_path = output_dir / "subgroup_error_analysis.csv"
    subgroup_table.to_csv(subgroup_table_path, index=False)

    # Use Phase 3's selected artifacts for a serialization/load checksum audit.
    artifact_checks = {}
    for task in ("churn", "value"):
        artifact_path = model_dir / ("churn_model.joblib" if task == "churn" else "value_model.joblib")
        manifest = manifests[task]
        artifact_sha = sha256_file(artifact_path)
        manifest_sha = manifest.get("artifact_sha256")
        if manifest_sha and artifact_sha != manifest_sha:
            raise RuntimeError(f"{task} artifact checksum differs from its Phase 3 manifest")
        artifact_checks[task] = {
            "artifact_path": str(artifact_path.relative_to(ROOT)),
            "artifact_sha256": artifact_sha,
            "matches_phase3_manifest": artifact_sha == manifest_sha if manifest_sha else None,
            "model_version": manifest.get("model_version"),
            "selected_model": manifest.get("selected_model"),
            "phase3_status": manifest.get("status"),
        }

    markdown_path = output_dir / "phase4_model_comparison_report.md"
    _write_markdown_report(markdown_path, comparisons, manifests, preflight)
    fold_paths = {
        task: str((report_dir / f"{task}_cv_fold_metrics.csv").relative_to(ROOT))
        for task in ("churn", "value")
    }
    manifest = {
        "phase": 4,
        "title": "Supervised model comparison, error analysis and latency comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase3_input_preflight": preflight,
        "phase3_snapshot_path": str(source_path.relative_to(ROOT)),
        "phase3_snapshot_sha256": current_hash,
        "phase3_cv_fold_inputs": fold_paths,
        "selected_phase3_artifacts": artifact_checks,
        "train_and_validation_only": True,
        "test_split_used_for_fitting_or_scoring": False,
        "test_metrics_used_for_selection": False,
        "diagnostic_validation_role": "model_selection_window; development diagnostic only, not unbiased final estimate",
        "candidate_fit_rows_split": "train",
        "candidate_scored_rows_split": "validation model_selection window",
        "outputs": {
            "comparison": str(combined_path.relative_to(ROOT)),
            "latency": str(latency_path.relative_to(ROOT)),
            "subgroups": str(subgroup_table_path.relative_to(ROOT)),
            "report": str(markdown_path.relative_to(ROOT)),
            "validation_errors": output_prediction_paths,
        },
        "notes": [
            "Phase 4 connects to Phase 3: it consumes the Phase 3 comparison tables, CV fold tables, manifests, snapshots, and saved model artifacts.",
            "Phase 3 comparisons are preserved. Phase 4 re-fits candidates on train rows only to produce comparable row-level validation diagnostics and local latency measures.",
            "Phase 4 report deliberately excludes test metrics even if the Phase 3 comparison table has been amended by a past test run.",
            "Current Phase 3 selected-model artifacts and checksums are verified; no artifact is approved automatically by Phase 4.",
            "Local latency is a diagnostic benchmark, not a production load test.",
        ],
    }
    manifest_path = output_dir / "phase4_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Comparison: {combined_path}")
    print(f"Latency comparison: {latency_path}")
    print(f"Subgroup analysis: {subgroup_table_path}")
    print(f"Report: {markdown_path}")
    print(f"Manifest: {manifest_path}")
    print("Phase 4 finished. Test rows were not used for fitting, scoring, or selection.")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_model_comparison()


# --- Consolidated from phase_5_search_spaces.py ---

"""Expanded, reproducible Optuna search spaces for all Phase 3 estimators.

This module intentionally tunes only hyperparameters supported by the project's
existing build_estimator/build_pipeline implementation. It exposes a resolver
for composite voting and polynomial pipelines so the stored best parameters are
usable directly by Phase 5 refitting.
"""

from typing import Any

import optuna


WEIGHT_PROFILES: dict[str, list[float]] = {
    "equal": [1.0, 1.0, 1.0],
    "linear_emphasis": [2.0, 1.0, 1.0],
    "tree_emphasis": [1.0, 2.0, 1.0],
    "boost_emphasis": [1.0, 1.0, 2.0],
    "linear_strong": [3.0, 1.0, 1.0],
    "tree_strong": [1.0, 3.0, 1.0],
    "boost_strong": [1.0, 1.0, 3.0],
}


def suggest_params(trial: optuna.Trial, task: str, model: str) -> dict[str, Any]:
    """Suggest parameters for all supported tunable classifier/regressor models."""
    # Linear / generalized linear models.
    if model == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    if model == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e4, log=True),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }
    if model == "lasso":
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e2, log=True),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
        }
    if model == "elastic_net":
        return {
            "alpha": trial.suggest_float("alpha", 1e-6, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
        }
    if model == "polynomial_regression":
        # Phase 5 applies this degree to the fitted PolynomialFeatures step.
        return {"_polynomial_degree": trial.suggest_int("polynomial_degree", 2, 3)}

    # Single decision trees.
    if model == "decision_tree":
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    if model == "decision_tree_regressor":
        return {
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 18, 25, 32]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.5, 0.8]),
            "criterion": trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error"]),
        }

    # Bagged tree ensembles.
    if model in {"random_forest", "extra_trees"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 8, 12, 18, 24, 32]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.35, 0.5, 0.65, 0.8, 1.0]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None])
                if model == "random_forest"
                else trial.suggest_categorical("class_weight", ["balanced", None]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if model in {"random_forest_regressor", "extra_trees_regressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 8, 12, 18, 24, 32]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.35, 0.5, 0.65, 0.8, 1.0]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

    # Boosting estimators from scikit-learn.
    if model in {"adaboost", "adaboost_regressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        }
    if model in {"gradient_boosting", "gradient_boosting_regressor"}:
        params: dict[str, Any] = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 1, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.25, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 80),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2", 0.5, 0.8]),
        }
        return params
    if model in {"hist_gradient_boosting", "hist_gradient_boosting_regressor"}:
        return {
            "max_iter": trial.suggest_int("max_iter", 100, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.25, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 127),
            "max_depth": trial.suggest_categorical("max_depth", [None, 3, 5, 8, 12, 16]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 100.0, log=True),
        }

    # Margin- and distance-based classifiers.
    if model == "svm":
        return {
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    if model == "svm_rbf":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
    if model == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 101, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
            "leaf_size": trial.suggest_int("leaf_size", 15, 80),
        }
    if model == "naive_bayes":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}

    # Gradient boosting libraries. Parameters deliberately cover tree complexity,
    # shrinkage, row/column sampling, leaf support and regularization.
    if model in {"xgboost", "xgboost_regressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 30.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1000.0, log=True),
        }
    if model in {"lightgbm", "lightgbm_regressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
            "num_leaves": trial.suggest_int("num_leaves", 7, 255),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 5, 7, 10, 15, 20]),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.25, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1000.0, log=True),
        }
    if model in {"catboost", "catboost_regressor"}:
        return {
            "iterations": trial.suggest_int("iterations", 200, 1500, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.25, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        }

    # Voting ensembles already contain three estimator families. The search tunes
    # only their contribution weights, without turning this into nested stacking.
    if model in {"voting", "voting_regressor"}:
        return {
            "_phase5_weight_profile": trial.suggest_categorical(
                "weight_profile", list(WEIGHT_PROFILES)
            )
        }

    raise ValueError(f"Expanded Optuna search space not defined for {task}/{model}")


def resolve_params(task: str, model: str, proposed: dict[str, Any]) -> dict[str, Any]:
    """Convert search-control values to estimator constructor parameters."""
    params = dict(proposed)
    if model in {"voting", "voting_regressor"}:
        profile = params.pop("_phase5_weight_profile", params.pop("weight_profile", "equal"))
        params["weights"] = list(WEIGHT_PROFILES[str(profile)])
    # `_polynomial_degree` is consumed by build_estimator to flag the polynomial
    # pipeline; Phase 5 additionally applies it to PolynomialFeatures itself.
    return params


# --- Consolidated from phase_5_tuning_explanations.py ---

"""Phase 5: bounded tuning, calibration, threshold policy, explanations, and model cards.

Consumes Phase 3 snapshots/models and Phase 4 comparison reports. Writes distinct
Phase 5 artifacts and never uses the test split for fitting, tuning, calibration,
threshold selection, explanation, or model selection.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    brier_score_loss,
)

from src.training.common import (
    ROOT,
    build_pipeline,
    feature_columns,
    json_dump,
    load_config as load_phase3_config,
    load_snapshots,
    make_run_id,
    persist_split_membership,
    project_path,
    purged_walk_forward_folds,
    split_validation_roles,
    PlattCalibrator,
    select_threshold,
    classification_metrics,
    regression_metrics,
    process_rss_mb,
    log_environment,
    sha256_file,
)
from src.training.train_supervised import _score_probability

CONFIG_PATH = ROOT / "configs" / "project.yaml"


TUNABLE_MODELS = {
    "churn": {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "adaboost",
        "gradient_boosting",
        "hist_gradient_boosting",
        "svm",
        "svm_rbf",
        "knn",
        "naive_bayes",
        "voting",
        "xgboost",
        "lightgbm",
        "catboost",
    },
    "value": {
        "ridge",
        "lasso",
        "elastic_net",
        "polynomial_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "adaboost_regressor",
        "gradient_boosting_regressor",
        "hist_gradient_boosting_regressor",
        "voting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
        "catboost_regressor",
    },
}

BASELINE_NAMES = {
    "churn": "dummy",
    "value": "dummy_median",
}

BASELINE_NAMES = {"churn": "dummy", "value": "dummy_median"}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_phase5_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing Phase 5 config: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, value: Any) -> None:
    json_dump(path, value)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _parse_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _numeric_target(frame: pd.DataFrame, target: str, task: str) -> pd.DataFrame:
    out = frame.loc[frame[target].notna()].copy()
    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out.loc[np.isfinite(out[target].to_numpy(dtype=float))].copy()
    if task == "churn":
        out = out.loc[out[target].isin([0, 1])].copy()
        out[target] = out[target].astype(int)
    elif "target_value_valid" in out.columns:
        # This target-quality column is metadata and must never be a feature.
        out = out.loc[out["target_value_valid"].fillna(False).astype(bool)].copy()
    return out.reset_index(drop=True)


def _phase4_preflight(cfg3: dict[str, Any], report_dir: Path, model_dir: Path) -> dict[str, Any]:
    """Validate lineage before Phase 5 reads Phase 3/4 decisions."""
    phase4_dir = project_path("reports/phase4")
    phase4_manifest_path = phase4_dir / "phase4_manifest.json"
    comparison_path = phase4_dir / "supervised_model_comparison.csv"
    snapshot_path = project_path(cfg3["paths"]["customer_snapshots"])
    required = [
        phase4_manifest_path,
        comparison_path,
        report_dir / "churn_model_manifest.json",
        report_dir / "value_model_manifest.json",
        report_dir / "churn_cv_fold_metrics.csv",
        report_dir / "value_cv_fold_metrics.csv",
        model_dir / "churn_model.joblib",
        model_dir / "value_model.joblib",
        snapshot_path,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Phase 5 requires completed Phase 3 and Phase 4 outputs. Missing:\n- "
            + "\n- ".join(missing)
            + "\nRun Phase 3 and Phase 4 first."
        )

    p4_manifest = read_json(phase4_manifest_path)
    current_snapshot_hash = sha256_file(snapshot_path)
    p4_snapshot_hash = p4_manifest.get("phase3_snapshot_sha256")
    if p4_snapshot_hash and current_snapshot_hash != p4_snapshot_hash:
        raise RuntimeError(
            "Phase 4 was generated from a different Phase 3 snapshot. "
            "Re-run Phase 4 after verifying Phase 3 artifacts."
        )

    phase3_hashes: dict[str, str] = {}
    for task in ("churn", "value"):
        manifest = read_json(report_dir / f"{task}_model_manifest.json")
        artifact_path = project_path(manifest.get("artifact_path", model_dir / f"{task}_model.joblib"))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Phase 3 model artifact missing: {artifact_path}")
        actual = sha256_file(artifact_path)
        expected = manifest.get("artifact_sha256")
        if expected and actual != expected:
            raise RuntimeError(
                f"Phase 3 {task} artifact checksum differs from its manifest. "
                "Re-run Phase 3 and Phase 4 rather than mixing versions."
            )
        if manifest.get("data_sha256") and manifest["data_sha256"] != current_snapshot_hash:
            raise RuntimeError(f"Phase 3 {task} manifest does not match the current snapshot data.")
        phase3_hashes[task] = actual

    return {
        "status": "PASSED",
        "phase3_snapshot_path": str(snapshot_path.relative_to(ROOT)),
        "phase3_snapshot_sha256": current_snapshot_hash,
        "phase4_manifest_sha256": sha256_file(phase4_manifest_path),
        "phase4_comparison_sha256": sha256_file(comparison_path),
        "phase3_artifact_sha256": phase3_hashes,
        "test_split_used_for_fit_tuning_calibration_explanation_or_selection": False,
        "shared_snapshot_loader_reads_complete_file_for_schema_and_split_integrity_checks": True,
    }


def _load_comparison() -> pd.DataFrame:
    path = project_path("reports/phase4/supervised_model_comparison.csv")
    frame = pd.read_csv(path, low_memory=False)
    needed = {"task", "model", "status"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Phase 4 comparison is missing columns: {sorted(needed - set(frame.columns))}")
    return frame


def _ranked_phase4_candidates(comparison: pd.DataFrame, task: str) -> pd.DataFrame:
    table = comparison.loc[comparison["task"].astype(str).eq(task)].copy()
    if table.empty:
        raise ValueError(f"Phase 4 comparison has no rows for task={task}")
    # Phase 4's status is the diagnostic refit status where available.
    status_col = "phase4_diagnostic_status" if "phase4_diagnostic_status" in table else "status"
    table = table.loc[table[status_col].astype(str).eq("PASSED")].copy()
    if "phase4_validation_rank" in table:
        table["phase4_validation_rank"] = pd.to_numeric(table["phase4_validation_rank"], errors="coerce")
        table = table.sort_values(["phase4_validation_rank", "model"], na_position="last")
    elif "primary_validation_metric" in table:
        metric = pd.to_numeric(table["primary_validation_metric"], errors="coerce")
        direction = table.get("primary_metric_direction", pd.Series("higher_is_better", index=table.index))
        table["_metric"] = metric
        table["_asc"] = direction.astype(str).eq("lower_is_better")
        # Sort by task's metric direction; use known task convention as fallback.
        table = table.sort_values("_metric", ascending=(task == "value"), na_position="last")
    else:
        raise ValueError("Phase 4 comparison lacks ranking metrics")
    return table.reset_index(drop=True)


def _target_for(task: str, cfg3: dict[str, Any]) -> str:
    return cfg3["supervised"]["churn_target"] if task == "churn" else cfg3["supervised"]["value_target"]


def _horizon_for(task: str, cfg3: dict[str, Any]) -> int:
    key = "churn_horizon_days" if task == "churn" else "value_horizon_days"
    return int(cfg3["supervised"].get(key, cfg3["supervised"]["horizon_days"]))


def _metric_name(task: str) -> str:
    return "pr_auc_average_precision" if task == "churn" else "mae_gbp"


def _score_model(task: str, model: Any, X: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
    if task == "churn":
        prob = _score_probability(model, X)
        metrics: dict[str, Any] = {
            "selection_pr_auc": float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else None,
            "selection_brier_uncalibrated": float(brier_score_loss(y, prob)),
            "prediction_mean": float(np.mean(prob)),
        }
    else:
        pred = np.asarray(model.predict(X), dtype=float)
        metrics = {
            "selection_mae_gbp": float(mean_absolute_error(y, pred)),
            "selection_rmse_gbp": float(np.sqrt(mean_squared_error(y, pred))),
            "selection_r2": float(r2_score(y, pred)) if len(y) > 1 and np.var(y) > 0 else None,
            "prediction_mean_gbp": float(np.mean(pred)),
        }
    return metrics


def _suggest_and_tune(
    task: str,
    model_name: str,
    train: pd.DataFrame,
    feature_names: list[str],
    numeric: list[str],
    categorical: list[str],
    cfg3: dict[str, Any],
    cfg5: dict[str, Any],
    data_hash: str,
    report_dir: Path,
    optuna_dir: Path,
    trials_budget: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    import optuna

    target = _target_for(task, cfg3)
    horizon = _horizon_for(task, cfg3)
    folds = purged_walk_forward_folds(
        train, horizon_days=horizon,
        max_folds=int(cfg3["supervised"].get("cv_max_folds", 3)),
    )
    if not folds:
        raise RuntimeError(f"No purged chronological CV folds for {task}/{model_name}")

    optuna_dir.mkdir(parents=True, exist_ok=True)
    storage_path = optuna_dir / "phase5_studies.sqlite3"
    study_name = f"phase5_{task}_{model_name}_{data_hash[:12]}"
    storage_uri = f"sqlite:///{storage_path.as_posix()}"
    direction = "maximize" if task == "churn" else "minimize"
    sampler_seed = int(cfg5["tuning"]["optuna"].get("seed", 42))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        load_if_exists=True,
    )

    X = train[feature_names]
    y = train[target].to_numpy(dtype=int if task == "churn" else float)
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, task, model_name)
        scores: list[float] = []
        fold_records: list[dict[str, Any]] = []
        for fold_idx, (tr_idx, va_idx, train_end, val_date) in enumerate(folds, start=1):
            y_fit = y[tr_idx]
            y_val = y[va_idx]
            if task == "churn" and len(np.unique(y_fit)) < 2:
                trial.set_user_attr(f"fold_{fold_idx}_status", "skipped_single_class_train")
                continue
            if task == "churn" and len(np.unique(y_val)) < 2:
                trial.set_user_attr(f"fold_{fold_idx}_status", "skipped_single_class_validation")
                continue
            try:
                pipeline = build_pipeline(
                    task, model_name, numeric, categorical,
                    int(cfg3["project"]["random_seed"]), params,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipeline.fit(X.iloc[tr_idx], y_fit)
                if task == "churn":
                    score = float(average_precision_score(y_val, _score_probability(pipeline, X.iloc[va_idx])))
                else:
                    score = float(mean_absolute_error(y_val, pipeline.predict(X.iloc[va_idx])))
                scores.append(score)
                fold_records.append({
                    "fold": fold_idx,
                    "score": score,
                    "train_end_snapshot": train_end,
                    "validation_snapshot": val_date,
                })
                trial.set_user_attr(f"fold_{fold_idx}_score", score)
            except Exception as exc:
                trial.set_user_attr(f"fold_{fold_idx}_error", f"{type(exc).__name__}: {exc}")
        if not scores:
            raise optuna.TrialPruned("No valid purged CV fold completed")
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("fold_mean", float(np.mean(scores)))
        trial.set_user_attr("fold_std", float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
        trial.set_user_attr("fold_records", fold_records)
        return float(np.mean(scores))

    # Treat --trials as a persistent study-wide budget, not extra trials per rerun.
    completed_or_running = len(study.trials)
    remaining = max(0, int(trials_budget) - completed_or_running)
    if remaining > 0:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=int(timeout_seconds),
            gc_after_trial=True,
            show_progress_bar=True,
        )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"No completed Optuna trials for {task}/{model_name}; inspect {storage_path}")
    best = study.best_trial
    best_obj = {
        "task": task,
        "model": model_name,
        "study_name": study_name,
        "study_storage": str(storage_path.relative_to(ROOT)),
        "direction": direction,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_trial_number": int(best.number),
        "best_trial_fold_scores": best.user_attrs.get("fold_scores", []),
        "best_trial_fold_std": best.user_attrs.get("fold_std"),
        "completed_trials": len(completed),
        "total_study_trials": len(study.trials),
        "budget_trials": int(trials_budget),
        "input_sha256": data_hash,
        "test_split_used_for_cv_or_objective": False,
        "created_at_utc": now_utc(),
    }
    safe_name = f"optuna_best_{task}_{model_name}.json"
    write_json(report_dir / safe_name, best_obj)
    study.trials_dataframe(attrs=("number", "value", "datetime_start", "datetime_complete", "duration", "params", "user_attrs", "state")).to_csv(
        report_dir / f"optuna_trials_{task}_{model_name}.csv", index=False
    )
    return best_obj


def _fit_pipeline(
    task: str,
    model_name: str,
    params: dict[str, Any],
    train: pd.DataFrame,
    target: str,
    feature_names: list[str],
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> Any:
    pipeline = build_pipeline(task, model_name, numeric, categorical, seed, params)
    y = train[target].astype(int if task == "churn" else float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(train[feature_names], y)
    return pipeline


def _time_roles(validation: pd.DataFrame, cfg3: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selection, calibration, threshold, roles = split_validation_roles(validation, cfg3["supervised"])
    if selection.empty:
        raise ValueError("Phase 5 has no model-selection validation rows")
    if threshold.empty:
        threshold = selection.copy()
        roles["threshold_fallback_to_selection"] = True
        roles["threshold_warning"] = "Threshold window unavailable; selection window used as fallback. Threshold is not independent."
    else:
        roles["threshold_fallback_to_selection"] = False
    return selection, calibration, threshold, roles


def _calibration_and_threshold(
    model: Any,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selection: pd.DataFrame,
    calibration: pd.DataFrame,
    threshold_rows: pd.DataFrame,
    roles: dict[str, Any],
    cfg3: dict[str, Any],
    cfg5: dict[str, Any],
    target: str,
    feature_names: list[str],
    report_dir: Path,
) -> tuple[Any | None, float | None, dict[str, Any]]:
    policy: dict[str, Any] = {
        "validation_roles": roles,
        "calibration_method": "none_for_regression",
        "threshold_objective": cfg5["tuning"]["threshold_policy"].get("objective", "f1"),
        "test_used": False,
    }
    if task_is_value := target == cfg3["supervised"]["value_target"]:
        policy["threshold"] = None
        policy["status"] = "not_applicable_regression"
        return None, None, policy

    y_cal = calibration[target].astype(int).to_numpy() if not calibration.empty else np.array([], dtype=int)
    if len(calibration) and len(np.unique(y_cal)) > 1:
        raw_cal = _score_probability(model, calibration[feature_names])
        calibrator = PlattCalibrator().fit(raw_cal, y_cal)
        policy["calibration_method"] = "Platt sigmoid" if calibrator.fitted else "identity fallback"
        policy["calibration_status"] = "fitted" if calibrator.fitted else "fallback"
        policy["calibration_reason"] = calibrator.reason
        policy["calibration_rows"] = int(len(calibration))
        policy["calibration_dates"] = sorted(calibration.snapshot_date.dt.date.astype(str).unique().tolist())
    else:
        calibrator = PlattCalibrator()
        calibrator.reason = "Calibration window empty or contains one class; identity calibration used."
        policy["calibration_method"] = "identity fallback"
        policy["calibration_status"] = "fallback"
        policy["calibration_reason"] = calibrator.reason
        policy["calibration_rows"] = int(len(calibration))
        policy["calibration_dates"] = sorted(calibration.snapshot_date.dt.date.astype(str).unique().tolist()) if len(calibration) else []

    y_thr = threshold_rows[target].astype(int).to_numpy()
    raw_thr = _score_probability(model, threshold_rows[feature_names])
    cal_thr = calibrator.predict(raw_thr)
    threshold, details = select_threshold(y_thr, cal_thr)
    policy.update({
        "threshold": float(threshold),
        "threshold_details": details,
        "threshold_rows": int(len(threshold_rows)),
        "threshold_dates": sorted(threshold_rows.snapshot_date.dt.date.astype(str).unique().tolist()),
        "capacity_fraction": float(cfg5["tuning"]["threshold_policy"].get("capacity_fraction", cfg3["supervised"].get("retention_capacity_fraction", 0.1))),
        "raw_brier_threshold_window": float(brier_score_loss(y_thr, raw_thr)) if len(y_thr) else None,
        "calibrated_brier_threshold_window": float(brier_score_loss(y_thr, cal_thr)) if len(y_thr) else None,
        "threshold_window_calibration_is_independent": False,
        "threshold_selection_note": "Threshold is chosen on a reserved validation time window; this is still development data, not the final test estimate.",
    })

    # Reliability plot is a validation diagnostic, never a test plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        if len(np.unique(y_thr)) > 1:
            fig, ax = plt.subplots(figsize=(7, 6))
            for probs, label in ((raw_thr, "Raw"), (cal_thr, "Calibrated")):
                frac_pos, mean_pred = calibration_curve(y_thr, probs, n_bins=10, strategy="quantile")
                ax.plot(mean_pred, frac_pos, marker="o", label=label)
            ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed positive fraction")
            ax.set_title("Phase 5 churn reliability — validation window")
            ax.legend()
            fig.tight_layout()
            fig.savefig(report_dir / "churn_calibration_reliability.png", dpi=160)
            plt.close(fig)
            policy["reliability_plot"] = "churn_calibration_reliability.png"
        else:
            policy["reliability_plot_status"] = "skipped: threshold window contains only one class"
    except Exception as exc:
        policy["reliability_plot_status"] = f"unavailable: {type(exc).__name__}: {exc}"

    return calibrator, float(threshold), policy


def _permutation_and_coefficients(
    task: str,
    bundle: dict[str, Any],
    explain_frame: pd.DataFrame,
    target: str,
    report_dir: Path,
    cfg5: dict[str, Any],
) -> dict[str, Any]:
    explain_started = time.perf_counter()
    result: dict[str, Any] = {"status": "started", "test_used": False}
    feature_names = bundle["feature_names"]
    explain_rows = int(cfg5["tuning"]["explainability"].get("permutation_rows", 500))
    repeat_count = int(cfg5["tuning"]["explainability"].get("permutation_repeats", 3))
    explain_frame = explain_frame.loc[explain_frame[target].notna()].copy()
    if task == "value" and "target_value_valid" in explain_frame:
        explain_frame = explain_frame.loc[explain_frame["target_value_valid"].fillna(False).astype(bool)]
    if explain_frame.empty:
        return {"status": "skipped", "reason": "No validation rows for explanation", "test_used": False}
    explain_frame = explain_frame.sample(
        n=min(explain_rows, len(explain_frame)),
        random_state=int(cfg5["tuning"].get("seed", 42)),
    )
    X = explain_frame[feature_names]
    y = explain_frame[target].astype(int if task == "churn" else float).to_numpy()
    pipeline = bundle["model"]
    scorer = "average_precision" if task == "churn" else "neg_mean_absolute_error"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perm = permutation_importance(
                pipeline, X, y,
                scoring=scorer,
                n_repeats=repeat_count,
                random_state=int(cfg5["tuning"].get("seed", 42)),
                n_jobs=1,
            )
        importance = pd.DataFrame({
            "feature": feature_names,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
            "task": task,
            "model_version": bundle["model_version"],
            "scoring": scorer,
            "rows": len(X),
        }).sort_values("permutation_importance_mean", ascending=False)
        importance_path = report_dir / f"{task}_permutation_importance.csv"
        importance.to_csv(importance_path, index=False)
        result["permutation_importance"] = str(importance_path.relative_to(ROOT))
        result["permutation_rows"] = int(len(X))
    except Exception as exc:
        importance = pd.DataFrame(columns=["feature", "permutation_importance_mean", "permutation_importance_std"])
        result["permutation_status"] = f"failed: {type(exc).__name__}: {exc}"

    # Coefficients are model-specific. Report transformed names and state scale.
    try:
        pipeline_steps = pipeline.named_steps
        estimator = pipeline_steps["model"]
        if hasattr(estimator, "coef_"):
            names = list(pipeline_steps["preprocessor"].get_feature_names_out())
            if "polynomial" in pipeline_steps:
                names = list(pipeline_steps["polynomial"].get_feature_names_out(names))
            coefs = np.asarray(estimator.coef_).reshape(-1)
            n = min(len(names), len(coefs))
            coef_df = pd.DataFrame({"transformed_feature": names[:n], "coefficient": coefs[:n]})
            if task == "churn":
                coef_df["odds_ratio_per_transformed_unit"] = np.exp(np.clip(coefs[:n], -30, 30))
                coef_df["coefficient_scale"] = "log-odds per transformed feature unit; not causal"
            else:
                coef_df["coefficient_scale"] = "predicted GBP per transformed feature unit; not causal"
            coef_df["absolute_coefficient"] = coef_df["coefficient"].abs()
            path = report_dir / f"{task}_coefficients_phase5.csv"
            coef_df.sort_values("absolute_coefficient", ascending=False).to_csv(path, index=False)
            result["coefficients"] = str(path.relative_to(ROOT))
    except Exception as exc:
        result["coefficient_status"] = f"unavailable: {type(exc).__name__}: {exc}"

    result["interpretation_warning"] = "Feature importance describes predictive association for this fitted model, not causation. Correlated features can share or mask importance."
    result["explanation_data_role"] = "validation diagnostics; not an independent generalization estimate"
    result["duration_seconds"] = round(time.perf_counter() - explain_started, 4)
    return result


def _shap_report(
    task: str,
    bundle: dict[str, Any],
    explain_frame: pd.DataFrame,
    target: str,
    report_dir: Path,
    cfg5: dict[str, Any],
) -> dict[str, Any]:
    explain_started = time.perf_counter()
    max_rows = int(cfg5["tuning"]["explainability"].get("shap_rows", 100))
    try:
        import shap
    except Exception as exc:
        return {"status": "unavailable", "reason": f"SHAP import failed: {type(exc).__name__}: {exc}", "test_used": False}

    feature_names = bundle["feature_names"]
    frame = explain_frame.loc[explain_frame[target].notna()].copy()
    if task == "value" and "target_value_valid" in frame:
        frame = frame.loc[frame["target_value_valid"].fillna(False).astype(bool)]
    if frame.empty:
        return {"status": "skipped", "reason": "No validation rows", "test_used": False}
    frame = frame.sample(n=min(max_rows, len(frame)), random_state=int(cfg5["tuning"].get("seed", 42)))
    X = frame[feature_names]
    pipeline = bundle["model"]
    try:
        transformed = pipeline.named_steps["preprocessor"].transform(X)
        names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        if "polynomial" in pipeline.named_steps:
            transformed = pipeline.named_steps["polynomial"].transform(transformed)
            names = list(pipeline.named_steps["polynomial"].get_feature_names_out(names))
        estimator = pipeline.named_steps["model"]
        background = transformed[:min(50, len(transformed))]
        # SHAP auto-selects a supported explainer when possible; failure is
        # recorded rather than blocking non-SHAP explanations.
        explainer = shap.Explainer(estimator, background, feature_names=names)
        explanation = explainer(transformed)
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)
        output_scale = "model-native output; inspect estimator/explainer documentation"
        if values.ndim == 3:
            class_idx = 1 if values.shape[-1] > 1 else 0
            values = values[:, :, class_idx]
            if base_values.ndim >= 2:
                base_values = base_values[..., class_idx]
            output_scale = "class-1 native model output; not necessarily calibrated probability"
        elif task == "churn":
            output_scale = "native classifier output; not the post-calibration probability"
        else:
            output_scale = "predicted net purchase value contribution scale (GBP) when supported by explainer"
        if base_values.ndim > 1:
            base_values = base_values.reshape(len(values), -1)[:, 0]
        elif base_values.ndim == 0:
            base_values = np.repeat(float(base_values), len(values))

        def original_feature_name(name: str) -> str:
            # ColumnTransformer names look like num__feature or cat__country_UK.
            part = name.split("__", 1)[-1]
            if "_" in part and part.startswith("country_"):
                return "country"
            return part

        global_df = pd.DataFrame({
            "transformed_feature": names,
            "original_feature_group": [original_feature_name(name) for name in names],
            "mean_absolute_shap": np.mean(np.abs(values), axis=0),
            "model_version": bundle["model_version"],
            "output_scale": output_scale,
        }).sort_values("mean_absolute_shap", ascending=False)
        global_path = report_dir / f"{task}_shap_global.csv"
        global_df.to_csv(global_path, index=False)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        exp_obj = shap.Explanation(
            values=values,
            base_values=base_values,
            data=np.asarray(transformed),
            feature_names=names,
        )
        plot_status: dict[str, Any] = {}
        try:
            shap.plots.beeswarm(exp_obj, max_display=20, show=False)
            plt.gcf().set_size_inches(10, 7)
            plt.tight_layout()
            plt.savefig(report_dir / f"{task}_shap_beeswarm.png", dpi=160, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            plot_status["beeswarm"] = f"failed: {type(exc).__name__}: {exc}"
        try:
            shap.plots.waterfall(exp_obj[0], max_display=15, show=False)
            plt.gcf().set_size_inches(10, 7)
            plt.tight_layout()
            plt.savefig(report_dir / f"{task}_shap_local_waterfall.png", dpi=160, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            plot_status["waterfall"] = f"failed: {type(exc).__name__}: {exc}"

        # Include one deterministic ID and the top contributions for each local explanation.
        if task == "churn":
            raw_prob = _score_probability(pipeline, X)
            calibrator = bundle.get("calibrator")
            scored = calibrator.predict(raw_prob) if calibrator is not None else raw_prob
            score_name = "calibrated_churn_probability"
        else:
            scored = np.asarray(pipeline.predict(X), dtype=float)
            score_name = "predicted_net_value_90d_gbp"
        local_rows = []
        local_limit = int(cfg5["tuning"]["explainability"].get("local_explanation_rows", 25))
        for i in range(min(len(frame), local_limit, values.shape[0])):
            row = frame.iloc[i]
            prediction_id = hashlib.sha256(
                f"{task}|{row.get('customer_id')}|{row.get('snapshot_date')}|{bundle['model_version']}".encode()
            ).hexdigest()[:24]
            top = np.argsort(np.abs(values[i]))[::-1][:5]
            contributions = [
                {"feature": names[j], "original_feature_group": original_feature_name(names[j]), "shap_value": float(values[i, j])}
                for j in top
            ]
            local_rows.append({
                "prediction_id": prediction_id,
                "customer_id": str(row.get("customer_id")),
                "snapshot_date": str(row.get("snapshot_date")),
                "model_version": bundle["model_version"],
                score_name: float(scored[i]),
                "baseline_output": float(base_values[i]),
                "output_scale": output_scale,
                "explainer_version": getattr(shap, "__version__", "unknown"),
                "top_contributions_json": json.dumps(contributions),
                "warning": "Model explanation only; not causation or a treatment recommendation.",
            })
        local_path = report_dir / f"{task}_shap_local_contributions.csv"
        pd.DataFrame(local_rows).to_csv(local_path, index=False)
        return {
            "status": "completed",
            "global_report": str(global_path.relative_to(ROOT)),
            "local_report": str(local_path.relative_to(ROOT)),
            "rows_explained": int(len(values)),
            "feature_count": int(len(names)),
            "output_scale": output_scale,
            "duration_seconds": round(time.perf_counter() - explain_started, 4),
            "plot_status": plot_status,
            "test_used": False,
            "warning": "SHAP explains the model output under the explainer's assumptions; it does not explain causality.",
        }
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "test_used": False}


def _coefficient_fold_stability(
    task: str, model_name: str, params: dict[str, Any], train: pd.DataFrame,
    target: str, features: list[str], numeric: list[str], categorical: list[str],
    cfg3: dict[str, Any], report_dir: Path,
) -> dict[str, Any]:
    """Record coefficient variation across purged training folds for linear candidates."""
    try:
        max_folds = int(cfg3["supervised"].get("cv_max_folds", 3))
        folds = purged_walk_forward_folds(train, _horizon_for(task, cfg3), max_folds)
        if not folds:
            return {"status": "skipped", "reason": "No purged training folds"}
        rows: list[dict[str, Any]] = []
        for fold_no, (train_idx, _, train_end, val_date) in enumerate(folds, start=1):
            fold_train = train.iloc[train_idx]
            if task == "churn" and fold_train[target].nunique() < 2:
                continue
            fitted = _fit_pipeline(
                task, model_name, params, fold_train, target, features,
                numeric, categorical, int(cfg3["project"]["random_seed"]),
            )
            est = fitted.named_steps["model"]
            if not hasattr(est, "coef_"):
                return {"status": "not_applicable", "reason": f"{model_name} has no linear coefficients"}
            names = list(fitted.named_steps["preprocessor"].get_feature_names_out())
            if "polynomial" in fitted.named_steps:
                names = list(fitted.named_steps["polynomial"].get_feature_names_out(names))
            coefs = np.asarray(est.coef_).reshape(-1)
            for name, coef in zip(names, coefs):
                rows.append({
                    "fold": fold_no,
                    "train_end_snapshot": train_end,
                    "validation_snapshot": val_date,
                    "transformed_feature": name,
                    "coefficient": float(coef),
                })
        if not rows:
            return {"status": "skipped", "reason": "No fold supplied coefficients"}
        fold_df = pd.DataFrame(rows)
        fold_path = report_dir / f"{task}_coefficient_folds_phase5.csv"
        fold_df.to_csv(fold_path, index=False)
        summary = fold_df.groupby("transformed_feature")["coefficient"].agg(
            n_folds="count", mean_coefficient="mean", std_coefficient="std",
            min_coefficient="min", max_coefficient="max",
        ).reset_index()
        sign_counts = fold_df.assign(positive=fold_df["coefficient"].gt(0)).groupby("transformed_feature")["positive"].agg(
            positive_folds="sum", total_folds="count"
        ).reset_index()
        sign_counts["sign_consistency_fraction"] = np.maximum(
            sign_counts["positive_folds"],
            sign_counts["total_folds"] - sign_counts["positive_folds"],
        ) / sign_counts["total_folds"].clip(lower=1)
        summary = summary.merge(sign_counts[["transformed_feature", "sign_consistency_fraction"]], on="transformed_feature", how="left")
        summary.to_csv(report_dir / f"{task}_coefficient_stability_phase5.csv", index=False)
        return {
            "status": "completed",
            "fold_coefficients": str(fold_path.relative_to(ROOT)),
            "stability_summary": str((report_dir / f"{task}_coefficient_stability_phase5.csv").relative_to(ROOT)),
            "fold_count": int(fold_df["fold"].nunique()),
            "interpretation": "Coefficient consistency describes model fit stability across chronological folds, not causal effect stability.",
        }
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _tree_diagnostic(task: str, train: pd.DataFrame, validation: pd.DataFrame,
                     target: str, numeric: list[str], categorical: list[str],
                     features: list[str], cfg3: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    """Fit a shallow, explicitly diagnostic tree to produce a path/plot report."""
    tree_name = "decision_tree" if task == "churn" else "decision_tree_regressor"
    try:
        model = _fit_pipeline(task, tree_name, {}, train, target, features, numeric, categorical, int(cfg3["project"]["random_seed"]))
        estimator = model.named_steps["model"]
        names = list(model.named_steps["preprocessor"].get_feature_names_out())
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(estimator, feature_names=names, filled=False, rounded=True,
                  max_depth=4, impurity=True, ax=ax)
        fig.tight_layout()
        path = report_dir / f"{task}_shallow_tree_diagnostic.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        if not validation.empty:
            sample = validation.head(10)
            transformed = model.named_steps["preprocessor"].transform(sample[features])
            node_indicator = estimator.decision_path(transformed)
            leaf_ids = estimator.apply(transformed)
            rows = []
            for i, (_, row) in enumerate(sample.iterrows()):
                start, end = node_indicator.indptr[i], node_indicator.indptr[i + 1]
                nodes = node_indicator.indices[start:end]
                steps = []
                for node in nodes:
                    if node == leaf_ids[i]:
                        steps.append({"node": int(node), "leaf": True})
                        continue
                    feature_idx = int(estimator.tree_.feature[node])
                    if feature_idx >= 0:
                        steps.append({
                            "node": int(node), "feature": names[feature_idx],
                            "threshold": float(estimator.tree_.threshold[node]),
                            "comparison": "<= threshold moves left; > threshold moves right",
                        })
                pred = model.predict(sample[features].iloc[[i]])[0]
                rows.append({
                    "customer_id": str(row.get("customer_id")),
                    "snapshot_date": str(row.get("snapshot_date")),
                    "diagnostic_tree_model": tree_name,
                    "diagnostic_prediction": float(pred),
                    "decision_path_json": json.dumps(steps),
                    "warning": "Shallow tree is a diagnostic model and may not be the selected champion; paths are not causal explanations.",
                })
            pd.DataFrame(rows).to_csv(report_dir / f"{task}_shallow_tree_paths.csv", index=False)
        return {"status": "completed", "plot": str(path.relative_to(ROOT)), "diagnostic_only": True}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}", "diagnostic_only": True}


def _partial_dependence(task: str, bundle: dict[str, Any], X: pd.DataFrame,
                        importance_path: Path, report_dir: Path) -> dict[str, Any]:
    try:
        if not importance_path.exists():
            return {"status": "skipped", "reason": "Permutation importance unavailable"}
        imp = pd.read_csv(importance_path)
        valid = [name for name in imp["feature"].tolist() if name in X.columns and pd.api.types.is_numeric_dtype(X[name])]
        selected = valid[:3]
        if not selected:
            return {"status": "skipped", "reason": "No eligible numeric features"}
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(15, 4 * len(selected)))
        PartialDependenceDisplay.from_estimator(
            bundle["model"], X, features=selected, kind="average", ax=ax,
        )
        fig.tight_layout()
        path = report_dir / f"{task}_partial_dependence.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return {"status": "completed", "features": selected, "path": str(path.relative_to(ROOT))}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _make_model_card(task: str, bundle: dict[str, Any], manifest: dict[str, Any]) -> str:
    target = manifest["target"]
    if task == "churn":
        target_definition = "1 if an eligible customer makes no qualifying purchase in the next 30 days, otherwise 0. This is an inactivity proxy, not an official cancellation label."
        metric_line = f"Selection-window PR-AUC: {manifest.get('selection_metric')}; threshold: {manifest.get('threshold')}; calibrated Brier on threshold window: {manifest.get('threshold_policy', {}).get('calibrated_brier_threshold_window')}."
    else:
        target_definition = "Net transaction-line value in the next 90 days, including recorded negative/cancellation lines; measured in GBP, not profit or contribution margin."
        metric_line = f"Selection-window MAE: {manifest.get('selection_metric')} GBP; RMSE: {manifest.get('selection_rmse_gbp')} GBP; R²: {manifest.get('selection_r2')}."
    return f"""# Phase 5 Model Card — {task.title()}

## Identity
- Model version: `{bundle['model_version']}`
- Run ID: `{bundle['run_id']}`
- Selected candidate: `{bundle['model_name']}`
- Parameters: `{json.dumps(bundle.get('model_params', {}), sort_keys=True)}`
- Artifact SHA-256: `{manifest.get('artifact_sha256')}`
- Approval status: `{manifest.get('approval_status')}`

## Intended use
Analytical decision support and offline model evaluation. This candidate is not automatically approved for customer treatment or production service.

## Prediction target
- Target column: `{target}`
- Definition: {target_definition}

## Feature policy
- Feature names: `{', '.join(bundle['feature_names'])}`
- Customer IDs, split markers, timestamps used only as metadata, and target/label columns are excluded from model inputs.
- Learned preprocessing remains inside the fitted model pipeline.
- Feature values must represent information available before the snapshot time.

## Tuning and validation
- Hyperparameter method: Optuna with a persistent SQLite study and a fixed total trial budget.
- Cross-validation: purged walk-forward folds over training snapshots only.
- Model-selection validation metric: {metric_line}
- Test data used during Phase 5: **No**.

## Calibration and threshold
- Calibration: `{manifest.get('threshold_policy', {}).get('calibration_method')}`
- Threshold policy: `{json.dumps(manifest.get('threshold_policy', {}), sort_keys=True)}`

## Explanations
- Permutation importance and SHAP status are reported in the Phase 5 run manifest.
- Explanations describe model behaviour, not causation.
- Correlated features can share or mask importance; stability and privacy review are required before external exposure.

## Limitations
- Online Retail II does not provide official churn/cancellation ground truth or investigator-confirmed anomaly labels.
- Churn is an inactivity proxy; net transaction-line value is not profit or contribution margin.
- Retention uplift is not estimable without a controlled experiment and explicit value assumptions.
- Validation data has been used for model selection/calibration/threshold decisions; it is not an unbiased final test estimate.
- Review error and subgroup reports before approval.

## Reproducibility
- Input snapshot SHA-256: `{manifest.get('data_sha256')}`
- Phase 4 comparison SHA-256: `{manifest.get('phase4_comparison_sha256')}`
- Git commit: `{manifest.get('git_commit')}`
- Environment: `{json.dumps(manifest.get('environment', {}), sort_keys=True)}`
"""


def _fit_task(
    task: str,
    frame: pd.DataFrame,
    phase4: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    cfg3: dict[str, Any],
    cfg5: dict[str, Any],
    source_path: Path,
    preflight: dict[str, Any],
    output_dir: Path,
    model_dir: Path,
    trials: int,
    timeout_seconds: int,
    max_models: int,
    skip_shap: bool,
) -> dict[str, Any]:
    run_id = make_run_id()
    target = _target_for(task, cfg3)
    horizon = _horizon_for(task, cfg3)
    task_frame = _numeric_target(frame.loc[frame["split"].isin(["train", "validation"])], target, task)
    train = task_frame.loc[task_frame["split"].eq("train")].copy().reset_index(drop=True)
    validation = task_frame.loc[task_frame["split"].eq("validation")].copy().reset_index(drop=True)
    if task == "churn" and (train.empty or train[target].nunique() < 2):
        raise ValueError("Churn training data needs both classes and non-empty training rows")
    if train.empty or validation.empty:
        raise ValueError(f"{task} needs non-empty train and validation rows")
    features = numeric + categorical

    selection, calibration, threshold_rows, roles = _time_roles(validation, cfg3)
    ranked = _ranked_phase4_candidates(phase4, task)
    baseline_name = None
    baseline_params: dict[str, Any] = {}
    for _, row in ranked.iterrows():
        candidate = str(row["model"])
        if candidate != BASELINE_NAMES[task]:
            baseline_name = candidate
            baseline_params = _parse_params(row.get("diagnostic_refit_params", row.get("params", {})))
            break
    if baseline_name is None:
        raise ValueError(f"Phase 4 has no successful non-dummy candidate for {task}")

    # Keep a same-window, Phase 4-ranked comparator with default/recorded parameters.
    comparison_rows: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    baseline_pipeline = _fit_pipeline(
        task, baseline_name, baseline_params, train, target, features,
        numeric, categorical, int(cfg3["project"]["random_seed"]),
    )
    baseline_metrics = _score_model(task, baseline_pipeline, selection[features], selection[target].to_numpy(dtype=int if task == "churn" else float))
    baseline_entry = {
        "task": task, "model": baseline_name, "status": "PASSED",
        "candidate_type": "Phase 4 ranked baseline re-fit with recorded/default parameters",
        "is_tuned": False, "params": baseline_params,
        **baseline_metrics,
    }
    candidates[f"baseline::{baseline_name}"] = {
        "model_name": baseline_name, "params": baseline_params,
        "pipeline": baseline_pipeline, "row": baseline_entry,
    }
    comparison_rows.append(baseline_entry)

    candidate_cfg = cfg5["tuning"].get("candidate_models", {}).get(task, [])
    candidate_cfg = [str(x) for x in candidate_cfg]
    ranked_tunable = []
    for _, row in ranked.iterrows():
        name = str(row["model"])
        if name in candidate_cfg and name in TUNABLE_MODELS[task] and name not in ranked_tunable:
            ranked_tunable.append(name)
    ranked_tunable = ranked_tunable[:max(0, max_models)]

    optuna_dir = output_dir / "optuna"
    tuning_records = []
    if not ranked_tunable:
        tuning_records.append({
            "status": "SKIPPED_NO_TUNABLE_CANDIDATE_IN_PHASE4",
            "reason": (
                f"Phase 4 has no successfully evaluated tunable model for {task}. "
                "The best-ranked candidate remains an untuned comparator; this can be expected "
                "for a model such as ordinary Linear Regression that has no meaningful hyperparameters. "
                "For actual tuning, rerun Phase 3/4 with the desired tunable model family included."
            ),
        })
    for model_name in ranked_tunable:
        try:
            tuning = _suggest_and_tune(
                task, model_name, train, features, numeric, categorical,
                cfg3, cfg5, preflight["phase3_snapshot_sha256"],
                output_dir, optuna_dir, trials, timeout_seconds,
            )
            params = tuning["best_params"]
            pipeline = _fit_pipeline(
                task, model_name, params, train, target, features,
                numeric, categorical, int(cfg3["project"]["random_seed"]),
            )
            metrics = _score_model(
                task, pipeline, selection[features],
                selection[target].to_numpy(dtype=int if task == "churn" else float),
            )
            entry = {
                "task": task, "model": model_name, "status": "PASSED",
                "candidate_type": "Optuna-tuned candidate",
                "is_tuned": True,
                "optuna_best_cv_value": tuning["best_value"],
                "optuna_cv_direction": tuning["direction"],
                "optuna_best_trial": tuning["best_trial_number"],
                "optuna_total_trials": tuning["total_study_trials"],
                "optuna_cv_fold_std": tuning.get("best_trial_fold_std"),
                "params": params,
                **metrics,
            }
            candidates[f"tuned::{model_name}"] = {
                "model_name": model_name, "params": params,
                "pipeline": pipeline, "row": entry,
                "tuning": tuning,
            }
            comparison_rows.append(entry)
            tuning_records.append({"model": model_name, "status": "PASSED", **tuning})
            print(f"[{task}] tuned {model_name}; validation metric={metrics.get('selection_pr_auc', metrics.get('selection_mae_gbp'))}")
        except Exception as exc:
            failure = {
                "task": task, "model": model_name, "status": "FAILED",
                "candidate_type": "Optuna-tuned candidate",
                "error": f"{type(exc).__name__}: {exc}",
            }
            comparison_rows.append(failure)
            tuning_records.append({"model": model_name, "status": "FAILED", "error": failure["error"]})
            print(f"[{task}] tuning failed for {model_name}: {failure['error']}")

    good = [item for item in candidates.values() if item["row"].get("status") == "PASSED"]
    if task == "churn":
        winner = max(good, key=lambda item: item["row"].get("selection_pr_auc", -np.inf))
        metric = winner["row"].get("selection_pr_auc")
        dummy_metrics = phase4.loc[(phase4.task.eq(task)) & (phase4.model.eq("dummy"))]
        dummy_value = pd.to_numeric(dummy_metrics.get("validation_pr_auc_average_precision", pd.Series(dtype=float)), errors="coerce").dropna()
        if dummy_value.empty:
            dummy_pipeline = _fit_pipeline(task, "dummy", {}, train, target, features, numeric, categorical, int(cfg3["project"]["random_seed"]))
            dummy_value_number = float(average_precision_score(selection[target].astype(int), _score_probability(dummy_pipeline, selection[features]))) if selection[target].nunique() > 1 else float(selection[target].mean())
        else:
            # Refit and score dummy on exactly the selection window for like-for-like comparison.
            dummy_pipeline = _fit_pipeline(task, "dummy", {}, train, target, features, numeric, categorical, int(cfg3["project"]["random_seed"]))
            dummy_value_number = float(average_precision_score(selection[target].astype(int), _score_probability(dummy_pipeline, selection[features]))) if selection[target].nunique() > 1 else float(selection[target].mean())
        baseline_minimum = float(cfg5["tuning"]["approval"].get("minimum_churn_pr_auc_improvement_over_dummy", 0.0005))
        beats_dummy = metric is not None and metric > dummy_value_number + baseline_minimum
        improvement = (metric - dummy_value_number) if metric is not None else None
    else:
        winner = min(good, key=lambda item: item["row"].get("selection_mae_gbp", np.inf))
        metric = winner["row"].get("selection_mae_gbp")
        dummy_pipeline = _fit_pipeline(task, "dummy_median", {}, train, target, features, numeric, categorical, int(cfg3["project"]["random_seed"]))
        dummy_value_number = float(mean_absolute_error(selection[target].astype(float), dummy_pipeline.predict(selection[features])))
        baseline_minimum = float(cfg5["tuning"]["approval"].get("minimum_value_mae_improvement_over_dummy_gbp", 0.01))
        beats_dummy = metric is not None and metric < dummy_value_number - baseline_minimum
        improvement = (dummy_value_number - metric) if metric is not None else None

    approval_status = (
        "candidate_beats_dummy_on_model_selection_validation"
        if beats_dummy else
        "not_approved_did_not_beat_dummy_on_model_selection_validation"
    )
    selected_name = winner["model_name"]
    selected_params = winner["params"]
    model = winner["pipeline"]
    # Calibration and threshold use separate validation date ranges where available.
    calibrator, threshold_value, threshold_policy = _calibration_and_threshold(
        model, train, validation, selection, calibration, threshold_rows, roles,
        cfg3, cfg5, target, features, output_dir,
    )

    model_version = f"{task}-phase5-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{run_id}"
    bundle = {
        "task": task,
        "model_name": selected_name,
        "model": model,
        "model_params": selected_params,
        "calibrator": calibrator,
        "threshold": threshold_value,
        "threshold_policy": threshold_policy,
        "model_version": model_version,
        "run_id": run_id,
        "approval_status": approval_status,
        "target": target,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_names": features,
        "data_sha256": preflight["phase3_snapshot_sha256"],
        "created_at_utc": now_utc(),
        "test_used": False,
    }

    # Serialization round-trip: same raw feature rows must give same predictions.
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"{task}_model.joblib"
    sample_X = selection[features].head(min(20, len(selection)))
    if task == "churn":
        before_raw = _score_probability(model, sample_X)
        before = calibrator.predict(before_raw) if calibrator is not None else before_raw
    else:
        before = np.asarray(model.predict(sample_X), dtype=float)
    joblib.dump(bundle, artifact_path)
    reloaded_bundle = joblib.load(artifact_path)
    reloaded_model = reloaded_bundle["model"]
    if task == "churn":
        after_raw = _score_probability(reloaded_model, sample_X)
        after = reloaded_bundle["calibrator"].predict(after_raw) if reloaded_bundle.get("calibrator") is not None else after_raw
    else:
        after = np.asarray(reloaded_model.predict(sample_X), dtype=float)
    max_abs_diff = float(np.max(np.abs(np.asarray(before) - np.asarray(after)))) if len(sample_X) else 0.0
    if not np.allclose(before, after, rtol=1e-10, atol=1e-10):
        raise RuntimeError(f"Serialization round-trip predictions changed for {task}; max difference={max_abs_diff}")

    validation_metrics: dict[str, Any]
    if task == "churn":
        y_thr = threshold_rows[target].astype(int).to_numpy()
        raw_thr = _score_probability(model, threshold_rows[features])
        cal_thr = calibrator.predict(raw_thr) if calibrator is not None else raw_thr
        validation_metrics = classification_metrics(
            y_thr, cal_thr, float(threshold_value), threshold_rows,
            float(cfg5["tuning"]["threshold_policy"].get("capacity_fraction", cfg3["supervised"].get("retention_capacity_fraction", 0.1))),
        )
        validation_metrics["brier_raw"] = float(brier_score_loss(y_thr, raw_thr)) if len(y_thr) else None
        validation_metrics["brier_calibrated"] = float(brier_score_loss(y_thr, cal_thr)) if len(y_thr) else None
        selection_metric_field = "selection_pr_auc"
        cv_metric_field = "optuna_best_cv_value" if winner.get("tuning") else "phase4_baseline_comparator"
    else:
        y_thr = threshold_rows[target].astype(float).to_numpy()
        pred_thr = model.predict(threshold_rows[features])
        validation_metrics = regression_metrics(y_thr, pred_thr, train[target].astype(float).to_numpy())
        validation_metrics["selection_window_metrics"] = _score_model(task, model, selection[features], selection[target].astype(float).to_numpy())
        selection_metric_field = "selection_mae_gbp"
        cv_metric_field = "optuna_best_cv_value" if winner.get("tuning") else "phase4_baseline_comparator"

    # Explain the final candidate on validation only; no test data is passed here.
    explanation = _permutation_and_coefficients(task, bundle, validation, target, output_dir, cfg5)
    coefficient_stability = _coefficient_fold_stability(
        task, selected_name, selected_params, train, target, features,
        numeric, categorical, cfg3, output_dir,
    )
    explanation["coefficient_fold_stability"] = coefficient_stability
    shap_result = {"status": "disabled_by_flag", "test_used": False}
    if not skip_shap and bool(cfg5["tuning"]["explainability"].get("run_shap", True)):
        shap_result = _shap_report(task, bundle, validation, target, output_dir, cfg5)
    pd_result = {"status": "disabled_by_config"}
    if bool(cfg5["tuning"]["explainability"].get("run_partial_dependence", True)):
        exp_frame = validation.loc[validation[target].notna()].copy()
        if task == "value" and "target_value_valid" in exp_frame:
            exp_frame = exp_frame.loc[exp_frame["target_value_valid"].fillna(False).astype(bool)]
        exp_frame = exp_frame.sample(n=min(500, len(exp_frame)), random_state=int(cfg5["tuning"].get("seed", 42))) if len(exp_frame) else exp_frame
        pd_result = _partial_dependence(task, bundle, exp_frame[features] if len(exp_frame) else exp_frame, output_dir / f"{task}_permutation_importance.csv", output_dir)
    tree_result = _tree_diagnostic(task, train, validation, target, numeric, categorical, features, cfg3, output_dir)

    report_rows = []
    for item in comparison_rows:
        row = dict(item)
        row["selected_phase5_candidate"] = row.get("model") == selected_name and row.get("is_tuned", False) == winner["row"].get("is_tuned", False)
        row["dummy_selection_metric"] = dummy_value_number
        row["improvement_over_dummy"] = improvement if row["selected_phase5_candidate"] else None
        row["beats_dummy_baseline"] = bool(beats_dummy) if row["selected_phase5_candidate"] else None
        row["test_used_for_selection"] = False
        report_rows.append(row)
    pd.DataFrame(report_rows).to_csv(output_dir / f"{task}_phase5_candidate_comparison.csv", index=False)

    manifest = {
        "phase": 5,
        "task": task,
        "run_id": run_id,
        "model_version": model_version,
        "selected_model": selected_name,
        "selected_params": selected_params,
        "selected_candidate_was_tuned": bool(winner["row"].get("is_tuned", False)),
        "optuna_study": winner.get("tuning"),
        "target": target,
        "horizon_days": horizon,
        "selection_metric": metric,
        "selection_metric_field": selection_metric_field,
        "cv_metric_field": cv_metric_field,
        "dummy_selection_metric": dummy_value_number,
        "improvement_over_dummy": improvement,
        "beats_dummy_baseline": bool(beats_dummy),
        "approval_status": approval_status,
        "threshold": threshold_value,
        "threshold_policy": threshold_policy,
        "selection_rows": int(len(selection)),
        "calibration_rows": int(len(calibration)),
        "threshold_rows": int(len(threshold_rows)),
        "validation_metrics_on_threshold_window": validation_metrics,
        "cv_tuning_records": tuning_records,
        "data_path": str(source_path.relative_to(ROOT)),
        "data_sha256": preflight["phase3_snapshot_sha256"],
        "phase4_comparison_sha256": preflight["phase4_comparison_sha256"],
        "artifact_path": str(artifact_path.relative_to(ROOT)),
        "artifact_sha256": sha256_file(artifact_path),
        "artifact_size_bytes": int(artifact_path.stat().st_size),
        "serialization_roundtrip_max_abs_difference": max_abs_diff,
        "feature_names": features,
        "feature_time_contract": "Features were built from transaction history strictly before each snapshot timestamp in Phase 2; Phase 5 excludes identifier/timestamp/target metadata from model inputs.",
        "explanations": explanation,
        "shap": shap_result,
        "partial_dependence": pd_result,
        "shallow_tree_diagnostic": tree_result,
        "test_split_used_for_fit_tuning_calibration_explanation_or_selection": False,
        "test_metrics_used_for_selection": False,
        "git_commit": git_commit(),
        "environment": log_environment(),
        "created_at_utc": now_utc(),
        "notes": [
            "Optuna objectives use purged chronological folds from training snapshots only.",
            "Model selection, calibration, and thresholding use separate validation time windows when enough dates exist.",
            "The validation report is model-development evidence, not an unbiased final test estimate.",
            "The churn target is inactivity proxy, not verified cancellation.",
            "Customer value is net transaction-line value in GBP, not profit or contribution margin.",
            "No retention experiment or investigator-confirmed anomaly outcomes are present in Online Retail II.",
            "SHAP and permutation reports explain the model, not causation.",
        ],
    }
    manifest_path = output_dir / f"{task}_phase5_manifest.json"
    write_json(manifest_path, manifest)
    card_path = output_dir / f"model_card_{task}_phase5.md"
    card_path.write_text(_make_model_card(task, bundle, manifest), encoding="utf-8")
    bundle["artifact_sha256"] = manifest["artifact_sha256"]
    bundle["approval_status"] = approval_status
    # Save status/checksum in the same artifact bundle after computing predictions; the checksum
    # in the manifest describes the final file only if we do not rewrite the artifact after hash.
    # Avoid mutating the artifact here to keep the manifest checksum valid.
    return manifest


def run_model_tuning(task_filter: str = "all", trials: int | None = None,
               timeout_seconds: int | None = None, max_models_per_task: int | None = None,
               skip_shap: bool = False) -> dict[str, Any]:
    cfg3 = load_phase3_config()
    cfg5 = load_phase5_config()
    root_cfg = cfg5["tuning"]
    output_dir = project_path(root_cfg["output_dir"])
    model_dir = project_path(root_cfg["model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    report_dir3 = project_path(cfg3["paths"]["reports_dir"])
    model_dir3 = project_path(cfg3["paths"]["models_dir"])
    preflight = _phase4_preflight(cfg3, report_dir3, model_dir3)
    frame, source_path = load_snapshots()
    numeric, categorical, excluded = feature_columns(frame)
    # Partition membership was persisted in Phase 3; Phase 5 only consumes train/validation rows.
    phase4 = _load_comparison()
    tasks = ["churn", "value"] if task_filter == "all" else [task_filter]
    results = {}
    config_trials = int(root_cfg["optuna"].get("trials_per_model", 10))
    config_timeout = int(root_cfg["optuna"].get("timeout_seconds_per_model", 900))
    config_models = int(root_cfg["optuna"].get("max_models_per_task", 2))
    final_trials = config_trials if trials is None else int(trials)
    final_timeout = config_timeout if timeout_seconds is None else int(timeout_seconds)
    final_max_models = config_models if max_models_per_task is None else int(max_models_per_task)
    if final_trials < 1 or final_timeout < 1 or final_max_models < 1:
        raise ValueError("Trials, timeout, and max-model count must all be positive")

    for task in tasks:
        print(f"\n=== PHASE 5: {task.upper()} TUNING AND EXPLANATION ===")
        results[task] = _fit_task(
            task, frame, phase4, numeric, categorical, cfg3, cfg5,
            source_path, preflight, output_dir, model_dir,
            final_trials, final_timeout, final_max_models, skip_shap,
        )

    run_manifest = {
        "phase": 5,
        "title": "Tuning, calibration, threshold policy, and explainability",
        "preflight": preflight,
        "configuration_path": str(CONFIG_PATH.relative_to(ROOT)),
        "configuration_sha256": sha256_file(CONFIG_PATH),
        "tasks": results,
        "test_split_used_for_fit_tuning_calibration_explanation_or_selection": False,
        "test_metrics_used_for_selection": False,
        "created_at_utc": now_utc(),
    }
    write_json(output_dir / "phase5_run_manifest.json", run_manifest)
    report_lines = [
        "# Phase 5 — Tuning and Explanation",
        "",
        "This run is connected to Phase 3 feature/model pipelines and Phase 4 supervised comparison. It uses Optuna on purged train-only chronological folds, then validation-date windows for candidate selection, calibration and thresholding where available. No test rows are used for fitting, tuning, calibration, threshold selection, explanations, ranking, or metrics; the shared loader reads the complete snapshot file only for schema/split integrity checks.",
        "",
    ]
    for task, item in results.items():
        report_lines.extend([
            f"## {task.title()}",
            f"- Model: `{item['selected_model']}`",
            f"- Candidate status: `{item['approval_status']}`",
            f"- Validation selection metric: `{item.get('selection_metric')}`",
            f"- Tuned: `{item.get('selected_candidate_was_tuned')}`",
            f"- Threshold: `{item.get('threshold')}`",
            f"- Calibration: `{item.get('threshold_policy', {}).get('calibration_method')}`",
            f"- SHAP status: `{item.get('shap', {}).get('status')}`",
            f"- Model card: `reports/phase5/model_card_{task}_phase5.md`",
            "",
        ])
    report_lines.extend([
        "## Interpretation and limitations",
        "- Validation metrics are development diagnostics, not unbiased final test metrics.",
        "- Churn is inactivity proxy, not verified cancellation.",
        "- Value means net recorded transaction-line value, not profit.",
        "- Retention uplift and investigator-confirmed anomaly precision require independent labelled experiments/reviews.",
        "- SHAP and permutation importance are model explanations, not causal effects.",
    ])
    (output_dir / "phase5_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nPhase 5 complete. Report: {output_dir / 'phase5_report.md'}")
    print("No test rows were used for fitting, tuning, calibration, threshold selection, explanations, ranking, or metrics. The shared snapshot loader reads the full file for schema/split integrity checks.")
    return run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["churn", "value", "all"], default="all")
    parser.add_argument("--trials", type=int, default=None, help="Persistent total Optuna trial budget per model")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Timeout per model study")
    parser.add_argument("--max-models-per-task", type=int, default=None, help="Maximum Phase 4-ranked tunable candidates per task")
    parser.add_argument("--skip-shap", action="store_true", help="Skip optional SHAP calculations; permutation/coefficients remain")
    args = parser.parse_args()
    run_model_tuning(args.task, args.trials, args.timeout_seconds, args.max_models_per_task, args.skip_shap)


if __name__ == "__main__":
    main()
