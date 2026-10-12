from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.training.common import (
    ROOT,
    PlattCalibrator,
    append_jsonl,
    build_pipeline,
    classification_metrics,
    feature_columns,
    git_commit,
    json_dump,
    load_config,
    load_snapshots,
    log_environment,
    make_run_id,
    persist_split_membership,
    project_path,
    purged_walk_forward_folds,
    regression_metrics,
    select_threshold,
    sha256_file,
    split_validation_roles,
)


# Required model families are always included in the default comparison.
CORE_CLASSIFIERS = [
    "dummy",
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
    "svm",
    "knn",
]
CORE_REGRESSORS = [
    "dummy_median",
    "linear_regression",
    "ridge",
    "elastic_net",
    "random_forest_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
]

OPTIONAL_CLASSIFIERS = [
    "extra_trees",
    "adaboost",
    "gradient_boosting",
    "hist_gradient_boosting",
    "svm_rbf",
    "naive_bayes",
    "catboost",
    "voting",
]
OPTIONAL_REGRESSORS = [
    "lasso",
    "polynomial_regression",
    "decision_tree_regressor",
    "extra_trees_regressor",
    "adaboost_regressor",
    "gradient_boosting_regressor",
    "hist_gradient_boosting_regressor",
    "catboost_regressor",
    "voting_regressor",
]


def _bool_series(series: pd.Series) -> pd.Series:
    """Convert bool/string/0-1 columns into a safe Boolean Series."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _safe_target_frame(frame: pd.DataFrame, target: str, task: str) -> pd.DataFrame:
    """Keep rows with finite labels; validate binary churn labels."""
    out = frame.loc[frame[target].notna()].copy()
    out[target] = pd.to_numeric(out[target], errors="coerce")
    out = out.loc[np.isfinite(out[target].to_numpy(dtype=float))].copy()
    if task == "churn":
        out = out.loc[out[target].isin([0, 1])].copy()
        out[target] = out[target].astype("int8")
    return out


def _score_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return the probability for class 1 from an sklearn-style pipeline."""
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not implement predict_proba")
    probabilities = np.asarray(model.predict_proba(X))
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Classifier did not return two-class probabilities")
    classes = getattr(model, "classes_", None)
    if classes is not None and 1 in list(classes):
        return probabilities[:, list(classes).index(1)]
    return probabilities[:, -1]


def _probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Metrics that do not require choosing a class threshold."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "positive_rate": float(y.mean()) if len(y) else None,
        "pr_auc_average_precision": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else None,
        "brier_score": float(brier_score_loss(y, p)) if len(y) else None,
    }


def _latency_benchmark(
    model: Any,
    X: pd.DataFrame,
    task: str,
    n_rows: int = 30,
) -> dict[str, Any]:
    """Measure approximate per-row latency and batch throughput locally."""
    if X.empty:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "batch_rows_per_second": None}
    sample = X.iloc[: min(len(X), max(1, n_rows))]
    timings: list[float] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task == "churn":
                _score_probability(model, sample.iloc[:1])
            else:
                model.predict(sample.iloc[:1])
            for i in range(len(sample)):
                one = sample.iloc[[i]]
                started = time.perf_counter()
                if task == "churn":
                    _score_probability(model, one)
                else:
                    model.predict(one)
                timings.append((time.perf_counter() - started) * 1000.0)
            batch = X.iloc[: min(512, len(X))]
            started = time.perf_counter()
            if task == "churn":
                _score_probability(model, batch)
            else:
                model.predict(batch)
            elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            "p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)),
            "p99_ms": float(np.percentile(timings, 99)),
            "batch_rows_per_second": float(len(batch) / elapsed),
        }
    except Exception as exc:
        return {
            "p50_ms": None, "p95_ms": None, "p99_ms": None,
            "batch_rows_per_second": None,
            "benchmark_error": f"{type(exc).__name__}: {exc}",
        }


def _serialized_size(model: Any) -> int | None:
    try:
        return int(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))
    except Exception:
        return None


def _read_params_file(path: Path | None, task: str) -> tuple[str | None, dict[str, Any]]:
    if path is None:
        return None, {}
    if not path.exists():
        raise FileNotFoundError(f"Optuna parameters file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != task:
        raise ValueError(
            f"Parameters file task={payload.get('task')!r} does not match task={task!r}"
        )
    model_name = payload.get("model")
    best_params = payload.get("best_params", {})
    if not model_name or not isinstance(best_params, dict):
        raise ValueError("Parameters JSON must contain 'model' and object 'best_params'")
    return str(model_name), best_params


def _write_churn_split_manifest(
    frame: pd.DataFrame,
    config: dict[str, Any],
    source_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use 30-day label windows to safely reclaim suitable purged-gap snapshots.

    Phase 2 assigned gaps for a 90-day outcome. The PDF's churn horizon is 30 days,
    so November/December gaps can be added to churn train and later gaps can be
    added to churn validation only when their full 30-day outcome ends before the
    next partition. Value regression keeps the original 90-day split unchanged.
    """
    result = frame.copy()
    if "split" not in result.columns:
        raise ValueError("The snapshot dataset has no 'split' column")
    result["original_phase2_split"] = result["split"].astype("string")
    original_split = result["original_phase2_split"].fillna("").astype(str)
    snapshots = pd.to_datetime(result["snapshot_date"], errors="coerce").dt.normalize()
    if snapshots.isna().any():
        raise ValueError("snapshot_date has missing or invalid values")

    horizon = int(config["supervised"].get("churn_horizon_days", 30))
    if "churn_30d_outcome_end_exclusive" in result.columns:
        outcome_end = pd.to_datetime(
            result["churn_30d_outcome_end_exclusive"], errors="coerce"
        ).dt.normalize()
    else:
        outcome_end = snapshots + pd.Timedelta(days=horizon)

    if "target_churn_30d_matured" in result.columns:
        matured = _bool_series(result["target_churn_30d_matured"])
    else:
        # Null target rows are still removed later by _safe_target_frame.
        matured = pd.Series(True, index=result.index)

    original_validation_dates = snapshots.loc[original_split.eq("validation") & matured]
    original_test_dates = snapshots.loc[original_split.eq("test") & matured]
    if original_validation_dates.empty or original_test_dates.empty:
        counts = pd.crosstab(original_split, result[config["supervised"]["churn_target"]], dropna=False)
        raise ValueError(
            "Cannot establish churn validation/test date boundaries. "
            f"Original split/target counts:\n{counts.to_string()}"
        )

    validation_start = original_validation_dates.min()
    validation_last = original_validation_dates.max()
    test_start = original_test_dates.min()

    task_split = pd.Series("purged_gap", index=result.index, dtype="string")

    # Keep only mature rows for assigned development/test partitions.
    eligible_train = (
        original_split.eq("train")
        & matured
        & outcome_end.notna()
        & outcome_end.le(validation_start)
    )
    eligible_early_gap = (
        original_split.eq("purged_gap")
        & matured
        & snapshots.lt(validation_start)
        & outcome_end.notna()
        & outcome_end.le(validation_start)
    )
    eligible_validation = (
        original_split.eq("validation")
        & matured
        & outcome_end.notna()
        & outcome_end.le(test_start)
    )
    eligible_late_gap = (
        original_split.eq("purged_gap")
        & matured
        & snapshots.gt(validation_last)
        & snapshots.lt(test_start)
        & outcome_end.notna()
        & outcome_end.le(test_start)
    )
    eligible_test = original_split.eq("test") & matured

    task_split.loc[eligible_train | eligible_early_gap] = "train"
    task_split.loc[eligible_validation | eligible_late_gap] = "validation"
    task_split.loc[eligible_test] = "test"
    result["split"] = task_split
    result["churn_horizon_days_used"] = horizon

    # Verify that complete target windows cannot cross the partition boundaries.
    assigned_train_end = outcome_end.loc[result["split"].eq("train")].max()
    assigned_validation_end = outcome_end.loc[result["split"].eq("validation")].max()
    assigned_validation_start = snapshots.loc[result["split"].eq("validation")].min()
    assigned_test_start = snapshots.loc[result["split"].eq("test")].min()
    if pd.isna(assigned_train_end) or pd.isna(assigned_validation_start):
        raise ValueError("Churn split is empty after maturity and boundary checks")
    if assigned_train_end > assigned_validation_start:
        raise ValueError(
            "Churn train outcome windows overlap validation start: "
            f"{assigned_train_end} > {assigned_validation_start}"
        )
    if pd.isna(assigned_validation_end) or pd.isna(assigned_test_start):
        raise ValueError("Churn validation/test split is empty after boundary checks")
    if assigned_validation_end > assigned_test_start:
        raise ValueError(
            "Churn validation outcome windows overlap test start: "
            f"{assigned_validation_end} > {assigned_test_start}"
        )

    counts_by_source = pd.crosstab(
        result["original_phase2_split"].fillna("<missing>"),
        result[config["supervised"]["churn_target"]],
        dropna=False,
    )
    original_counts = {
        str(row_name): {str(k): int(v) for k, v in row.items() if pd.notna(k)}
        for row_name, row in counts_by_source.iterrows()
    }
    assigned_counts = {
        split: {
            str(k): int(v)
            for k, v in result.loc[result["split"].eq(split), config["supervised"]["churn_target"]]
            .value_counts(dropna=False).items()
        }
        for split in ("train", "validation", "test", "purged_gap")
    }

    manifest = {
        "task": "churn",
        "target": config["supervised"]["churn_target"],
        "horizon_days": horizon,
        "source_snapshot_file": str(source_path.relative_to(ROOT)),
        "source_snapshot_sha256": sha256_file(source_path),
        "original_phase2_split_counts_by_target": original_counts,
        "churn_task_split_counts_by_target": assigned_counts,
        "validation_start": assigned_validation_start,
        "validation_last_snapshot": snapshots.loc[result["split"].eq("validation")].max(),
        "test_start": assigned_test_start,
        "train_outcome_end_exclusive_max": assigned_train_end,
        "validation_outcome_end_exclusive_max": assigned_validation_end,
        "gap_snapshots_reassigned_to_train": int(eligible_early_gap.sum()),
        "gap_snapshots_reassigned_to_validation": int(eligible_late_gap.sum()),
        "policy": "Only mature 30-day labels are used. Purged-gap rows are reassigned only when their full outcome window ends before the next partition starts.",
    }
    return result, manifest


def _print_target_diagnostic(frame: pd.DataFrame, target: str, heading: str) -> None:
    print(f"\n{heading}")
    print("Overall labels:")
    print(frame[target].value_counts(dropna=False).sort_index().to_string())
    print("\nLabels by current split:")
    if "split" in frame.columns:
        print(pd.crosstab(frame["split"], frame[target], dropna=False).to_string())


def _class_distribution_error(
    train: pd.DataFrame,
    full_frame: pd.DataFrame,
    target: str,
) -> ValueError:
    train_counts = train[target].value_counts(dropna=False).sort_index().to_dict()
    all_counts = full_frame[target].value_counts(dropna=False).sort_index().to_dict()
    split_counts = pd.crosstab(
        full_frame["split"], full_frame[target], dropna=False
    ) if "split" in full_frame.columns else pd.DataFrame()
    detail = split_counts.to_string() if not split_counts.empty else "No split distribution available."
    return ValueError(
        "Churn training still contains only one class after applying the 30-day maturity "
        "and safe purged-gap reassignment rules. A binary classifier cannot be trained "
        "from a single-class training target.\n"
        f"Training target counts: {train_counts}\n"
        f"All eligible target counts: {all_counts}\n"
        f"Split-by-target counts:\n{detail}\n\n"
        "Check reports/phase3/churn_target_manifest.json and the target generation. "
        "Do not fix this by relabelling customers, lowering thresholds, or applying SMOTE "
        "to validation/test data. Confirm customer IDs match between snapshots and transactions, "
        "and verify that both target values 0 and 1 genuinely exist in the training period."
    )


def _classification_cv(
    frame: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    model_name: str,
    seed: int,
    horizon: int,
    max_folds: int,
    params: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    X = frame[numeric + categorical]
    y = frame[target].astype(int)
    folds = purged_walk_forward_folds(frame, horizon_days=horizon, max_folds=max_folds)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for fold_id, (train_idx, valid_idx, train_end, valid_date) in enumerate(folds, start=1):
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        row: dict[str, Any] = {
            "fold": fold_id,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "last_train_snapshot": train_end,
            "validation_snapshot": valid_date,
            "train_class_counts": json.dumps({str(k): int(v) for k, v in y_train.value_counts().items()}),
            "validation_class_counts": json.dumps({str(k): int(v) for k, v in y_valid.value_counts().items()}),
        }
        try:
            if y_train.nunique() < 2:
                raise ValueError(f"fold training target contains one class: {y_train.value_counts().to_dict()}")
            model = build_pipeline("churn", model_name, numeric, categorical, seed, params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.iloc[train_idx], y_train)
                probability = _score_probability(model, X.iloc[valid_idx])
            row.update(_probability_metrics(y_valid.to_numpy(), probability))
            row["status"] = "PASSED"
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(row["error"])
        rows.append(row)
    return rows, errors


def _regression_cv(
    frame: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    model_name: str,
    seed: int,
    horizon: int,
    max_folds: int,
    params: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    X = frame[numeric + categorical]
    y = frame[target].astype(float)
    folds = purged_walk_forward_folds(frame, horizon_days=horizon, max_folds=max_folds)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for fold_id, (train_idx, valid_idx, train_end, valid_date) in enumerate(folds, start=1):
        row: dict[str, Any] = {
            "fold": fold_id,
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
            "last_train_snapshot": train_end,
            "validation_snapshot": valid_date,
        }
        try:
            model = build_pipeline("value", model_name, numeric, categorical, seed, params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                prediction = model.predict(X.iloc[valid_idx])
            actual = y.iloc[valid_idx].to_numpy()
            row["mae_gbp"] = float(mean_absolute_error(actual, prediction))
            row["rmse_gbp"] = float(np.sqrt(mean_squared_error(actual, prediction)))
            row["r2"] = float(__import__("sklearn.metrics", fromlist=["r2_score"]).r2_score(actual, prediction)) if len(actual) > 1 else None
            row["status"] = "PASSED"
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors.append(row["error"])
        rows.append(row)
    return rows, errors


def _model_card(manifest: dict[str, Any], task: str) -> str:
    if task == "churn":
        target_description = (
            "1 if an eligible customer makes no qualifying purchase during the next "
            "30 days; 0 otherwise. This is an inactivity proxy, not verified cancellation."
        )
        evaluation = (
            f"Selection-window PR-AUC: {manifest.get('selected_selection_pr_auc')}; "
            f"CV PR-AUC mean: {manifest.get('cv_pr_auc_mean')}; "
            f"frozen threshold: {manifest.get('threshold')}."
        )
    else:
        target_description = (
            "Net recorded transaction-line value in the following 90 days, including "
            "recorded negative and cancellation lines. This is not profit or margin."
        )
        evaluation = (
            f"Selection-window MAE: {manifest.get('selected_selection_mae_gbp')} GBP; "
            f"CV MAE mean: {manifest.get('cv_mae_mean_gbp')} GBP."
        )
    return f"""# Model Card — {task.title()}

## Identity
- Model version: `{manifest.get('model_version')}`
- Run ID: `{manifest.get('run_id')}`
- Selected model: `{manifest.get('selected_model')}`
- Model status: `{manifest.get('status')}`
- Artifact SHA-256: `{manifest.get('artifact_sha256')}`

## Intended use
Analytical decision support for the Enterprise Customer Intelligence Platform. This artifact is not approved for automated high-impact customer treatment without business and risk review.

## Target
{target_description}

## Data and leakage controls
- Input snapshot SHA-256: `{manifest.get('data_sha256')}`
- Features come from Phase 2 snapshots and must be available before the prediction timestamp.
- IDs, target columns, timestamps, and split metadata are excluded from model features.
- Preprocessing is fitted inside each sklearn pipeline and each time-aware CV training fold.
- Chronological partitions and target-window boundaries are recorded in the split manifest.

## Evaluation
{evaluation}

## Limitations
- Online Retail II has no official customer churn label; churn is future inactivity under the documented rule.
- Future value is recorded transaction-line value, not profit or contribution margin.
- A feature's importance is not evidence of causality.
- Retention uplift needs a controlled experiment.
- Review subgroup performance, data quality, calibration, latency, and the complete comparison report before approval.

## Reproducibility
- Git commit: `{manifest.get('git_commit')}`
- Environment: `{json.dumps(manifest.get('environment', {}), sort_keys=True)}`
"""


def _classify_task(
    source_frame: pd.DataFrame,
    config: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
    source_path: Path,
    only_model: str | None = None,
    params_file: Path | None = None,
    full_model_coverage: bool = False,
) -> dict[str, Any]:
    report_dir = project_path(config["paths"]["reports_dir"])
    model_dir = project_path(config["paths"]["models_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if (report_dir / "test_evaluation_lock_churn.json").exists():
        raise RuntimeError(
            "The churn test set has already been evaluated. Start a new versioned experiment "
            "with a genuinely new test period before retraining."
        )

    target = config["supervised"]["churn_target"]
    task_frame, split_manifest = _write_churn_split_manifest(source_frame, config, source_path)
    json_dump(report_dir / "churn_task_split_manifest.json", split_manifest)
    task_frame = _safe_target_frame(task_frame, target, "churn")
    train = task_frame.loc[task_frame["split"].eq("train")].copy()
    validation = task_frame.loc[task_frame["split"].eq("validation")].copy()
    test = task_frame.loc[task_frame["split"].eq("test")].copy()

    _print_target_diagnostic(task_frame, target, "CHURN 30-DAY TARGET DISTRIBUTION AFTER SAFE SPLIT REASSIGNMENT")
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Churn needs non-empty train, validation, and test sets with matured 30-day labels. "
            f"Rows: train={len(train)}, validation={len(validation)}, test={len(test)}. "
            "Inspect reports/phase3/churn_task_split_manifest.json."
        )
    if train[target].nunique() < 2:
        raise _class_distribution_error(train, task_frame, target)

    selection_val, calibration_val, threshold_val, validation_roles = split_validation_roles(
        validation, config["supervised"]
    )
    if selection_val.empty:
        raise ValueError("The churn model-selection validation window is empty")
    if selection_val[target].nunique() < 2:
        raise ValueError(
            "The churn model-selection validation window has only one class, so PR-AUC cannot compare models reliably. "
            f"Counts: {selection_val[target].value_counts().to_dict()}. "
            "Use more validation snapshot dates or adjust the validation role fractions; do not use test labels."
        )

    params_model_name, params_override = _read_params_file(params_file, "churn")
    if params_model_name and only_model and params_model_name != only_model:
        raise ValueError(
            f"--only-model={only_model!r} does not match parameters-file model={params_model_name!r}"
        )
    if params_model_name and not only_model:
        only_model = params_model_name
    model_names = [only_model] if only_model else (
        CORE_CLASSIFIERS + OPTIONAL_CLASSIFIERS if full_model_coverage else CORE_CLASSIFIERS
    )

    seed = int(config["project"]["random_seed"])
    horizon = int(config["supervised"].get("churn_horizon_days", 30))
    feature_names = numeric + categorical
    X_train, y_train = train[feature_names], train[target].astype(int)
    X_sel, y_sel = selection_val[feature_names], selection_val[target].astype(int)
    comparison_rows: list[dict[str, Any]] = []
    fold_rows_all: list[dict[str, Any]] = []
    baseline_score: float | None = None

    for model_name in model_names:
        run_id = make_run_id()
        params = params_override if model_name == params_model_name else {}
        row: dict[str, Any] = {
            "run_id": run_id,
            "task": "churn",
            "model": model_name,
            "params": json.dumps(params, sort_keys=True),
            "status": "PENDING",
        }
        try:
            folds, cv_errors = _classification_cv(
                train, target, numeric, categorical, model_name, seed, horizon,
                int(config["supervised"].get("cv_max_folds", 3)), params,
            )
            for fold in folds:
                fold_rows_all.append({"run_id": run_id, "model": model_name, **fold})
            valid_folds = [f for f in folds if f.get("status") == "PASSED"]
            rankable_folds = [f for f in valid_folds if f.get("pr_auc_average_precision") is not None]
            row["cv_folds"] = len(valid_folds)
            row["cv_pr_auc_mean"] = float(np.mean([f["pr_auc_average_precision"] for f in rankable_folds])) if rankable_folds else None
            row["cv_pr_auc_std"] = float(np.std([f["pr_auc_average_precision"] for f in rankable_folds], ddof=1)) if len(rankable_folds) > 1 else (0.0 if rankable_folds else None)
            row["cv_roc_auc_mean"] = float(np.mean([f["roc_auc"] for f in valid_folds if f.get("roc_auc") is not None])) if any(f.get("roc_auc") is not None for f in valid_folds) else None
            model = build_pipeline("churn", model_name, numeric, categorical, seed, params)
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                raw_probability = _score_probability(model, X_sel)
            row["training_fit_ms"] = (time.perf_counter() - started) * 1000.0
            pm = _probability_metrics(y_sel.to_numpy(), raw_probability)
            row["selection_pr_auc"] = pm["pr_auc_average_precision"]
            row["selection_roc_auc"] = pm["roc_auc"]
            row["selection_brier_uncalibrated"] = pm["brier_score"]
            row["selection_positive_rate"] = pm["positive_rate"]
            # Operational ranking metrics at a provisional 0.5 threshold; final metrics use the frozen threshold.
            provisional = classification_metrics(
                y_sel.to_numpy(), raw_probability, 0.5, selection_val,
                float(config["supervised"].get("retention_capacity_fraction", 0.10)),
            )
            row["selection_recall_at_capacity"] = provisional.get("recall_at_capacity")
            row["selection_precision_among_contacted"] = provisional.get("precision_among_contacted")
            row.update(_latency_benchmark(model, X_sel, "churn", int(config["supervised"].get("latency_rows", 100))))
            row["artifact_size_estimate_bytes"] = _serialized_size(model)
            row["status"] = "PASSED"
            row["cv_errors"] = " | ".join(cv_errors)
            if model_name == "dummy":
                baseline_score = row["selection_pr_auc"]
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}: {exc}"
        comparison_rows.append(row)
        append_jsonl(report_dir / "experiments.jsonl", {
            **row,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_sha256": sha256_file(source_path),
            "git_commit": git_commit(),
            "environment": log_environment(),
        })
        print(
            f"[churn] {model_name}: {row.get('status')} "
            f"CV PR-AUC={row.get('cv_pr_auc_mean')} "
            f"selection PR-AUC={row.get('selection_pr_auc')}"
        )

    results = pd.DataFrame(comparison_rows)
    successful = results.loc[results["status"].eq("PASSED")].copy()
    if successful.empty or successful["selection_pr_auc"].notna().sum() == 0:
        raise RuntimeError("No churn classifier completed successfully; inspect the comparison rows and experiment log")
    if baseline_score is None:
        # AP of a constant-prior classifier is the positive prevalence for a rankable set.
        baseline_score = float(y_sel.mean())
    candidates = successful.loc[~successful["model"].eq("dummy")].copy()
    if candidates.empty:
        winner_row = successful.sort_values("selection_pr_auc", ascending=False).iloc[0]
    else:
        winner_row = candidates.sort_values(
            ["selection_pr_auc", "cv_pr_auc_mean"], ascending=[False, False], na_position="last"
        ).iloc[0]
    winner = str(winner_row["model"])
    winner_params = params_override if winner == params_model_name else {}
    winner_model = build_pipeline("churn", winner, numeric, categorical, seed, winner_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        winner_model.fit(X_train, y_train)
    selected_score = float(winner_row["selection_pr_auc"])
    beats_baseline = (
        winner != "dummy"
        and selected_score > float(baseline_score) + float(config["supervised"].get("minimum_candidate_improvement", 0.0005))
    )

    calibrator = PlattCalibrator()
    calibration_note: str
    if not calibration_val.empty and calibration_val[target].nunique() == 2:
        raw_calibration = _score_probability(winner_model, calibration_val[feature_names])
        calibrator.fit(raw_calibration, calibration_val[target].astype(int).to_numpy())
        calibration_note = calibrator.reason or "Platt sigmoid calibrator fitted on separate chronological validation dates."
    else:
        calibration_note = "Calibration window was empty or had one class; raw probability calibration retained."

    threshold_note = ""
    if threshold_val.empty:
        raise ValueError("The threshold-selection validation window is empty; do not select a threshold on the test set")
    if threshold_val[target].nunique() < 2:
        threshold_note = "Threshold window contains one class; common.select_threshold will apply its documented fallback."
    raw_threshold_probability = _score_probability(winner_model, threshold_val[feature_names])
    calibrated_threshold_probability = calibrator.predict(raw_threshold_probability)
    threshold, threshold_details = select_threshold(
        threshold_val[target].astype(int).to_numpy(), calibrated_threshold_probability
    )
    brier_raw = float(brier_score_loss(threshold_val[target].astype(int).to_numpy(), raw_threshold_probability))
    brier_calibrated = float(brier_score_loss(threshold_val[target].astype(int).to_numpy(), calibrated_threshold_probability))
    threshold_metrics = classification_metrics(
        threshold_val[target].astype(int).to_numpy(), calibrated_threshold_probability,
        float(threshold), threshold_val,
        float(config["supervised"].get("retention_capacity_fraction", 0.10)),
    )

    model_version = f"churn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{winner_row['run_id']}"
    artifact = model_dir / "churn_model.joblib"
    bundle = {
        "task": "churn",
        "model_name": winner,
        "model_params": winner_params,
        "model": winner_model,
        "calibrator": calibrator,
        "threshold": float(threshold),
        "model_version": model_version,
        "run_id": str(winner_row["run_id"]),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_names": feature_names,
        "target": target,
        "threshold_policy": threshold_details,
        "calibration_note": calibration_note,
        "retention_capacity_fraction": float(config["supervised"].get("retention_capacity_fraction", 0.10)),
        "validation_roles": validation_roles,
        "churn_split_manifest": split_manifest,
    }
    joblib.dump(bundle, artifact)
    artifact_hash = sha256_file(artifact)
    load_started = time.perf_counter()
    joblib.load(artifact)
    cold_load_ms = (time.perf_counter() - load_started) * 1000.0

    comparison_name = "churn_model_comparison.csv" if not only_model else f"churn_model_comparison_tuned_{winner}.csv"
    folds_name = "churn_cv_fold_metrics.csv" if not only_model else f"churn_cv_fold_metrics_tuned_{winner}.csv"
    results["selected_champion"] = results["model"].eq(winner)
    results["champion_beats_dummy_on_selection_window"] = results["model"].eq(winner) & beats_baseline
    results = results.sort_values("selection_pr_auc", ascending=False, na_position="last")
    results.to_csv(report_dir / comparison_name, index=False)
    pd.DataFrame(fold_rows_all).to_csv(report_dir / folds_name, index=False)

    feature_manifest_path = report_dir / "churn_feature_manifest.json"
    json_dump(feature_manifest_path, {
        "source_file": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256_file(source_path),
        "target": target,
        "target_horizon_days": horizon,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_order": feature_names,
        "excluded_columns": {
            col: "identifier, target/label, timestamp, split metadata, or non-approved dtype"
            for col in sorted(set(source_frame.columns) - set(feature_names))
        },
        "feature_time_contract": "Phase 2 features must use transaction history strictly before snapshot_date.",
    })
    manifest = {
        "task": "churn",
        "run_id": str(winner_row["run_id"]),
        "model_version": model_version,
        "selected_model": winner,
        "selected_params": winner_params,
        "artifact_path": str(artifact.relative_to(ROOT)),
        "artifact_sha256": artifact_hash,
        "target": target,
        "target_horizon_days": horizon,
        "data_sha256": sha256_file(source_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "split_manifest_sha256": sha256_file(report_dir / "churn_task_split_manifest.json"),
        "git_commit": git_commit(),
        "status": "candidate" if beats_baseline else "not_approved_baseline_not_beaten",
        "comparison_table": comparison_name,
        "selection_rule": "Best model-selection-window PR-AUC, supported by purged chronological CV metrics.",
        "dummy_selection_pr_auc": float(baseline_score),
        "selected_selection_pr_auc": selected_score,
        "beats_dummy_baseline": bool(beats_baseline),
        "cv_folds": int(winner_row.get("cv_folds", 0) or 0),
        "cv_pr_auc_mean": winner_row.get("cv_pr_auc_mean"),
        "cv_pr_auc_std": winner_row.get("cv_pr_auc_std"),
        "threshold": float(threshold),
        "threshold_policy": threshold_details,
        "retention_capacity_fraction": float(config["supervised"].get("retention_capacity_fraction", 0.10)),
        "calibration_method": "Platt sigmoid" if getattr(calibrator, "fitted", False) else "identity/raw probabilities",
        "calibration_note": calibration_note,
        "threshold_note": threshold_note,
        "brier_raw_threshold_window": brier_raw,
        "brier_calibrated_threshold_window": brier_calibrated,
        "threshold_window_metrics": threshold_metrics,
        "cold_load_ms": cold_load_ms,
        "artifact_size_bytes": int(artifact.stat().st_size),
        "environment": log_environment(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": False,
        "retention_value": "not measurable from observational transaction rows; requires a controlled intervention experiment",
    }
    json_dump(report_dir / "churn_model_manifest.json", manifest)
    (report_dir / "model_card_churn.md").write_text(_model_card(manifest, "churn"), encoding="utf-8")
    append_jsonl(report_dir / "experiments.jsonl", {
        "run_type": "selected_artifact_manifest",
        **manifest,
        "data_version": sha256_file(source_path),
        "feature_version": sha256_file(feature_manifest_path),
        "split_version": sha256_file(report_dir / "churn_task_split_manifest.json"),
    })
    print(f"\nSelected churn candidate: {winner}")
    print(f"Beats dummy on selection validation PR-AUC: {beats_baseline}")
    print(f"Task-specific split report: {report_dir / 'churn_task_split_manifest.json'}")
    print(f"Artifact: {artifact}")
    if not beats_baseline:
        print("WARNING: Selected candidate did not beat the dummy baseline; it is not approved for final test evaluation.")
    return manifest


def _value_task(
    source_frame: pd.DataFrame,
    config: dict[str, Any],
    numeric: list[str],
    categorical: list[str],
    source_path: Path,
    only_model: str | None = None,
    params_file: Path | None = None,
    full_model_coverage: bool = False,
) -> dict[str, Any]:
    report_dir = project_path(config["paths"]["reports_dir"])
    model_dir = project_path(config["paths"]["models_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if (report_dir / "test_evaluation_lock_value.json").exists():
        raise RuntimeError(
            "The value test set has already been evaluated. Start a new versioned experiment "
            "with a genuinely new test period before retraining."
        )

    target = config["supervised"]["value_target"]
    df = _safe_target_frame(source_frame.loc[source_frame["split"].isin(["train", "validation", "test"])].copy(), target, "value")
    if "target_value_valid" in df.columns:
        df = df.loc[_bool_series(df["target_value_valid"])].copy()
    train = df.loc[df["split"].eq("train")].copy()
    validation = df.loc[df["split"].eq("validation")].copy()
    test = df.loc[df["split"].eq("test")].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Value regression requires non-empty train/validation/test sets with mature 90-day value labels. "
            f"Rows: train={len(train)}, validation={len(validation)}, test={len(test)}."
        )

    selection_val, _, _, validation_roles = split_validation_roles(validation, config["supervised"])
    if selection_val.empty:
        raise ValueError("The value model-selection validation window is empty")
    params_model_name, params_override = _read_params_file(params_file, "value")
    if params_model_name and only_model and params_model_name != only_model:
        raise ValueError(
            f"--only-model={only_model!r} does not match parameters-file model={params_model_name!r}"
        )
    if params_model_name and not only_model:
        only_model = params_model_name
    model_names = [only_model] if only_model else (
        CORE_REGRESSORS + OPTIONAL_REGRESSORS if full_model_coverage else CORE_REGRESSORS
    )

    seed = int(config["project"]["random_seed"])
    horizon = int(config["supervised"].get("value_horizon_days", config["supervised"].get("horizon_days", 90)))
    feature_names = numeric + categorical
    X_train, y_train = train[feature_names], train[target].astype(float)
    X_sel, y_sel = selection_val[feature_names], selection_val[target].astype(float)
    comparison_rows: list[dict[str, Any]] = []
    fold_rows_all: list[dict[str, Any]] = []
    baseline_mae: float | None = None

    for model_name in model_names:
        run_id = make_run_id()
        params = params_override if model_name == params_model_name else {}
        row: dict[str, Any] = {
            "run_id": run_id,
            "task": "value",
            "model": model_name,
            "params": json.dumps(params, sort_keys=True),
            "status": "PENDING",
        }
        try:
            folds, cv_errors = _regression_cv(
                train, target, numeric, categorical, model_name, seed, horizon,
                int(config["supervised"].get("cv_max_folds", 3)), params,
            )
            for fold in folds:
                fold_rows_all.append({"run_id": run_id, "model": model_name, **fold})
            valid_folds = [f for f in folds if f.get("status") == "PASSED"]
            row["cv_folds"] = len(valid_folds)
            row["cv_mae_mean_gbp"] = float(np.mean([f["mae_gbp"] for f in valid_folds])) if valid_folds else None
            row["cv_mae_std_gbp"] = float(np.std([f["mae_gbp"] for f in valid_folds], ddof=1)) if len(valid_folds) > 1 else (0.0 if valid_folds else None)
            row["cv_rmse_mean_gbp"] = float(np.mean([f["rmse_gbp"] for f in valid_folds])) if valid_folds else None
            model = build_pipeline("value", model_name, numeric, categorical, seed, params)
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
                prediction = model.predict(X_sel)
            row["training_fit_ms"] = (time.perf_counter() - started) * 1000.0
            metrics = regression_metrics(y_sel.to_numpy(), prediction, y_train.to_numpy())
            row["selection_mae_gbp"] = metrics["mae_gbp"]
            row["selection_rmse_gbp"] = metrics["rmse_gbp"]
            row["selection_mse_gbp_squared"] = metrics.get("mse_gbp_squared")
            row["selection_r2"] = metrics.get("r2")
            row["selection_mape_pct_nonzero_actual"] = metrics.get("mape_pct_nonzero_actual")
            row["selection_mape_nonzero_actual_count"] = metrics.get("mape_nonzero_actual_count")
            row["selection_value_tier_metrics"] = json.dumps(metrics.get("error_by_value_tier", {}), sort_keys=True)
            allocation = metrics.get("top_decile_allocation", {})
            row["selection_top_decile_actual_value_gbp"] = allocation.get("actual_net_value_in_top_predicted_decile_gbp")
            row["selection_top_decile_fraction_total_value"] = allocation.get("fraction_of_total_net_value_captured")
            row.update(_latency_benchmark(model, X_sel, "value", int(config["supervised"].get("latency_rows", 100))))
            row["artifact_size_estimate_bytes"] = _serialized_size(model)
            row["status"] = "PASSED"
            row["cv_errors"] = " | ".join(cv_errors)
            if model_name == "dummy_median":
                baseline_mae = float(row["selection_mae_gbp"])
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}: {exc}"
        comparison_rows.append(row)
        append_jsonl(report_dir / "experiments.jsonl", {
            **row,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_sha256": sha256_file(source_path),
            "git_commit": git_commit(),
            "environment": log_environment(),
        })
        print(
            f"[value] {model_name}: {row.get('status')} "
            f"CV MAE={row.get('cv_mae_mean_gbp')} "
            f"selection MAE={row.get('selection_mae_gbp')}"
        )

    results = pd.DataFrame(comparison_rows)
    successful = results.loc[results["status"].eq("PASSED")].copy()
    if successful.empty or successful["selection_mae_gbp"].notna().sum() == 0:
        raise RuntimeError("No value regressor completed successfully; inspect comparison rows and experiment log")
    if baseline_mae is None:
        # Reproduce the median baseline using training data only.
        median_prediction = float(y_train.median())
        baseline_mae = float(mean_absolute_error(y_sel, np.full(len(y_sel), median_prediction)))
    candidates = successful.loc[~successful["model"].eq("dummy_median")].copy()
    if candidates.empty:
        winner_row = successful.sort_values("selection_mae_gbp", ascending=True).iloc[0]
    else:
        winner_row = candidates.sort_values(
            ["selection_mae_gbp", "cv_mae_mean_gbp"], ascending=[True, True], na_position="last"
        ).iloc[0]
    winner = str(winner_row["model"])
    winner_params = params_override if winner == params_model_name else {}
    winner_model = build_pipeline("value", winner, numeric, categorical, seed, winner_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        winner_model.fit(X_train, y_train)
    selected_mae = float(winner_row["selection_mae_gbp"])
    min_improvement = float(config["supervised"].get("minimum_candidate_improvement", 0.0005))
    beats_baseline = winner != "dummy_median" and selected_mae < float(baseline_mae) - min_improvement

    model_version = f"value-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{winner_row['run_id']}"
    artifact = model_dir / "value_model.joblib"
    bundle = {
        "task": "value",
        "model_name": winner,
        "model_params": winner_params,
        "model": winner_model,
        "model_version": model_version,
        "run_id": str(winner_row["run_id"]),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_names": feature_names,
        "target": target,
        "validation_roles": validation_roles,
    }
    joblib.dump(bundle, artifact)
    artifact_hash = sha256_file(artifact)
    load_started = time.perf_counter()
    joblib.load(artifact)
    cold_load_ms = (time.perf_counter() - load_started) * 1000.0

    comparison_name = "value_model_comparison.csv" if not only_model else f"value_model_comparison_tuned_{winner}.csv"
    folds_name = "value_cv_fold_metrics.csv" if not only_model else f"value_cv_fold_metrics_tuned_{winner}.csv"
    results["selected_champion"] = results["model"].eq(winner)
    results["champion_beats_dummy_on_selection_window"] = results["model"].eq(winner) & beats_baseline
    results = results.sort_values("selection_mae_gbp", ascending=True, na_position="last")
    results.to_csv(report_dir / comparison_name, index=False)
    pd.DataFrame(fold_rows_all).to_csv(report_dir / folds_name, index=False)

    feature_manifest_path = report_dir / "value_feature_manifest.json"
    json_dump(feature_manifest_path, {
        "source_file": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256_file(source_path),
        "target": target,
        "target_horizon_days": horizon,
        "currency": "GBP",
        "numeric_features": numeric,
        "categorical_features": categorical,
        "feature_order": feature_names,
        "excluded_columns": {
            col: "identifier, target/label, timestamp, split metadata, or non-approved dtype"
            for col in sorted(set(source_frame.columns) - set(feature_names))
        },
        "feature_time_contract": "Phase 2 features must use transaction history strictly before snapshot_date.",
    })
    manifest = {
        "task": "value",
        "run_id": str(winner_row["run_id"]),
        "model_version": model_version,
        "selected_model": winner,
        "selected_params": winner_params,
        "artifact_path": str(artifact.relative_to(ROOT)),
        "artifact_sha256": artifact_hash,
        "target": target,
        "target_horizon_days": horizon,
        "currency": "GBP",
        "data_sha256": sha256_file(source_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "git_commit": git_commit(),
        "status": "candidate" if beats_baseline else "not_approved_baseline_not_beaten",
        "comparison_table": comparison_name,
        "selection_rule": "Lowest model-selection-window MAE, supported by purged chronological CV metrics.",
        "dummy_selection_mae_gbp": float(baseline_mae),
        "selected_selection_mae_gbp": selected_mae,
        "beats_dummy_baseline": bool(beats_baseline),
        "cv_folds": int(winner_row.get("cv_folds", 0) or 0),
        "cv_mae_mean_gbp": winner_row.get("cv_mae_mean_gbp"),
        "cv_mae_std_gbp": winner_row.get("cv_mae_std_gbp"),
        "cold_load_ms": cold_load_ms,
        "artifact_size_bytes": int(artifact.stat().st_size),
        "environment": log_environment(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_evaluated": False,
    }
    json_dump(report_dir / "value_model_manifest.json", manifest)
    (report_dir / "model_card_value.md").write_text(_model_card(manifest, "value"), encoding="utf-8")
    append_jsonl(report_dir / "experiments.jsonl", {
        "run_type": "selected_artifact_manifest",
        **manifest,
        "data_version": sha256_file(source_path),
        "feature_version": sha256_file(feature_manifest_path),
    })
    print(f"\nSelected value candidate: {winner}")
    print(f"Beats dummy on selection validation MAE: {beats_baseline}")
    print(f"Artifact: {artifact}")
    if not beats_baseline:
        print("WARNING: Selected candidate did not beat the dummy baseline; it is not approved for final test evaluation.")
    return manifest


def _evaluate_test(task: str, config: dict[str, Any], allow_repeat: bool = False) -> dict[str, Any]:
    report_dir = project_path(config["paths"]["reports_dir"])
    model_dir = project_path(config["paths"]["models_dir"])
    artifact_path = model_dir / ("churn_model.joblib" if task == "churn" else "value_model.joblib")
    manifest_path = report_dir / f"{task}_model_manifest.json"
    metrics_path = report_dir / f"test_{task}_metrics.json"
    lock_path = report_dir / f"test_evaluation_lock_{task}.json"

    if lock_path.exists() and not allow_repeat:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"The {task} test set was already evaluated at {lock.get('evaluated_at_utc')}. "
            "Do not use it to choose models or thresholds."
        )
    if not artifact_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"Missing {task} model artifact or manifest; train candidates first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate":
        raise RuntimeError(
            f"The {task} model has status={manifest.get('status')!r} and did not beat its dummy baseline. "
            "Resolve model/label problems before final test evaluation."
        )
    if sha256_file(artifact_path) != manifest.get("artifact_sha256"):
        raise RuntimeError("Model artifact checksum does not match the saved model manifest")

    frame, source_path = load_snapshots()
    bundle = joblib.load(artifact_path)
    target = bundle["target"]
    test = frame.loc[frame["split"].eq("test") & frame[target].notna()].copy()
    if task == "value" and "target_value_valid" in test.columns:
        test = test.loc[_bool_series(test["target_value_valid"])].copy()
    if task == "churn" and "target_churn_30d_matured" in test.columns:
        test = test.loc[_bool_series(test["target_churn_30d_matured"])].copy()
    if test.empty:
        raise ValueError(f"No valid held-out test rows exist for {task}")
    feature_names = bundle["feature_names"]
    missing = [col for col in feature_names if col not in test.columns]
    if missing:
        raise ValueError(f"Test data is missing features expected by the model: {missing}")
    X = test[feature_names]
    task_started = time.perf_counter()

    if task == "churn":
        y = test[target].astype(int).to_numpy()
        raw_probability = _score_probability(bundle["model"], X)
        calibrated_probability = bundle["calibrator"].predict(raw_probability)
        threshold = float(bundle["threshold"])
        metrics = classification_metrics(
            y, calibrated_probability, threshold, test,
            float(bundle.get("retention_capacity_fraction", 0.10)),
        )
        predicted = (calibrated_probability >= threshold).astype(int)
        predictions = test[["customer_id", "snapshot_date"]].copy()
        predictions["actual_churn_proxy"] = y
        predictions["raw_probability"] = raw_probability
        predictions["calibrated_probability"] = calibrated_probability
        predictions["decision"] = predicted
        predictions["threshold"] = threshold
        predictions["model_version"] = bundle["model_version"]
        predictions["error_type"] = np.select(
            [(y == 1) & (predicted == 1), (y == 0) & (predicted == 1), (y == 1) & (predicted == 0)],
            ["TP", "FP", "FN"], default="TN",
        )
        if "country" in test.columns:
            predictions["country"] = test["country"].astype("string").to_numpy()
            subgroup = []
            for country, group in predictions.groupby("country", dropna=False):
                tp = int(((group["actual_churn_proxy"] == 1) & (group["decision"] == 1)).sum())
                fp = int(((group["actual_churn_proxy"] == 0) & (group["decision"] == 1)).sum())
                fn = int(((group["actual_churn_proxy"] == 1) & (group["decision"] == 0)).sum())
                subgroup.append({
                    "country": str(country), "n": int(len(group)),
                    "positive_rate": float(group["actual_churn_proxy"].mean()),
                    "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                    "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                })
            pd.DataFrame(subgroup).to_csv(report_dir / "test_churn_country_metrics.csv", index=False)
            metrics["country_subgroup_metrics_file"] = "reports/phase3/test_churn_country_metrics.csv"
        predictions.to_csv(report_dir / "test_churn_predictions.csv", index=False)
    else:
        y = test[target].astype(float).to_numpy()
        prediction = np.asarray(bundle["model"].predict(X), dtype=float)
        training_y = frame.loc[frame["split"].eq("train") & frame[target].notna(), target].astype(float).to_numpy()
        metrics = regression_metrics(y, prediction, training_y)
        predictions = test[["customer_id", "snapshot_date"]].copy()
        predictions["actual_net_value_90d_gbp"] = y
        predictions["predicted_net_value_90d_gbp"] = prediction
        predictions["residual_gbp"] = y - prediction
        predictions["model_version"] = bundle["model_version"]
        if "country" in test.columns:
            predictions["country"] = test["country"].astype("string").to_numpy()
            predictions.groupby("country", dropna=False).agg(
                n=("residual_gbp", "size"),
                mae_gbp=("residual_gbp", lambda values: float(np.mean(np.abs(values)))),
                rmse_gbp=("residual_gbp", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            ).reset_index().to_csv(report_dir / "test_value_country_metrics.csv", index=False)
            metrics["country_subgroup_metrics_file"] = "reports/phase3/test_value_country_metrics.csv"
        predictions.to_csv(report_dir / "test_value_predictions.csv", index=False)

    metrics.update({
        "task": task,
        "model_version": bundle["model_version"],
        "run_id": bundle["run_id"],
        "artifact_sha256": sha256_file(artifact_path),
        "data_sha256": sha256_file(source_path),
        "test_rows": int(len(test)),
        "test_class_counts": {str(k): int(v) for k, v in test[target].value_counts(dropna=False).items()} if task == "churn" else None,
        "test_snapshot_dates": sorted(pd.to_datetime(test["snapshot_date"]).dt.date.astype(str).unique().tolist()),
        "evaluation_runtime_seconds": round(time.perf_counter() - task_started, 4),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_used_for_final_evaluation_only": True,
    })
    json_dump(metrics_path, metrics)

    # Add test metrics to the selected model's comparison row for one comparable table.
    comparison_path = report_dir / manifest.get(
        "comparison_table", "churn_model_comparison.csv" if task == "churn" else "value_model_comparison.csv"
    )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        if task == "churn":
            values = {
                "test_pr_auc": metrics.get("pr_auc_average_precision"),
                "test_roc_auc": metrics.get("roc_auc"),
                "test_recall_at_capacity": metrics.get("recall_at_capacity"),
                "test_precision_among_contacted": metrics.get("precision_among_contacted"),
                "test_brier_score": metrics.get("brier_score"),
                "frozen_threshold": bundle["threshold"],
            }
        else:
            allocation = metrics.get("top_decile_allocation", {})
            values = {
                "test_mae_gbp": metrics.get("mae_gbp"),
                "test_rmse_gbp": metrics.get("rmse_gbp"),
                "test_r2": metrics.get("r2"),
                "test_mse_gbp_squared": metrics.get("mse_gbp_squared"),
                "test_mape_pct_nonzero_actual": metrics.get("mape_pct_nonzero_actual"),
                "test_top_decile_actual_value_gbp": allocation.get("actual_net_value_in_top_predicted_decile_gbp"),
                "test_top_decile_fraction_total_value": allocation.get("fraction_of_total_net_value_captured"),
            }
        for column, value in values.items():
            if column not in comparison.columns:
                comparison[column] = np.nan
            comparison.loc[comparison["model"].eq(bundle["model_name"]), column] = value
        comparison.to_csv(comparison_path, index=False)

    lock = {
        "task": task,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": sha256_file(artifact_path),
        "metrics_file": str(metrics_path.relative_to(ROOT)),
        "warning": "Do not use test results to choose models, features, hyperparameters, threshold, or calibration.",
    }
    json_dump(lock_path, lock)
    manifest["test_evaluated"] = True
    manifest["test_metrics_file"] = str(metrics_path.relative_to(ROOT))
    manifest["test_evaluated_at_utc"] = lock["evaluated_at_utc"]
    json_dump(manifest_path, manifest)
    print(f"\nFINAL TEST — {task.upper()}")
    print(json.dumps(metrics, indent=2, default=str))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["churn", "value", "all"], default="all")
    parser.add_argument("--only-model", help="Fit/compare only one named model")
    parser.add_argument("--params-file", type=Path, help="JSON file from tune_optuna.py")
    parser.add_argument("--full-model-coverage", action="store_true", help="Also compare optional checklist models; this takes longer")
    parser.add_argument("--evaluate-test", action="store_true", help="Run the final held-out evaluation after choices are frozen")
    parser.add_argument("--allow-repeat-test", action="store_true", help="Override the test lock only for documented debugging/research reasons")
    args = parser.parse_args()

    if args.only_model and args.task == "all":
        parser.error("Use --task churn or --task value with --only-model so model names/parameters are unambiguous")
    if args.params_file and args.task == "all":
        parser.error("Use --task churn or --task value with --params-file")

    config = load_config()
    report_dir = project_path(config["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_test:
        tasks = ["churn", "value"] if args.task == "all" else [args.task]
        for task in tasks:
            _evaluate_test(task, config, args.allow_repeat_test)
        return

    frame, source_path = load_snapshots()
    target_names = [config["supervised"]["churn_target"], config["supervised"]["value_target"]]
    for target in target_names:
        if target not in frame.columns:
            raise ValueError(f"Required target column {target!r} was not found in {source_path}")

    numeric, categorical, excluded = feature_columns(frame)
    split_manifest_path = report_dir / "split_manifest.json"
    try:
        split_manifest = persist_split_membership(
            frame,
            project_path(config["paths"]["splits_file"]),
            int(config["supervised"].get("value_horizon_days", config["supervised"].get("horizon_days", 90))),
        )
        json_dump(split_manifest_path, split_manifest)
    except Exception as exc:
        raise RuntimeError(
            "Could not validate the original Phase 2 train/validation/test boundaries. "
            "Inspect Phase 2's split dates and outcome windows before training. "
            f"Details: {type(exc).__name__}: {exc}"
        ) from exc

    json_dump(report_dir / "feature_manifest.json", {
        "source_path": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256_file(source_path),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "excluded_columns": excluded,
        "feature_order": numeric + categorical,
        "preprocessing_policy": "Imputation, scaling and encoding are fitted inside each model pipeline and each CV training fold.",
        "target_leakage_check": "target_* and label_* columns, IDs, timestamps, and split metadata are excluded from features.",
    })

    tasks = ["churn", "value"] if args.task == "all" else [args.task]
    for task in tasks:
        if task == "churn":
            _classify_task(
                frame, config, numeric, categorical, source_path,
                args.only_model, args.params_file, args.full_model_coverage,
            )
        else:
            _value_task(
                frame, config, numeric, categorical, source_path,
                args.only_model, args.params_file, args.full_model_coverage,
            )
    print("\nSupervised comparison completed. The final test set has NOT been evaluated.")
    print("Review the model-comparison CSVs, tune a candidate if justified, and freeze decisions before --evaluate-test.")


if __name__ == "__main__":
    main()
