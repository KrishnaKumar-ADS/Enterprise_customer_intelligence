from __future__ import annotations


# --- Consolidated from test_customer_datasets.py ---

"""Tests for time-safe customer dataset construction."""

import pandas as pd
import pytest

from src.data.schema import normalize_transactions
from src.data.build_customer_datasets import (
    assign_split,
    build_anomaly_transactions,
    build_customer_snapshots,
)


@pytest.fixture
def tiny_transactions() -> pd.DataFrame:
    """Controlled fixture for time-window and target logic."""
    raw = pd.DataFrame({
        "InvoiceNo": ["A001", "A002", "CA003", "B001"],
        "StockCode": ["P1", "P1", "P1", "P2"],
        "Description": [
            "Product 1", "Product 1", "Product 1", "Product 2"
        ],
        "Quantity": [2, 1, -1, 1],
        "InvoiceDate": [
            "15/02/2020 10:00",
            "15/06/2020 10:00",
            "15/07/2020 10:00",
            "01/02/2020 11:00",
        ],
        "UnitPrice": [10.0, 10.0, 10.0, 5.0],
        "CustomerID": ["A", "A", "A", "B"],
        "Country": [
            "United Kingdom",
            "United Kingdom",
            "United Kingdom",
            "France",
        ],
    })
    return normalize_transactions(raw)


def test_split_assignment() -> None:
    cfg = {
        "train_start": "2010-06-01",
        "train_end_exclusive": "2010-11-01",
        "validation_start": "2011-01-01",
        "validation_end_exclusive": "2011-05-01",
        "test_start": "2011-07-01",
    }

    assert assign_split(pd.Timestamp("2010-06-01"), cfg) == "train"
    assert assign_split(pd.Timestamp("2010-11-01"), cfg) == "purged_gap"
    assert assign_split(pd.Timestamp("2011-02-01"), cfg) == "validation"
    assert assign_split(pd.Timestamp("2011-06-01"), cfg) == "purged_gap"
    assert assign_split(pd.Timestamp("2011-07-01"), cfg) == "test"


def test_customer_snapshot_targets(
    tiny_transactions: pd.DataFrame,
) -> None:
    phase2_cfg = {
        "snapshot": {
            "history_days": 180,
            "horizon_days": 90,
        },
        "chronological_splits": {
            "train_start": "2020-06-01",
            "train_end_exclusive": "2020-11-01",
            "validation_start": "2021-01-01",
            "validation_end_exclusive": "2021-05-01",
            "test_start": "2021-07-01",
        },
    }

    result = build_customer_snapshots(
        tiny_transactions,
        [pd.Timestamp("2020-06-01")],
        phase2_cfg,
    ).set_index("customer_id")

    # A purchases during the future window.
    assert int(result.loc["A", "target_churn_90d"]) == 0

    # A's future sale of GBP 10 is offset by a negative line of GBP 10.
    assert result.loc[
        "A", "target_net_value_90d_gbp"
    ] == pytest.approx(0.0)

    # B has historical activity but no qualifying future purchase.
    assert int(result.loc["B", "target_churn_90d"]) == 1
    assert result.loc[
        "B", "target_net_value_90d_gbp"
    ] == pytest.approx(0.0)

    assert result.index.is_unique


def test_features_are_from_before_snapshot_only(
    tiny_transactions: pd.DataFrame,
) -> None:
    phase2_cfg = {
        "snapshot": {
            "history_days": 180,
            "horizon_days": 90,
        },
        "chronological_splits": {
            "train_start": "2020-06-01",
            "train_end_exclusive": "2020-11-01",
            "validation_start": "2021-01-01",
            "validation_end_exclusive": "2021-05-01",
            "test_start": "2021-07-01",
        },
    }

    result = build_customer_snapshots(
        tiny_transactions,
        [pd.Timestamp("2020-06-01")],
        phase2_cfg,
    ).set_index("customer_id")

    # The June 15 purchase is in the future, not historical spend.
    assert result.loc["A", "spend_180d_gbp"] == pytest.approx(20.0)


def test_anomaly_dataset_preserves_transaction_rows(
    tiny_transactions: pd.DataFrame,
) -> None:
    result = build_anomaly_transactions(tiny_transactions)

    assert len(result) == len(tiny_transactions)
    assert "invoice_line_count" in result.columns
    assert "invoice_net_value_gbp" in result.columns
    assert "invoice_hour" in result.columns


# --- Consolidated from test_phase_3_core.py ---

import numpy as np
import pandas as pd

from src.training.common import (
    PlattCalibrator, classification_metrics, feature_columns,
    purged_walk_forward_folds, select_threshold,
)


def test_target_columns_are_excluded_from_features():
    frame = pd.DataFrame({
        "customer_id": ["A", "B"],
        "snapshot_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
        "split": ["train", "validation"],
        "target_churn_90d": [0, 1],
        "target_value_valid": [True, True],
        "orders_90d": [3, 2],
        "country": ["UK", None],
    })
    numeric, categorical, excluded = feature_columns(frame)
    assert numeric == ["orders_90d"]
    assert categorical == ["country"]
    assert "target_churn_90d" in excluded
    assert "customer_id" in excluded


def test_purged_folds_do_not_overlap_target_horizon():
    dates = pd.to_datetime([
        "2010-06-01", "2010-07-01", "2010-08-01",
        "2010-09-01", "2010-10-01",
    ])
    frame = pd.DataFrame({
        "snapshot_date": np.repeat(dates, 2),
        "target": np.tile([0, 1], len(dates)),
    })
    folds = purged_walk_forward_folds(frame, horizon_days=90, max_folds=3)
    assert folds
    for train_idx, val_idx, train_end, val_date in folds:
        assert pd.Timestamp(train_end) + pd.Timedelta(days=90) <= pd.Timestamp(val_date)
        assert set(train_idx).isdisjoint(set(val_idx))


def test_platt_calibrator_and_threshold():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.25, 0.6, 0.7, 0.8, 0.3, 0.65])
    cal = PlattCalibrator().fit(p, y)
    calibrated = cal.predict(p)
    assert calibrated.shape == p.shape
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    threshold, detail = select_threshold(y, calibrated)
    assert 0 <= threshold <= 1
    assert "method" in detail


def test_classification_capacity_metrics_are_reported():
    y = np.array([1, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    frame = pd.DataFrame({
        "snapshot_date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-02-01"] * 3)
    })
    metrics = classification_metrics(y, p, 0.5, frame, 0.34)
    assert "pr_auc_average_precision" in metrics
    assert "recall_at_capacity" in metrics
    assert "precision_among_contacted" in metrics


def test_pdf_aligned_30_day_churn_target_is_derived_without_future_feature_leakage():
    from src.training.modeling_pipeline import build_churn_30d_labels

    snapshots = pd.DataFrame({
        "customer_id": ["A", "B"],
        "snapshot_date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        "split": ["train", "train"],
        "target_net_value_90d_gbp": [10.0, 0.0],
    })
    transactions = pd.DataFrame({
        "customer_id": ["A", "A", "B", "A"],
        "invoice_no": ["I1", "I2", "I3", "I4"],
        "invoice_date": pd.to_datetime([
            "2019-12-20 10:00", "2020-01-15 10:00", "2019-12-10 11:00",
            "2020-02-01 12:00",
        ]),
        "quantity": [1, 1, 2, 1],
        "unit_price": [10.0, 10.0, 5.0, 10.0],
        "is_cancellation": [False, False, False, False],
    })
    result, cutoff = build_churn_30d_labels(snapshots, transactions, 30)
    values = result.set_index("customer_id")["target_churn_30d"].to_dict()
    assert values["A"] == 0
    assert values["B"] == 1
    assert result["target_churn_30d_matured"].all()
    assert len(result) == len(snapshots)
    assert cutoff == pd.Timestamp("2020-02-02")


# --- Consolidated from test_phase_4.py ---

import numpy as np
import pandas as pd

from src.training.modeling_pipeline import (
    comparison_classification_metrics,
    parse_params,
    comparison_regression_metrics,
    _standardize_comparison,
)


def test_parse_params_valid_and_missing() -> None:
    assert parse_params('{"max_depth": 4}') == {"max_depth": 4}
    assert parse_params(None) == {}
    assert parse_params("invalid") == {}


def test_classification_metrics_confusion_and_capacity() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.8, 0.7, 0.2])
    metrics = comparison_classification_metrics(y, score, 0.5, 0.5)
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["capacity_selected_count"] == 2
    assert 0 <= metrics["pr_auc_average_precision"] <= 1


def test_regression_metrics_and_top_decile() -> None:
    metrics = comparison_regression_metrics(np.array([10.0, 20.0, 30.0, 40.0]), np.array([11.0, 18.0, 35.0, 37.0]))
    assert metrics["mae_gbp"] == 2.75
    assert metrics["rmse_gbp"] > 0
    assert metrics["top_predicted_decile_actual_value_gbp"] is not None


def test_comparison_table_never_selects_from_test_metrics() -> None:
    phase3 = pd.DataFrame([
        {"model": "dummy", "status": "PASSED", "params": "{}", "selection_pr_auc": 0.20, "cv_pr_auc_mean": 0.21, "test_pr_auc": 0.99},
        {"model": "logistic_regression", "status": "PASSED", "params": "{}", "selection_pr_auc": 0.35, "cv_pr_auc_mean": 0.33, "test_pr_auc": 0.10},
    ])
    phase4 = pd.DataFrame([
        {"model": "dummy", "status": "PASSED", "validation_pr_auc_average_precision": 0.20, "fold_pr_auc_mean": 0.21, "fold_pr_auc_std": 0.01},
        {"model": "logistic_regression", "status": "PASSED", "validation_pr_auc_average_precision": 0.35, "fold_pr_auc_mean": 0.33, "fold_pr_auc_std": 0.02},
    ])
    manifest = {"selected_model": "logistic_regression"}
    result = _standardize_comparison("churn", phase3, phase4, manifest)
    assert "test_pr_auc" not in result.columns
    assert bool(result.loc[result["model"].eq("logistic_regression"), "matches_current_phase3_artifact"].iloc[0])
    assert int(result.loc[result["model"].eq("logistic_regression"), "phase4_validation_rank"].iloc[0]) == 1
    assert bool(result.loc[result["model"].eq("logistic_regression"), "beats_dummy_on_phase4_diagnostic"].iloc[0])


# --- Consolidated from test_phase_5.py ---

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.training.modeling_pipeline import (
    _numeric_target,
    _parse_params,
    _ranked_phase4_candidates,
    _score_model,
)


def test_parse_params_accepts_json_and_dict() -> None:
    assert _parse_params('{"C": 0.5}') == {"C": 0.5}
    assert _parse_params({"alpha": 2}) == {"alpha": 2}
    assert _parse_params(None) == {}


def test_numeric_target_filters_nan_and_binary_invalid_rows() -> None:
    frame = pd.DataFrame({
        "target_churn_30d": [0, 1, 2, np.nan],
        "split": ["train"] * 4,
    })
    result = _numeric_target(frame, "target_churn_30d", "churn")
    assert result["target_churn_30d"].tolist() == [0, 1]


def test_numeric_target_filters_invalid_value_rows() -> None:
    frame = pd.DataFrame({
        "target_net_value_90d_gbp": [1.0, 2.0, np.nan],
        "target_value_valid": [True, False, True],
    })
    result = _numeric_target(frame, "target_net_value_90d_gbp", "value")
    assert result["target_net_value_90d_gbp"].tolist() == [1.0]


def test_churn_scoring_returns_ap_and_brier() -> None:
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression().fit(X, y)
    metrics = _score_model("churn", model, X, y)
    assert "selection_pr_auc" in metrics
    assert "selection_brier_uncalibrated" in metrics
    assert 0 <= metrics["selection_pr_auc"] <= 1


def test_value_scoring_returns_currency_metrics() -> None:
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
    y = np.array([0, 2, 4, 6, 8, 10], dtype=float)
    model = LinearRegression().fit(X, y)
    metrics = _score_model("value", model, X, y)
    assert metrics["selection_mae_gbp"] < 1e-8
    assert metrics["selection_rmse_gbp"] < 1e-8


def test_comparison_ranks_higher_churn_metric_first() -> None:
    frame = pd.DataFrame({
        "task": ["churn", "churn", "churn"],
        "model": ["dummy", "logistic_regression", "random_forest"],
        "status": ["PASSED"] * 3,
        "phase4_diagnostic_status": ["PASSED"] * 3,
        "phase4_validation_rank": [2, 1, 3],
    })
    result = _ranked_phase4_candidates(frame, "churn")
    assert result.iloc[0]["model"] == "logistic_regression"


# --- Consolidated from test_phase_5_search_spaces.py ---

import optuna

from src.training.modeling_pipeline import resolve_params, suggest_params
from src.training.modeling_pipeline import TUNABLE_MODELS


def test_all_phase3_classifier_families_have_tuning_spaces() -> None:
    expected = {
        "logistic_regression", "decision_tree", "random_forest", "extra_trees",
        "adaboost", "gradient_boosting", "hist_gradient_boosting", "svm",
        "svm_rbf", "knn", "naive_bayes", "voting", "xgboost", "lightgbm", "catboost",
    }
    assert expected.issubset(TUNABLE_MODELS["churn"])


def test_all_phase3_regressor_families_have_tuning_spaces() -> None:
    expected = {
        "ridge", "lasso", "elastic_net", "polynomial_regression",
        "decision_tree_regressor", "random_forest_regressor", "extra_trees_regressor",
        "adaboost_regressor", "gradient_boosting_regressor",
        "hist_gradient_boosting_regressor", "voting_regressor", "xgboost_regressor",
        "lightgbm_regressor", "catboost_regressor",
    }
    assert expected.issubset(TUNABLE_MODELS["value"])


def test_voting_search_profile_resolves_to_estimator_weights() -> None:
    trial = optuna.trial.FixedTrial({"weight_profile": "tree_emphasis"})
    sampled = suggest_params(trial, "churn", "voting")
    params = resolve_params("churn", "voting", sampled)
    assert params == {"weights": [1.0, 2.0, 1.0]}


def test_polynomial_degree_is_expressed_as_pipeline_parameter() -> None:
    trial = optuna.trial.FixedTrial({"polynomial_degree": 3})
    sampled = suggest_params(trial, "value", "polynomial_regression")
    params = resolve_params("value", "polynomial_regression", sampled)
    assert params == {"_polynomial_degree": 3}


def test_every_configured_candidate_has_a_valid_optuna_search_space() -> None:
    for task, models in TUNABLE_MODELS.items():
        for model in sorted(models):
            study = optuna.create_study(
                direction="maximize" if task == "churn" else "minimize",
                sampler=optuna.samplers.RandomSampler(seed=42),
            )

            def objective(trial, current_task=task, current_model=model):
                proposed = suggest_params(trial, current_task, current_model)
                effective = resolve_params(current_task, current_model, proposed)
                assert isinstance(effective, dict)
                return float(max(1, len(effective)))

            study.optimize(objective, n_trials=1)
            assert study.trials[0].state == optuna.trial.TrialState.COMPLETE, (
                f"Search space failed for {task}/{model}"
            )


# --- Consolidated from test_schema.py ---

"""Unit tests for transaction schema normalization."""

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
