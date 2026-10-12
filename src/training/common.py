"""Shared, leakage-aware training utilities for Phase 3."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "project.yaml"
METADATA_COLUMNS = {
    "customer_id", "snapshot_date", "outcome_end_exclusive", "split",
    "target_churn_90d", "target_net_value_90d_gbp", "target_value_valid",
    "label_matured", "prediction_timestamp", "prediction_time",
    "churn_30d_outcome_end_exclusive", "target_churn_30d_matured",
}


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def make_run_id() -> str:
    return uuid.uuid4().hex[:12]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def load_snapshots() -> tuple[pd.DataFrame, Path]:
    cfg = load_config()
    path = project_path(cfg["paths"]["customer_snapshots"])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Complete Phase 2 first: "
            "data/processed/customer_snapshots.csv is required."
        )
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "customer_id", "snapshot_date", "split",
        cfg["supervised"]["churn_target"],
        cfg["supervised"]["value_target"],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Phase 2 snapshot file is missing columns: {missing}")
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    if frame["snapshot_date"].isna().any():
        raise ValueError("snapshot_date contains unparseable timestamps")
    frame["snapshot_date"] = frame["snapshot_date"].dt.normalize()
    if frame.duplicated(["customer_id", "snapshot_date"]).any():
        raise ValueError("Duplicate (customer_id, snapshot_date) rows found")
    valid_splits = {"train", "validation", "test", "purged_gap"}
    unexpected = sorted(set(frame["split"].dropna().astype(str)) - valid_splits)
    if unexpected:
        raise ValueError(f"Unexpected split labels: {unexpected}")
    return frame, path


def persist_split_membership(frame: pd.DataFrame, output_path: Path, horizon_days: int) -> dict[str, Any]:
    """Verify split windows and persist membership, including purged gaps."""
    if "outcome_end_exclusive" in frame.columns:
        end = pd.to_datetime(frame["outcome_end_exclusive"], errors="coerce")
    else:
        end = frame["snapshot_date"] + pd.to_timedelta(horizon_days, unit="D")
    check = frame[["customer_id", "snapshot_date", "split"]].copy()
    check["outcome_end_exclusive"] = end
    check.to_csv(output_path, index=False)

    dates: dict[str, list[pd.Timestamp]] = {}
    for name in ("train", "validation", "test"):
        group = check.loc[check["split"].eq(name)]
        if group.empty:
            raise ValueError(f"The Phase 2 file has no rows for split '{name}'")
        dates[name] = sorted(group["snapshot_date"].dropna().unique())

    last_train_end = check.loc[check.split.eq("train"), "outcome_end_exclusive"].max()
    first_val = min(dates["validation"])
    last_val_end = check.loc[check.split.eq("validation"), "outcome_end_exclusive"].max()
    first_test = min(dates["test"])
    if pd.isna(last_train_end) or last_train_end > first_val:
        raise ValueError(
            f"Training outcome windows overlap validation: {last_train_end} > {first_val}"
        )
    if pd.isna(last_val_end) or last_val_end > first_test:
        raise ValueError(
            f"Validation outcome windows overlap test: {last_val_end} > {first_test}"
        )

    return {
        "train_snapshot_dates": [pd.Timestamp(x).date().isoformat() for x in dates["train"]],
        "validation_snapshot_dates": [pd.Timestamp(x).date().isoformat() for x in dates["validation"]],
        "test_snapshot_dates": [pd.Timestamp(x).date().isoformat() for x in dates["test"]],
        "last_train_outcome_end_exclusive": pd.Timestamp(last_train_end).isoformat(),
        "first_validation_snapshot": pd.Timestamp(first_val).isoformat(),
        "last_validation_outcome_end_exclusive": pd.Timestamp(last_val_end).isoformat(),
        "first_test_snapshot": pd.Timestamp(first_test).isoformat(),
        "customer_repeats_across_time_are_allowed": True,
        "customer_id_used_as_feature": False,
    }


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], dict[str, str]]:
    """Build an explicit feature allow-list and record all excluded columns."""
    excluded: dict[str, str] = {}
    candidates: list[str] = []
    for col in frame.columns:
        if col in METADATA_COLUMNS:
            excluded[col] = "identifier, split metadata, timestamp, or target; never a model input"
        elif str(col).startswith("target_") or str(col).startswith("label_"):
            excluded[col] = "target/label-derived field; leakage risk"
        else:
            candidates.append(col)

    numeric: list[str] = []
    categorical: list[str] = []
    unsupported: list[str] = []
    for col in candidates:
        if pd.api.types.is_numeric_dtype(frame[col]) or pd.api.types.is_bool_dtype(frame[col]):
            numeric.append(col)
        elif pd.api.types.is_object_dtype(frame[col]) or pd.api.types.is_string_dtype(frame[col]) or isinstance(frame[col].dtype, pd.CategoricalDtype):
            # The current Phase 2 feature contract has 'country' as its only categorical feature.
            if col == "country":
                categorical.append(col)
            else:
                excluded[col] = "non-numeric unapproved feature; add an explicit encoding policy first"
                unsupported.append(col)
        else:
            excluded[col] = "unsupported dtype"
    if not numeric and not categorical:
        raise ValueError("No usable features were found")
    if any(c.startswith("target_") for c in numeric + categorical):
        raise RuntimeError("Target leakage detected in feature allow-list")
    return numeric, categorical, excluded


def make_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, categorical))
    return ColumnTransformer(
        transformers=transformers, remainder="drop",
        verbose_feature_names_out=True,
    )


def _optional_estimators() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        from xgboost import XGBClassifier, XGBRegressor
        result["xgboost_classifier"] = XGBClassifier
        result["xgboost_regressor"] = XGBRegressor
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        result["lightgbm_classifier"] = LGBMClassifier
        result["lightgbm_regressor"] = LGBMRegressor
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
        result["catboost_classifier"] = CatBoostClassifier
        result["catboost_regressor"] = CatBoostRegressor
    except Exception:
        pass
    return result


def build_estimator(task: str, model_name: str, seed: int, params: dict[str, Any] | None = None) -> tuple[Any, bool]:
    """Return (estimator, requires_numeric_scaling)."""
    params = dict(params or {})
    optional = _optional_estimators()
    if task == "churn":
        table: dict[str, tuple[Any, bool, dict[str, Any]]] = {
            "dummy": (DummyClassifier, False, {"strategy": "prior"}),
            "logistic_regression": (LogisticRegression, True, {"max_iter": 3000, "class_weight": "balanced", "random_state": seed}),
            "decision_tree": (DecisionTreeClassifier, False, {"max_depth": 5, "min_samples_leaf": 30, "class_weight": "balanced", "random_state": seed}),
            "random_forest": (RandomForestClassifier, False, {"n_estimators": 250, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": -1, "random_state": seed}),
            "extra_trees": (ExtraTreesClassifier, False, {"n_estimators": 250, "min_samples_leaf": 10, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": -1, "random_state": seed}),
            "adaboost": (AdaBoostClassifier, False, {"n_estimators": 150, "learning_rate": 0.05, "random_state": seed}),
            "gradient_boosting": (GradientBoostingClassifier, False, {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2, "random_state": seed}),
            "hist_gradient_boosting": (HistGradientBoostingClassifier, False, {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "random_state": seed}),
            "svm": (SVC, True, {"C": 1.0, "kernel": "linear", "probability": True, "class_weight": "balanced", "random_state": seed}),
            "svm_rbf": (SVC, True, {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True, "class_weight": "balanced", "random_state": seed}),
            "knn": (KNeighborsClassifier, True, {"n_neighbors": 15, "weights": "distance", "n_jobs": -1}),
            "naive_bayes": (GaussianNB, True, {}),
            "voting": (VotingClassifier, True, {"estimators": [
                ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
                ("forest", RandomForestClassifier(n_estimators=80, min_samples_leaf=10, class_weight="balanced_subsample", n_jobs=1, random_state=seed)),
                ("boost", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=seed)),
            ], "voting": "soft", "n_jobs": 1}),
        }
        if "xgboost_classifier" in optional:
            table["xgboost"] = (optional["xgboost_classifier"], False, {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "tree_method": "hist", "eval_metric": "logloss", "n_jobs": 4, "random_state": seed})
        if "lightgbm_classifier" in optional:
            table["lightgbm"] = (optional["lightgbm_classifier"], False, {"n_estimators": 250, "max_depth": -1, "num_leaves": 31, "learning_rate": 0.05, "verbosity": -1, "n_jobs": 4, "random_state": seed})
        if "catboost_classifier" in optional:
            table["catboost"] = (optional["catboost_classifier"], False, {"iterations": 250, "depth": 6, "learning_rate": 0.05, "verbose": False, "allow_writing_files": False, "thread_count": 4, "random_seed": seed})
    elif task == "value":
        table = {
            "dummy_median": (DummyRegressor, False, {"strategy": "median"}),
            "linear_regression": (LinearRegression, True, {}),
            "ridge": (Ridge, True, {"alpha": 10.0}),
            "lasso": (Lasso, True, {"alpha": 0.01, "max_iter": 5000}),
            "elastic_net": (ElasticNet, True, {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 5000}),
            "polynomial_regression": (LinearRegression, True, {"_polynomial_degree": 2}),
            "decision_tree_regressor": (DecisionTreeRegressor, False, {"max_depth": 6, "min_samples_leaf": 20, "random_state": seed}),
            "random_forest_regressor": (RandomForestRegressor, False, {"n_estimators": 250, "min_samples_leaf": 10, "max_features": 0.8, "n_jobs": -1, "random_state": seed}),
            "extra_trees_regressor": (ExtraTreesRegressor, False, {"n_estimators": 250, "min_samples_leaf": 10, "max_features": 0.8, "n_jobs": -1, "random_state": seed}),
            "adaboost_regressor": (AdaBoostRegressor, False, {"n_estimators": 150, "learning_rate": 0.05, "random_state": seed}),
            "gradient_boosting_regressor": (GradientBoostingRegressor, False, {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.05, "random_state": seed}),
            "hist_gradient_boosting_regressor": (HistGradientBoostingRegressor, False, {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "random_state": seed}),
            "voting_regressor": (VotingRegressor, True, {"estimators": [
                ("linear", Ridge(alpha=10.0)),
                ("forest", RandomForestRegressor(n_estimators=80, min_samples_leaf=10, max_features=0.8, n_jobs=1, random_state=seed)),
                ("boost", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=seed)),
            ], "n_jobs": 1}),
        }
        if "xgboost_regressor" in optional:
            table["xgboost_regressor"] = (optional["xgboost_regressor"], False, {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.9, "tree_method": "hist", "objective": "reg:squarederror", "n_jobs": 4, "random_state": seed})
        if "lightgbm_regressor" in optional:
            table["lightgbm_regressor"] = (optional["lightgbm_regressor"], False, {"n_estimators": 250, "num_leaves": 31, "learning_rate": 0.05, "verbosity": -1, "n_jobs": 4, "random_state": seed})
        if "catboost_regressor" in optional:
            table["catboost_regressor"] = (optional["catboost_regressor"], False, {"iterations": 250, "depth": 6, "learning_rate": 0.05, "verbose": False, "allow_writing_files": False, "thread_count": 4, "random_seed": seed})
    else:
        raise ValueError("task must be 'churn' or 'value'")

    if model_name not in table:
        raise ValueError(
            f"Model '{model_name}' not available for task '{task}'. "
            f"Available: {sorted(table)}. Install requirements-ml.txt for boosted models."
        )
    cls, scale, defaults = table[model_name]
    final_params = {**defaults, **params}
    poly_degree = final_params.pop("_polynomial_degree", None)
    estimator = cls(**final_params)
    setattr(estimator, "_phase3_polynomial_degree", poly_degree) if poly_degree else None
    return estimator, scale


def build_pipeline(task: str, model_name: str, numeric: list[str], categorical: list[str], seed: int, params: dict[str, Any] | None = None) -> Pipeline:
    estimator, scale_numeric = build_estimator(task, model_name, seed, params)
    preprocessor = make_preprocessor(numeric, categorical, scale_numeric)
    if getattr(estimator, "_phase3_polynomial_degree", None):
        return Pipeline([
            ("preprocessor", preprocessor),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", estimator),
        ])
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def purged_walk_forward_folds(frame: pd.DataFrame, horizon_days: int, max_folds: int = 3) -> list[tuple[np.ndarray, np.ndarray, str, str]]:
    """Walk-forward folds with prior outcome windows purged before validation date."""
    dates = sorted(pd.to_datetime(frame["snapshot_date"]).dt.normalize().unique())
    if len(dates) < 2:
        return []
    folds = []
    for val_date in dates[1:]:
        val_date = pd.Timestamp(val_date)
        eligible_train_dates = [
            pd.Timestamp(d) for d in dates
            if pd.Timestamp(d) + pd.Timedelta(days=horizon_days) <= val_date
        ]
        if not eligible_train_dates:
            continue
        train_mask = frame["snapshot_date"].isin(eligible_train_dates)
        val_mask = frame["snapshot_date"].eq(val_date)
        train_idx = np.flatnonzero(train_mask.to_numpy())
        val_idx = np.flatnonzero(val_mask.to_numpy())
        if len(train_idx) and len(val_idx):
            folds.append((train_idx, val_idx, pd.Timestamp(max(eligible_train_dates)).date().isoformat(), val_date.date().isoformat()))
    return folds[-max_folds:]


def split_validation_roles(validation: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Partition validation dates into model selection, calibration, threshold windows."""
    dates = sorted(pd.to_datetime(validation["snapshot_date"]).dt.normalize().unique())
    if len(dates) < 3:
        # The split may still work, but roles cannot be independently estimated.
        return validation.copy(), validation.iloc[0:0].copy(), validation.copy(), {
            "independent_roles_available": False,
            "warning": "Fewer than three validation snapshot dates; model selection/threshold separation is unavailable.",
            "selection_dates": [pd.Timestamp(x).date().isoformat() for x in dates],
            "calibration_dates": [],
            "threshold_dates": [pd.Timestamp(x).date().isoformat() for x in dates],
        }
    roles = cfg.get("validation_roles", {})
    selection_fraction = float(roles.get("model_selection_fraction", 0.5))
    calibration_fraction = float(roles.get("calibration_fraction", 0.25))
    n = len(dates)
    n_sel = min(max(1, int(np.floor(n * selection_fraction))), n - 2)
    n_cal = min(max(1, int(np.floor(n * calibration_fraction))), n - n_sel - 1)
    selection_dates = dates[:n_sel]
    calibration_dates = dates[n_sel:n_sel + n_cal]
    threshold_dates = dates[n_sel + n_cal:]
    pick = lambda ds: validation.loc[validation["snapshot_date"].isin(ds)].copy()
    return pick(selection_dates), pick(calibration_dates), pick(threshold_dates), {
        "independent_roles_available": True,
        "selection_dates": [pd.Timestamp(x).date().isoformat() for x in selection_dates],
        "calibration_dates": [pd.Timestamp(x).date().isoformat() for x in calibration_dates],
        "threshold_dates": [pd.Timestamp(x).date().isoformat() for x in threshold_dates],
    }


class PlattCalibrator:
    """Small, serializable sigmoid calibrator fit on a separate calibration window."""
    def __init__(self) -> None:
        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.fitted = False
        self.reason: str | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, probability: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        y = np.asarray(y, dtype=int)
        if len(np.unique(y)) < 2:
            self.reason = "Calibration window contains only one class; identity calibration used."
            return self
        self.model.fit(self._logit(probability), y)
        self.fitted = True
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
        if not self.fitted:
            return p
        return self.model.predict_proba(self._logit(p))[:, 1]


def select_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Choose a single fixed threshold on the dedicated threshold-selection window."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return 0.5, {"method": "fallback_0.5", "reason": "threshold window contains one class"}
    candidates = np.unique(np.quantile(prob, np.linspace(0.01, 0.99, 99)))
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], candidates)))
    scored = []
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        scored.append((f1_score(y_true, pred, zero_division=0), float(threshold), precision_score(y_true, pred, zero_division=0), recall_score(y_true, pred, zero_division=0)))
    best = max(scored, key=lambda row: (row[0], row[2], row[3]))
    return best[1], {"method": "maximize_f1_on_threshold_window", "f1": best[0], "precision": best[2], "recall": best[3]}


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float, frame: pd.DataFrame, capacity_fraction: float) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, precision_score, recall_score,
        f1_score, accuracy_score, brier_score_loss, confusion_matrix,
    )
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    result: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else None,
        "pr_auc_average_precision": float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else None,
        "precision_at_fixed_threshold": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_fixed_threshold": float(recall_score(y_true, pred, zero_division=0)),
        "f1_at_fixed_threshold": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy_at_fixed_threshold": float(accuracy_score(y_true, pred)),
        "brier_score": float(brier_score_loss(y_true, prob)),
        "expected_calibration_error_10_bins": float(np.mean([
            np.abs(np.mean(y_true[(prob >= lo) & (prob < hi)]) - np.mean(prob[(prob >= lo) & (prob < hi)])) * np.mean((prob >= lo) & (prob < hi))
            for lo, hi in zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1.0, 10))
            if np.any((prob >= lo) & ((prob < hi) if hi < 1.0 else (prob <= hi)))
        ])) if len(y_true) else None,
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y_true, pred, labels=[0, 1]).ravel().tolist(),
        "threshold": float(threshold),
    }
    if "snapshot_date" in frame.columns and len(frame):
        temp = pd.DataFrame({"y": y_true, "p": prob}, index=frame.index)
        temp["snapshot_date"] = frame["snapshot_date"].to_numpy()
        temp["rank"] = temp.groupby("snapshot_date")["p"].rank(method="first", ascending=False)
        temp["batch_n"] = temp.groupby("snapshot_date")["p"].transform("size")
        temp["contacted"] = temp["rank"] <= np.maximum(1, np.ceil(temp["batch_n"] * capacity_fraction))
        positives = int(temp["y"].sum())
        caught = int((temp["contacted"] & temp["y"].eq(1)).sum())
        contacted = int(temp["contacted"].sum())
        result["retention_capacity_fraction"] = float(capacity_fraction)
        result["contacts_at_capacity"] = contacted
        result["precision_among_contacted"] = float(caught / contacted) if contacted else 0.0
        result["recall_at_capacity"] = float(caught / positives) if positives else None
    result["retention_value_from_experiment"] = "not_estimable_without a controlled retention experiment and value assumptions"
    return result


def regression_metrics(y_true: np.ndarray, pred: np.ndarray, tier_reference: np.ndarray | None = None) -> dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return {"n": 0, "mae_gbp": None, "rmse_gbp": None, "r2": None, "error_by_value_tier": {}}
    mse = mean_squared_error(y, p)
    nonzero_actual = np.abs(y) > 1e-9
    mape = (
        float(np.mean(np.abs((y[nonzero_actual] - p[nonzero_actual]) / y[nonzero_actual])) * 100.0)
        if nonzero_actual.any() else None
    )
    result: dict[str, Any] = {
        "n": int(len(y)),
        "mae_gbp": float(mean_absolute_error(y, p)),
        "rmse_gbp": float(np.sqrt(mse)),
        "mse_gbp_squared": float(mse),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "mape_pct_nonzero_actual": mape,
        "mape_nonzero_actual_count": int(nonzero_actual.sum()),
        "mape_note": "MAPE excludes actual targets equal to zero because division by zero is undefined; interpret alongside MAE/RMSE.",
        "top_decile_allocation": {},
    }
    # Downstream allocation proxy: value captured by the top 10% ranked predictions.
    k = max(1, int(np.ceil(0.10 * len(y))))
    idx = np.argsort(p)[::-1][:k]
    total = float(np.sum(y))
    captured = float(np.sum(y[idx]))
    result["top_decile_allocation"] = {
        "rows_selected": int(k),
        "actual_net_value_in_top_predicted_decile_gbp": captured,
        "actual_mean_value_in_top_predicted_decile_gbp": float(np.mean(y[idx])),
        "fraction_of_total_net_value_captured": float(captured / total) if abs(total) > 1e-9 else None,
        "interpretation": "ranking diagnostic, not causal incremental revenue or profit",
    }
    # Define value tiers using training-distribution positive-value quantiles.
    ref = np.asarray(tier_reference if tier_reference is not None else y, dtype=float)
    ref = ref[np.isfinite(ref)]
    positive_ref = ref[ref > 0]
    q50 = float(np.quantile(positive_ref, 0.50)) if len(positive_ref) else 0.0
    q90 = float(np.quantile(positive_ref, 0.90)) if len(positive_ref) else 0.0
    tiers = np.where(y <= 0, "zero_or_negative", np.where(y <= q50, "low_positive", np.where(y <= q90, "medium_positive", "high_positive")))
    tier_metrics: dict[str, Any] = {}
    for tier in sorted(set(tiers)):
        mask = tiers == tier
        tier_metrics[str(tier)] = {
            "n": int(mask.sum()),
            "mae_gbp": float(mean_absolute_error(y[mask], p[mask])),
            "rmse_gbp": float(np.sqrt(mean_squared_error(y[mask], p[mask]))),
        }
    result["error_by_value_tier"] = tier_metrics
    result["tier_thresholds_from_training_gbp"] = {"positive_median": q50, "positive_90th_percentile": q90}
    return result


def log_environment() -> dict[str, Any]:
    import platform
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ("pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna", "shap"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def process_rss_mb() -> float | None:
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2))
    except Exception:
        return None
