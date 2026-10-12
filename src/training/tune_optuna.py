"""Bounded persistent Optuna tuning using purged, chronological training folds only."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.training.common import (
    ROOT, load_config, project_path, json_dump, sha256_file, load_snapshots,
    feature_columns, build_pipeline, purged_walk_forward_folds,
)


def suggest_params(trial: optuna.Trial, task: str, model: str) -> dict[str, Any]:
    if model in {"logistic_regression"}:
        return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    if model in {"svm", "svm_rbf"}:
        return {
            "C": trial.suggest_float("C", 1e-2, 50.0, log=True),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]) if model == "svm_rbf" else "scale",
        }
    if model == "knn":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 35, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
    if model in {"decision_tree", "decision_tree_regressor"}:
        return {
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 80),
        }
    if model in {"random_forest", "extra_trees", "random_forest_regressor", "extra_trees_regressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 450, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 8, 12, 18]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 40),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.5, 0.8]),
        }
    if model in {"xgboost", "xgboost_regressor"}:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        return params
    if model in {"lightgbm", "lightgbm_regressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63, step=4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 10]),
        }
    if model in {"catboost", "catboost_regressor"}:
        return {
            "iterations": trial.suggest_int("iterations", 100, 500, step=50),
            "depth": trial.suggest_int("depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    if model in {"ridge"}:
        return {"alpha": trial.suggest_float("alpha", 1e-4, 1000.0, log=True)}
    if model in {"lasso"}:
        return {"alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True)}
    if model in {"elastic_net"}:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
        }
    if model in {"gradient_boosting", "gradient_boosting_regressor"}:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 1, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    raise ValueError(f"Optuna search space not defined for {task}/{model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["churn", "value"], required=True)
    parser.add_argument("--model", required=True, help="Model name from the comparison report")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()
    if args.trials is not None and args.trials < 1:
        raise ValueError("--trials must be at least 1")

    cfg = load_config()
    frame, source_path = load_snapshots()
    numeric, categorical, _ = feature_columns(frame)
    target = cfg["supervised"]["churn_target"] if args.task == "churn" else cfg["supervised"]["value_target"]
    frame = frame.loc[frame.split.eq("train") & frame[target].notna()].copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.loc[np.isfinite(frame[target].to_numpy(dtype=float))].copy()
    if args.task == "churn":
        frame = frame.loc[frame[target].isin([0, 1])].copy()
        frame[target] = frame[target].astype(int)
    if frame.empty:
        raise ValueError("No training rows remain after target filtering")

    horizon = int(cfg["supervised"].get("churn_horizon_days" if args.task == "churn" else "value_horizon_days", cfg["supervised"]["horizon_days"]))
    folds = purged_walk_forward_folds(frame, horizon, int(cfg["supervised"]["cv_max_folds"]))
    if not folds:
        raise ValueError("No purged chronological CV folds are available. Extend the observation period or adjust the documented split design.")

    opt_cfg = cfg["optuna"]
    storage_path = project_path(opt_cfg["storage"])
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study_name = f"{args.task}_{args.model}_{sha256_file(source_path)[:10]}"
    storage_uri = f"sqlite:///{storage_path.as_posix()}"
    direction = "maximize" if args.task == "churn" else "minimize"
    study = optuna.create_study(
        study_name=study_name, storage=storage_uri, direction=direction,
        sampler=optuna.samplers.TPESampler(seed=int(opt_cfg["seed"])),
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, args.task, args.model)
        fold_scores: list[float] = []
        X = frame[numeric + categorical]
        y = frame[target]
        for fold_idx, (tr_idx, va_idx, train_end, val_date) in enumerate(folds, start=1):
            try:
                model = build_pipeline(args.task, args.model, numeric, categorical, int(cfg["project"]["random_seed"]), params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                if args.task == "churn":
                    from src.training.train_supervised import _score_probability
                    prob = _score_probability(model, X.iloc[va_idx])
                    y_val = y.iloc[va_idx].astype(int).to_numpy()
                    if len(np.unique(y_val)) < 2:
                        continue
                    from sklearn.metrics import average_precision_score
                    score = float(average_precision_score(y_val, prob))
                else:
                    from sklearn.metrics import mean_absolute_error
                    pred = model.predict(X.iloc[va_idx])
                    score = float(mean_absolute_error(y.iloc[va_idx], pred))
                fold_scores.append(score)
                trial.set_user_attr(f"fold_{fold_idx}_score", score)
                trial.set_user_attr(f"fold_{fold_idx}_train_end", train_end)
                trial.set_user_attr(f"fold_{fold_idx}_validation_date", val_date)
            except Exception as exc:
                trial.set_user_attr(f"fold_{fold_idx}_error", f"{type(exc).__name__}: {exc}")
        if not fold_scores:
            raise optuna.TrialPruned("No valid purged CV fold completed")
        trial.set_user_attr("fold_scores", fold_scores)
        trial.set_user_attr("fold_mean", float(np.mean(fold_scores)))
        trial.set_user_attr("fold_std", float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0)
        return float(np.mean(fold_scores))

    trials = int(args.trials if args.trials is not None else opt_cfg["default_trials"])
    timeout = int(args.timeout_seconds if args.timeout_seconds is not None else opt_cfg["timeout_seconds"])
    print(f"Study: {study_name}; trials to add: {trials}; folds: {len(folds)}; direction: {direction}")
    study.optimize(objective, n_trials=trials, timeout=timeout, gc_after_trial=True, show_progress_bar=True)

    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No Optuna trials completed successfully")
    best = study.best_trial
    out = project_path(cfg["paths"]["reports_dir"]) / f"optuna_best_{args.task}_{args.model}.json"
    data = {
        "task": args.task, "model": args.model, "study_name": study_name,
        "storage_path": str(storage_path.relative_to(ROOT)), "direction": direction,
        "best_value": float(best.value), "best_params": best.params,
        "best_trial_number": int(best.number), "best_trial_fold_scores": best.user_attrs.get("fold_scores"),
        "best_trial_fold_std": best.user_attrs.get("fold_std"),
        "completed_trials": len(completed), "total_study_trials": len(study.trials),
        "new_trials_requested": trials, "input_sha256": sha256_file(source_path),
        "note": "Tuning used training folds only. Treat CV best value as model-selection evidence, not an unbiased final estimate. Validation remains for candidate calibration/threshold or selection; test was not read.",
    }
    json_dump(out, data)
    study.trials_dataframe().to_csv(project_path(cfg["paths"]["reports_dir"]) / f"optuna_trials_{args.task}_{args.model}.csv", index=False)
    print(f"Best {args.task}/{args.model}: {best.value:.6f}")
    print(f"Best parameters saved: {out}")
    print("Test data was not accessed.")


if __name__ == "__main__":
    main()
