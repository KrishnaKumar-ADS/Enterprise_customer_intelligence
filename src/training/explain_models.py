"""Generate model explanations using permutation importance, coefficients and optional SHAP."""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.tree import plot_tree

from src.training.common import (
    load_config, project_path, json_dump, load_snapshots, sha256_file,
    feature_columns, build_pipeline, purged_walk_forward_folds,
)
from src.training.train_supervised import _score_probability


def _transform_for_explanation(pipeline: Any, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    names = list(preprocessor.get_feature_names_out())
    if "polynomial" in pipeline.named_steps:
        poly = pipeline.named_steps["polynomial"]
        transformed = poly.transform(transformed)
        names = list(poly.get_feature_names_out(names))
    return np.asarray(transformed), names


def _original_feature(transformed_name: str) -> str:
    name = transformed_name.split("__", 1)[-1]
    # One-hot features such as country_United Kingdom map back to country.
    if name.startswith("country_"):
        return "country"
    return name



def _write_decision_tree_diagnostic(
    task: str, frame: pd.DataFrame, validation: pd.DataFrame,
    numeric: list[str], categorical: list[str], target: str,
    config: dict[str, Any], report_dir: Path, selected_model_name: str,
    selected_model_version: str,
) -> dict[str, Any]:
    """Fit/use the shallow tree candidate on train only and write local paths.

    This diagnostic is generated even when another family wins model selection,
    so tree interpretability remains represented in the model-comparison work.
    """
    tree_name = "decision_tree" if task == "churn" else "decision_tree_regressor"
    train = frame.loc[frame["split"].eq("train") & frame[target].notna()].copy()
    if task == "value" and "target_value_valid" in train.columns:
        train = train.loc[train["target_value_valid"].fillna(False).astype(bool)]
    if train.empty or (task == "churn" and train[target].nunique() < 2):
        return {"status": "skipped", "reason": "Insufficient training rows/classes for decision-tree diagnostic"}

    tree_pipeline = build_pipeline(
        task, tree_name, numeric, categorical, int(config["project"]["random_seed"])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree_pipeline.fit(train[numeric + categorical], train[target].astype(int if task == "churn" else float))
    tree_estimator = tree_pipeline.named_steps["model"]
    n = min(10, len(validation))
    X_tree = validation[numeric + categorical].iloc[:n].copy()
    for col in numeric:
        X_tree[col] = pd.to_numeric(X_tree[col], errors="coerce").astype(float)
    transformed, tree_names = _transform_for_explanation(tree_pipeline, X_tree)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_path = report_dir / f"{task}_decision_tree_diagnostic.png"
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree_estimator, feature_names=tree_names, filled=False, max_depth=3, ax=ax)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    path_matrix = tree_estimator.decision_path(transformed).toarray()
    tree = tree_estimator.tree_
    path_rows = []
    for sample_idx in range(n):
        sample_row = validation.iloc[sample_idx]
        node_ids = np.flatnonzero(path_matrix[sample_idx])
        steps = []
        for node_id in node_ids:
            feature_id = int(tree.feature[node_id])
            if feature_id < 0:
                continue
            feature_value = float(transformed[sample_idx, feature_id])
            threshold = float(tree.threshold[node_id])
            steps.append({
                "node": int(node_id), "feature": tree_names[feature_id],
                "value": feature_value, "operator": "<=", "threshold": threshold,
                "branch": "left" if feature_value <= threshold else "right",
            })
        if task == "churn":
            class_list = list(getattr(tree_estimator, "classes_", [0, 1]))
            positive_index = class_list.index(1) if 1 in class_list else -1
            prediction = float(tree_estimator.predict_proba(transformed[[sample_idx]])[0, positive_index])
            prediction_kind = "tree_raw_churn_probability"
        else:
            prediction = float(tree_estimator.predict(transformed[[sample_idx]])[0])
            prediction_kind = "tree_predicted_net_value_gbp"
        row_version = selected_model_version if selected_model_name == tree_name else f"diagnostic-only-{tree_name}-seed-{int(config['project']['random_seed'])}"
        path_rows.append({
            "prediction_id": hashlib.sha256(f"{task}|{sample_row['customer_id']}|{sample_row['snapshot_date']}|{row_version}".encode()).hexdigest()[:24],
            "customer_id": str(sample_row["customer_id"]),
            "snapshot_date": str(sample_row["snapshot_date"]),
            "explanation_model_name": tree_name,
            "model_version": row_version,
            "is_selected_candidate": bool(selected_model_name == tree_name),
            prediction_kind: prediction,
            "decision_path_json": json.dumps(steps),
            "warning": "Training-only decision-tree diagnostic; its path describes model logic, not causation.",
        })
    paths_path = report_dir / f"{task}_decision_tree_paths.csv"
    pd.DataFrame(path_rows).to_csv(paths_path, index=False)
    return {
        "status": "completed", "diagnostic_model": tree_name,
        "selected_model_name": selected_model_name,
        "selected_candidate_is_tree": bool(selected_model_name == tree_name),
        "plot": str(plot_path.relative_to(project_path("."))),
        "paths": str(paths_path.relative_to(project_path("."))),
        "training_policy": "tree fit on training snapshots only; paths generated for up to 10 validation examples; test set not read",
    }


def _baseline_coefficient_reports(
    frame: pd.DataFrame, config: dict[str, Any], report_dir: Path
) -> dict[str, Any]:
    """Write interpretable baseline coefficients and fold-stability reports.

    The logistic/linear models are fit only on training snapshots; model selection
    does not depend on these coefficient explanations.
    """
    numeric, categorical, _ = feature_columns(frame)
    reports: dict[str, Any] = {}
    tasks = [
        ("churn", "logistic_regression", config["supervised"]["churn_target"], int(config["supervised"].get("churn_horizon_days", 30))),
        ("value", "linear_regression", config["supervised"]["value_target"], int(config["supervised"].get("value_horizon_days", 90))),
    ]
    for task, model_name, target, horizon in tasks:
        train = frame.loc[frame["split"].eq("train") & frame[target].notna()].copy()
        if task == "value" and "target_value_valid" in train.columns:
            train = train.loc[train["target_value_valid"].fillna(False).astype(bool)]
        if train.empty or (task == "churn" and train[target].nunique() < 2):
            reports[task] = {"status": "skipped", "reason": "No usable training rows/classes"}
            continue
        x_cols = numeric + categorical
        X = train[x_cols]
        y = train[target].astype(int if task == "churn" else float)

        baseline = build_pipeline(task, model_name, numeric, categorical, int(config["project"]["random_seed"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline.fit(X, y)
        estimator = baseline.named_steps["model"]
        names = list(baseline.named_steps["preprocessor"].get_feature_names_out())
        coef = np.asarray(estimator.coef_).reshape(-1)
        full = pd.DataFrame({
            "transformed_feature": names[:len(coef)],
            "original_feature_group": [_original_feature(name) for name in names[:len(coef)]],
            "coefficient": coef,
            "model_name": model_name,
            "task": task,
            "coefficient_scale": "log-odds per transformed feature unit" if task == "churn" else "GBP per transformed feature unit",
        })
        if task == "churn":
            full["odds_ratio_per_transformed_unit"] = np.exp(np.clip(coef, -30, 30))
        full["absolute_coefficient"] = full["coefficient"].abs()
        full.sort_values("absolute_coefficient", ascending=False).to_csv(
            report_dir / f"{task}_{model_name}_baseline_coefficients.csv", index=False
        )

        fold_rows: list[dict[str, Any]] = []
        for fold_idx, (tr_idx, _, train_end, val_date) in enumerate(
            purged_walk_forward_folds(train, horizon, int(config["supervised"].get("cv_max_folds", 3))),
            start=1,
        ):
            fold_train = train.iloc[tr_idx]
            fold_y = fold_train[target].astype(int if task == "churn" else float)
            if task == "churn" and fold_y.nunique() < 2:
                continue
            fold_model = build_pipeline(task, model_name, numeric, categorical, int(config["project"]["random_seed"]))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fold_model.fit(fold_train[x_cols], fold_y)
                fold_estimator = fold_model.named_steps["model"]
                fold_names = list(fold_model.named_steps["preprocessor"].get_feature_names_out())
                fold_coef = np.asarray(fold_estimator.coef_).reshape(-1)
                for feature, value in zip(fold_names, fold_coef):
                    fold_rows.append({
                        "fold": fold_idx, "train_end_snapshot": train_end,
                        "validation_snapshot": val_date, "transformed_feature": feature,
                        "original_feature_group": _original_feature(feature),
                        "coefficient": float(value),
                    })
            except Exception as exc:
                fold_rows.append({"fold": fold_idx, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})

        fold_path = report_dir / f"{task}_{model_name}_coefficient_folds.csv"
        pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
        successful = pd.DataFrame([r for r in fold_rows if "coefficient" in r])
        stability_path = report_dir / f"{task}_{model_name}_coefficient_stability.csv"
        if not successful.empty:
            grouped = successful.groupby(["transformed_feature", "original_feature_group"])["coefficient"]
            stability = grouped.agg(
                n_folds_present="count", mean_coefficient="mean", std_coefficient="std",
                minimum_coefficient="min", maximum_coefficient="max",
            ).reset_index()
            signs = successful.assign(positive=successful["coefficient"].gt(0))
            sign_stability = signs.groupby("transformed_feature")["positive"].agg(
                positive_fold_count="sum", folds_present="count"
            ).reset_index()
            sign_stability["sign_consistency_fraction"] = np.maximum(
                sign_stability["positive_fold_count"],
                sign_stability["folds_present"] - sign_stability["positive_fold_count"],
            ) / sign_stability["folds_present"].clip(lower=1)
            stability = stability.merge(
                sign_stability[["transformed_feature", "sign_consistency_fraction"]],
                on="transformed_feature", how="left",
            )
            stability.to_csv(stability_path, index=False)
            status = "PASSED"
        else:
            pd.DataFrame(columns=["transformed_feature", "mean_coefficient", "std_coefficient"]).to_csv(stability_path, index=False)
            status = "not_available_no_successful_time_cv_folds"

        reports[task] = {
            "status": status,
            "baseline_model": model_name,
            "coefficients": str((report_dir / f"{task}_{model_name}_baseline_coefficients.csv").relative_to(project_path("."))),
            "fold_coefficients": str(fold_path.relative_to(project_path("."))),
            "stability": str(stability_path.relative_to(project_path("."))),
            "fold_rows": int(len(successful)),
            "interpretation": "Standardized numeric coefficients describe the fitted model association, not causal effects. One-hot category coefficients compare a category with the encoded reference category.",
        }
    return reports

def main() -> None:
    script_started = time.perf_counter()
    config = load_config()
    report_dir = project_path(config["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    frame, input_path = load_snapshots()
    numeric, categorical = [], []

    outputs: dict[str, Any] = {}
    outputs["interpretable_baselines"] = _baseline_coefficient_reports(frame, config, report_dir)
    for task in ("churn", "value"):
        task_started = time.perf_counter()
        artifact_path = project_path(config["paths"]["models_dir"]) / f"{task}_model.joblib"
        manifest_path = report_dir / f"{task}_model_manifest.json"
        if not artifact_path.exists() or not manifest_path.exists():
            outputs[task] = {"status": "skipped", "reason": "Train model first"}
            continue
        bundle = joblib.load(artifact_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        feature_names = bundle["feature_names"]
        numeric = bundle["numeric_features"]
        categorical = bundle["categorical_features"]
        target = bundle["target"]
        validation = frame.loc[frame.split.eq("validation") & frame[target].notna()].copy()
        if task == "value" and "target_value_valid" in validation.columns:
            validation = validation.loc[validation["target_value_valid"].fillna(False).astype(bool)]
        if validation.empty:
            outputs[task] = {"status": "skipped", "reason": "No validation rows"}
            continue
        X = validation[feature_names]
        y = validation[target].astype(int).to_numpy() if task == "churn" else validation[target].astype(float).to_numpy()
        sample_n = min(2000, len(validation))
        X_sample = X.iloc[:sample_n]
        y_sample = y[:sample_n]
        pipeline = bundle["model"]

        # Raw-feature permutation importance preserves the input-to-feature mapping.
        try:
            scorer = "average_precision" if task == "churn" else "neg_mean_absolute_error"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                perm = permutation_importance(
                    pipeline, X_sample, y_sample, scoring=scorer,
                    n_repeats=3, random_state=int(config["project"]["random_seed"]), n_jobs=-1,
                )
            perm_df = pd.DataFrame({
                "feature": feature_names,
                "permutation_importance_mean": perm.importances_mean,
                "permutation_importance_std": perm.importances_std,
                "task": task,
                "model_version": bundle["model_version"],
            }).sort_values("permutation_importance_mean", ascending=False)
            perm_path = report_dir / f"{task}_permutation_importance.csv"
            perm_df.to_csv(perm_path, index=False)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                top = perm_df.head(20).sort_values("permutation_importance_mean")
                fig, ax = plt.subplots(figsize=(9, max(5, 0.3 * len(top))))
                ax.barh(top["feature"], top["permutation_importance_mean"], xerr=top["permutation_importance_std"])
                ax.set_title(f"Held-out permutation importance — {task}")
                ax.set_xlabel("Score decrease when permuted")
                fig.tight_layout()
                fig.savefig(report_dir / f"{task}_permutation_importance.png", dpi=160)
                plt.close(fig)
            except Exception as exc:
                outputs.setdefault(task, {})["permutation_plot_warning"] = str(exc)
            outputs.setdefault(task, {})["permutation_importance"] = str(perm_path.relative_to(project_path(".")))
        except Exception as exc:
            outputs.setdefault(task, {})["permutation_importance_error"] = f"{type(exc).__name__}: {exc}"
            perm_df = pd.DataFrame(columns=["feature", "permutation_importance_mean"])

        # Coefficients and odds ratios for linear models.
        estimator = pipeline.named_steps["model"]
        transformed_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
        if "polynomial" in pipeline.named_steps:
            transformed_names = list(pipeline.named_steps["polynomial"].get_feature_names_out(transformed_names))
        if hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_).reshape(-1)
            coef_df = pd.DataFrame({
                "transformed_feature": transformed_names[:len(coef)],
                "original_feature_group": [_original_feature(x) for x in transformed_names[:len(coef)]],
                "coefficient": coef,
            })
            if task == "churn":
                coef_df["odds_ratio_per_transformed_unit"] = np.exp(np.clip(coef, -30, 30))
                coef_df["interpretation_scale"] = "log-odds; odds ratio is for one transformed feature unit"
            else:
                coef_df["interpretation_scale"] = "predicted GBP change per transformed feature unit"
            coef_df = coef_df.sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
            coef_df.to_csv(report_dir / f"{task}_coefficients.csv", index=False)
            outputs.setdefault(task, {})["coefficients"] = str((report_dir / f"{task}_coefficients.csv").relative_to(project_path(".")))

        # Shallow decision-tree diagnostic and local paths are always produced.
        # If a different candidate won, this is explicitly labelled diagnostic-only.
        try:
            outputs.setdefault(task, {})["decision_tree_diagnostic"] = _write_decision_tree_diagnostic(
                task, frame, validation, numeric, categorical, target, config, report_dir,
                bundle["model_name"], bundle["model_version"],
            )
        except Exception as exc:
            outputs.setdefault(task, {})["decision_tree_diagnostic"] = {
                "status": "failed", "reason": f"{type(exc).__name__}: {exc}"
            }

        # SHAP summaries are optional and model-specific; unsupported models are documented.
        shap_status: dict[str, Any] = {"status": "not_run"}
        try:
            import shap
            transformed, shap_names = _transform_for_explanation(pipeline, X_sample.iloc[:min(250, len(X_sample))])
            if hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
                shap_exp = explainer(transformed)
            elif hasattr(estimator, "coef_"):
                background = transformed[:min(100, len(transformed))]
                explainer = shap.LinearExplainer(estimator, background)
                shap_exp = explainer(transformed)
            else:
                raise NotImplementedError("SHAP report is currently configured for linear and tree-based estimators; permutation importance is available for this model.")

            shap_values = np.asarray(shap_exp.values)
            if shap_values.ndim == 3:
                class_idx = 1 if shap_values.shape[-1] > 1 else 0
                shap_values = shap_values[:, :, class_idx]
                base_values = np.asarray(shap_exp.base_values)
                if base_values.ndim > 1:
                    base_values = base_values[:, class_idx]
            else:
                base_values = np.asarray(shap_exp.base_values)
            global_df = pd.DataFrame({
                "transformed_feature": shap_names,
                "original_feature_group": [_original_feature(x) for x in shap_names],
                "mean_absolute_shap": np.mean(np.abs(shap_values), axis=0),
            }).sort_values("mean_absolute_shap", ascending=False)
            global_path = report_dir / f"{task}_shap_global.csv"
            global_df.to_csv(global_path, index=False)

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                # Build a conventional Explanation for the binary class when necessary.
                if shap_values.ndim == 2:
                    explain_obj = shap.Explanation(
                        values=shap_values, base_values=base_values,
                        data=transformed, feature_names=shap_names,
                    )
                    shap.plots.beeswarm(explain_obj, max_display=20, show=False)
                    plt.gcf().set_size_inches(10, 7)
                    plt.tight_layout()
                    plt.savefig(report_dir / f"{task}_shap_beeswarm.png", dpi=160, bbox_inches="tight")
                    plt.close()
                    shap.plots.waterfall(explain_obj[0], max_display=15, show=False)
                    plt.gcf().set_size_inches(10, 7)
                    plt.tight_layout()
                    plt.savefig(report_dir / f"{task}_shap_local_waterfall.png", dpi=160, bbox_inches="tight")
                    plt.close()
            except Exception as exc:
                shap_status["plot_warning"] = str(exc)

            # Store local contributions as model explanation outputs with stable prediction IDs.
            local_rows = []
            for i in range(min(len(validation), shap_values.shape[0])):
                row = validation.iloc[i]
                pred_id = hashlib.sha256(f"{task}|{row['customer_id']}|{row['snapshot_date']}|{bundle['model_version']}".encode()).hexdigest()[:24]
                top_idx = np.argsort(np.abs(shap_values[i]))[::-1][:5]
                contributions = [
                    {"feature": _original_feature(shap_names[j]), "transformed_feature": shap_names[j], "shap_value": float(shap_values[i, j])}
                    for j in top_idx
                ]
                # Keep the SHAP output scale separate from the business output scale.
                # For logistic models, SHAP commonly explains raw margin/log-odds,
                # while the business score is the calibrated probability.
                base_array = np.asarray(base_values).reshape(-1)
                base = float(base_array[i]) if base_array.size > 1 else float(base_array[0])
                shap_reconstructed_output = float(base + np.sum(shap_values[i]))
                if task == "churn":
                    raw_p = float(_score_probability(pipeline, X.iloc[[i]])[0])
                    business_probability = float(bundle["calibrator"].predict(np.array([raw_p]))[0])
                    business_value = None
                    decision = int(business_probability >= float(bundle["threshold"]))
                else:
                    raw_prediction = float(pipeline.predict(X.iloc[[i]])[0])
                    business_probability = None
                    business_value = raw_prediction
                    decision = None
                local_rows.append({
                    "prediction_id": pred_id,
                    "customer_id": str(row["customer_id"]),
                    "prediction_timestamp": str(row["snapshot_date"]),
                    "model_version": bundle["model_version"],
                    "shap_base_output": base,
                    "shap_reconstructed_output": shap_reconstructed_output,
                    "business_prediction_probability": business_probability,
                    "business_prediction_value_gbp": business_value,
                    "business_decision_at_frozen_threshold": decision,
                    "frozen_threshold": bundle.get("threshold"),
                    "top_contributions_json": json.dumps(contributions),
                    "explainer_version": f"shap-{shap.__version__}",
                    "warning": "SHAP contributions explain the estimator output on its explained scale, not causation; calibrated probability/value is reported separately.",
                })
            local_path = report_dir / f"{task}_shap_local_contributions.csv"
            pd.DataFrame(local_rows).to_csv(local_path, index=False)
            shap_status.update({"status": "PASSED", "global_csv": str(global_path.relative_to(project_path("."))), "local_csv": str(local_path.relative_to(project_path("."))), "explainer_version": f"shap-{shap.__version__}", "transformed_to_original_mapping": {name: _original_feature(name) for name in shap_names}})
        except Exception as exc:
            shap_status.update({"status": "skipped_or_failed", "reason": f"{type(exc).__name__}: {exc}"})
        outputs.setdefault(task, {})["shap"] = shap_status

        # Partial dependence is run on the highest-importance numeric features only.
        try:
            top_features = [f for f in (perm_df["feature"].tolist() if not perm_df.empty else numeric) if f in numeric][:2]
            if top_features:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 5))
                X_pdp = X_sample.copy()
                for feature in numeric:
                    X_pdp[feature] = pd.to_numeric(X_pdp[feature], errors="coerce").astype(float)
                PartialDependenceDisplay.from_estimator(
                    pipeline, X_pdp, top_features, ax=ax,
                    response_method="predict_proba" if task == "churn" else "auto",
                )
                fig.tight_layout()
                fig.savefig(report_dir / f"{task}_partial_dependence.png", dpi=160)
                plt.close(fig)
                outputs.setdefault(task, {})["partial_dependence"] = f"reports/phase3/{task}_partial_dependence.png"
        except Exception as exc:
            outputs.setdefault(task, {})["partial_dependence_warning"] = str(exc)

        if task == "value":
            try:
                actual = validation[target].astype(float).to_numpy()
                predicted = pipeline.predict(X)
                residual = actual - predicted
                residuals = validation[["customer_id", "snapshot_date"]].copy()
                residuals["actual_net_value_90d_gbp"] = actual
                residuals["predicted_net_value_90d_gbp"] = predicted
                residuals["residual_gbp"] = residual
                residuals["absolute_error_gbp"] = np.abs(residual)
                if "country" in validation.columns:
                    residuals["country"] = validation["country"].astype("string").to_numpy()
                residual_path = report_dir / "value_residuals_validation.csv"
                residuals.to_csv(residual_path, index=False)
                if "country" in residuals.columns:
                    residuals.groupby("country", dropna=False).agg(
                        n=("residual_gbp", "size"),
                        mae_gbp=("residual_gbp", lambda values: float(np.mean(np.abs(values)))),
                        mean_residual_gbp=("residual_gbp", "mean"),
                        rmse_gbp=("residual_gbp", lambda values: float(np.sqrt(np.mean(np.square(values))))),
                    ).reset_index().to_csv(report_dir / "value_residuals_by_country_validation.csv", index=False)
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    axes[0].scatter(predicted, residual, s=12, alpha=0.45)
                    axes[0].axhline(0, linestyle="--")
                    axes[0].set_xlabel("Predicted net value (GBP)")
                    axes[0].set_ylabel("Residual (actual - predicted, GBP)")
                    axes[0].set_title("Validation residuals vs prediction")
                    axes[1].hist(residual, bins=30)
                    axes[1].set_xlabel("Residual (GBP)")
                    axes[1].set_ylabel("Count")
                    axes[1].set_title("Validation residual distribution")
                    fig.tight_layout()
                    fig.savefig(report_dir / "value_residual_diagnostics_validation.png", dpi=160)
                    plt.close(fig)
                except Exception as plot_exc:
                    outputs.setdefault(task, {})["residual_plot_warning"] = str(plot_exc)
                outputs.setdefault(task, {})["residual_diagnostics"] = str(residual_path.relative_to(project_path(".")))
            except Exception as exc:
                outputs.setdefault(task, {})["residual_diagnostics_warning"] = f"{type(exc).__name__}: {exc}"

        outputs[task].update({
            "status": "completed_with_review", "model_version": bundle["model_version"],
            "explanation_runtime_seconds": round(time.perf_counter() - task_started, 3),
            "validation_rows_used": int(len(validation)),
            "interpretation_warning": "Feature importance and SHAP are model explanations, not causal evidence. Review privacy, stability, and wording before exposing to business users.",
        })

    final_report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_runtime_seconds": round(time.perf_counter() - script_started, 3),
        "input_sha256": sha256_file(input_path), "tasks": outputs,
    }
    json_dump(report_dir / "explainability_manifest.json", final_report)
    print(f"Explainability outputs written under {report_dir}")
    print("Review permutation, coefficient/SHAP and partial-dependence outputs; explanations do not prove causality.")


if __name__ == "__main__":
    main()
