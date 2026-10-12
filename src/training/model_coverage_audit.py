"""Audit required PDF model-family coverage and flag any exclusions with reasons."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.training.common import ROOT, load_config, project_path, json_dump

REQUIRED_MODELS = {
    "churn": {
        "logistic_regression": "Interpretable churn baseline",
        "decision_tree": "Interpretable nonlinear churn model",
        "random_forest": "Robust nonlinear churn model",
        "xgboost": "Boosted-tree churn candidate",
        "lightgbm": "Efficient boosted-tree churn candidate",
        "svm": "Margin-based churn candidate",
        "knn": "Distance-based churn candidate",
    },
    "value": {
        "linear_regression": "Customer-value baseline",
        "ridge": "Regularized linear regression",
        "elastic_net": "Regularized linear regression",
        "random_forest_regressor": "Nonlinear customer-value regressor",
        "xgboost_regressor": "Boosted-tree customer-value regressor",
        "lightgbm_regressor": "Efficient boosted customer-value regressor",
    },
}
OPTIONAL_MODELS = {
    "churn": ["extra_trees", "adaboost", "gradient_boosting", "hist_gradient_boosting", "svm_rbf", "naive_bayes", "catboost", "voting"],
    "value": ["lasso", "polynomial_regression", "decision_tree_regressor", "extra_trees_regressor", "adaboost_regressor", "gradient_boosting_regressor", "hist_gradient_boosting_regressor", "catboost_regressor", "voting_regressor"],
}


def _rows_by_model(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path, low_memory=False)
    if "model" not in frame.columns:
        return {}
    # If a repeated candidate name exists, preserve the most recent row.
    result: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        result[str(row["model"])] = row.to_dict()
    return result


def main() -> None:
    config = load_config()
    report_dir = project_path(config["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for task in ("churn", "value"):
        table = report_dir / f"{task}_model_comparison.csv"
        available = _rows_by_model(table)
        for model, role in REQUIRED_MODELS[task].items():
            found = available.get(model)
            if found is None:
                status, reason = "NOT_RUN", f"Required candidate missing from {table.name}; rerun the supervised comparison."
            elif found.get("status") == "PASSED":
                status, reason = "TRAINED_AND_COMPARED", "Candidate trained; CV/validation metrics are in the comparison table."
            else:
                status = str(found.get("status", "UNKNOWN"))
                reason = str(found.get("error") or found.get("cv_errors") or "Review candidate row for details.")
            rows.append({"module": task, "model": model, "requirement": "required_by_pdf", "role": role, "status": status, "reason": reason})
        for model in OPTIONAL_MODELS[task]:
            found = available.get(model)
            if found is None:
                status = "OPTIONAL_NOT_RUN"
                reason = "Additional learning/checklist candidate; run --full-model-coverage if desired."
            elif found.get("status") == "PASSED":
                status = "TRAINED_AND_COMPARED"
                reason = "Optional candidate trained; results are in the comparison table."
            else:
                status = str(found.get("status", "UNKNOWN"))
                reason = str(found.get("error") or found.get("cv_errors") or "Review candidate row for details.")
            rows.append({"module": task, "model": model, "requirement": "optional_checklist", "role": "additional comparison", "status": status, "reason": reason})

    segmentation_manifest_path = report_dir / "segmentation_manifest.json"
    if segmentation_manifest_path.exists():
        seg = json.loads(segmentation_manifest_path.read_text(encoding="utf-8"))
        rows.extend([
            {"module": "segmentation", "model": "PCA", "requirement": "required_by_pdf", "role": "representation/visualization", "status": "TRAINED_AND_REPORTED", "reason": f"{seg.get('pca_components')} components; explained variance {seg.get('pca_explained_variance_total')}"},
            {"module": "segmentation", "model": "k-means", "requirement": "required_by_pdf", "role": "customer segmentation", "status": "TRAINED_AND_REPORTED", "reason": f"Provisional k={seg.get('selected_k_by_silhouette_supporting_metric')}; review profiles, stability and actionability before use."},
        ])
    else:
        for model in ("PCA", "k-means"):
            rows.append({"module": "segmentation", "model": model, "requirement": "required_by_pdf", "role": "segmentation", "status": "NOT_RUN", "reason": "Run python -m src.training.segment_customers."})

    seg_compare_path = report_dir / "segmentation_model_comparison.csv"
    seg_compare = _rows_by_model(seg_compare_path)
    for model, label in (("gmm_diag", "Gaussian Mixture comparison"), ("agglomerative", "hierarchical clustering comparison"), ("dbscan", "density clustering comparison")):
        found = seg_compare.get(model)
        rows.append({
            "module": "segmentation", "model": model, "requirement": "optional_comparison", "role": label,
            "status": "TRAINED_AND_REPORTED" if found and found.get("status") == "PASSED" else ("FAILED_OR_PARTIAL" if found else "NOT_RUN"),
            "reason": "Sampled exploratory comparison; not automatically selected for deployment." if found else "Run the segmentation module to generate the comparison.",
        })

    anomaly_path = report_dir / "anomaly_manifest.json"
    if anomaly_path.exists():
        anomaly = json.loads(anomaly_path.read_text(encoding="utf-8"))
        rows.append({"module": "anomaly", "model": "Isolation Forest", "requirement": "required_by_pdf", "role": "transaction/cust behaviour outlier score", "status": "TRAINED_AND_REPORTED", "reason": f"Alerts generated: {anomaly.get('alert_count')}; labels are not available for precision/fraud claims."})
        rows.append({"module": "anomaly", "model": "PCA reconstruction error", "requirement": "optional_comparison", "role": "reconstruction-based anomaly score", "status": anomaly.get("pca_reconstruction", {}).get("status", "NOT_RUN"), "reason": "Score is an unsupervised comparison, not a verified fraud label."})
        lof = anomaly.get("lof_comparison", {})
        rows.append({"module": "anomaly", "model": "LOF novelty", "requirement": "optional_comparison", "role": "density-based outlier comparison", "status": lof.get("status", "NOT_RUN"), "reason": lof.get("note") or lof.get("error") or "Optional comparison was disabled or has no scoring sample."})
    else:
        rows.append({"module": "anomaly", "model": "Isolation Forest", "requirement": "required_by_pdf", "role": "outlier scoring", "status": "NOT_RUN", "reason": "Run python -m src.training.detect_anomalies."})

    # Audit supporting project artifacts and explicit limitations.
    for name, path, why in [
        ("Optuna persistent study", project_path(config["optuna"]["storage"]), "Bounded tuning with SQLite persistence; run only after reviewing comparison tables."),
        ("Explainability report", report_dir / "explainability_manifest.json", "Contains baseline coefficient stability, permutation, SHAP where supported, and residual diagnostics."),
        ("Churn model card", report_dir / "model_card_churn.md", "Documents target, intended use, performance and limitations."),
        ("Value model card", report_dir / "model_card_value.md", "Documents target, intended use, performance and limitations."),
        ("Experiment log", report_dir / "experiments.jsonl", "Run lineage contains parameters, data hash, code commit when available, fold/validation metrics and artifact manifest."),
    ]:
        rows.append({"module": "ml_engineering", "model": name, "requirement": "blueprint_deliverable", "role": "reproducibility/explanation", "status": "PRESENT" if path.exists() else "NOT_YET_PRESENT", "reason": why})

    audit = pd.DataFrame(rows)
    audit_path = report_dir / "model_coverage_audit.csv"
    audit.to_csv(audit_path, index=False)
    json_dump(report_dir / "model_coverage_audit.json", {
        "required_model_families_from_pdf": REQUIRED_MODELS,
        "required_candidate_count": int(sum(len(v) for v in REQUIRED_MODELS.values())),
        "required_candidates_trained_and_compared": int(audit.loc[audit.requirement.eq("required_by_pdf") & audit.status.isin(["TRAINED_AND_COMPARED", "TRAINED_AND_REPORTED"]), "model"].nunique()),
        "audit_csv": str(audit_path.relative_to(ROOT)),
        "coverage_rows": rows,
        "limitations": [
            "The source dataset lacks verified churn/cancellation labels; churn is an inactivity proxy.",
            "Retention value requires a controlled experiment.",
            "Anomaly precision, investigator confirmation, false alerts and event value require investigation labels.",
            "The full-model-coverage option can report some optional models failed due to library compatibility or available resources; inspect row-level reasons.",
            "Leakage-safe stacking is deliberately excluded from this small-period pipeline; do not use random-fold stacking on repeated customer snapshots.",
        ],
    })
    print(f"Coverage audit saved: {audit_path}")
    required = audit.loc[audit.requirement.eq("required_by_pdf")]
    print(required[["module", "model", "status"]].to_string(index=False))
    failed_required = required.loc[~required.status.isin(["TRAINED_AND_COMPARED", "TRAINED_AND_REPORTED"])]
    if not failed_required.empty:
        print("WARNING: one or more required model families were not successfully trained/reported; inspect the audit reasons.")


if __name__ == "__main__":
    main()
