"""Fit outlier detectors and produce an auditable human-review queue."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.training.common import ROOT, load_config, project_path, json_dump, sha256_file

FEATURE_CANDIDATES = [
    "quantity", "unit_price", "line_value", "invoice_line_count",
    "invoice_net_value_gbp", "invoice_absolute_line_value_gbp",
    "invoice_hour", "invoice_dayofweek", "invoice_month",
    "is_cancellation", "is_negative_quantity", "is_valid_timestamp",
    "is_valid_quantity", "is_valid_unit_price", "is_valid_numeric_line",
]


def main() -> None:
    config = load_config()
    cfg = config["anomaly"]
    input_path = project_path(config["paths"]["anomaly_transactions"])
    report_dir = project_path(config["paths"]["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run Phase 2 first.")

    frame = pd.read_csv(input_path, low_memory=False)
    if "invoice_date" not in frame.columns:
        raise ValueError("invoice_date is missing from anomaly transaction data")
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
    frame["line_value"] = pd.to_numeric(frame.get("line_value"), errors="coerce")
    feature_cols = [c for c in FEATURE_CANDIDATES if c in frame.columns]
    if len(feature_cols) < 5:
        raise ValueError(f"Not enough anomaly features; available: {feature_cols}")

    # Reference population is a chronological prefix; later rows are scored out of time.
    known_dates = sorted(frame.loc[frame.invoice_date.notna(), "invoice_date"].dt.normalize().unique())
    if len(known_dates) < 10:
        raise ValueError("Not enough distinct transaction dates for chronological anomaly reference split")
    cut_idx = min(max(1, int(len(known_dates) * float(cfg["reference_fraction"]))), len(known_dates) - 1)
    cutoff_date = pd.Timestamp(known_dates[cut_idx])
    reference_mask = frame["invoice_date"].lt(cutoff_date)
    if int(reference_mask.sum()) < 100:
        raise ValueError("Anomaly reference population is unexpectedly small")

    for col in ("is_cancellation", "is_negative_quantity", "is_valid_timestamp", "is_valid_quantity", "is_valid_unit_price", "is_valid_numeric_line"):
        if col in frame.columns:
            frame[col] = frame[col].fillna(False).astype(int)
    x_frame = frame[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x_ref = imputer.fit_transform(x_frame.loc[reference_mask])
    x_all = imputer.transform(x_frame)

    seed = int(config["project"]["random_seed"])
    contamination = float(cfg["contamination"])
    detector = IsolationForest(
        n_estimators=300, max_samples="auto", contamination=contamination,
        random_state=seed, n_jobs=-1,
    )
    detector.fit(x_ref)
    # sklearn decision_function: lower values are more anomalous.
    frame["anomaly_score_isolation_forest"] = -detector.decision_function(x_all)

    # PCA reconstruction error: fit scaler/PCA on reference only.
    pca_status = "not_run"
    if bool(cfg.get("pca_reconstruction_enabled", True)):
        scaler = StandardScaler()
        ref_scaled = scaler.fit_transform(x_ref)
        all_scaled = scaler.transform(x_all)
        pca = PCA(n_components=0.95, svd_solver="full", random_state=seed)
        pca.fit(ref_scaled)
        reconstructed = pca.inverse_transform(pca.transform(all_scaled))
        frame["anomaly_score_pca_reconstruction"] = np.mean((all_scaled - reconstructed) ** 2, axis=1)
        pca_status = "PASSED"

    # LOF's distance computations are expensive. Train on a bounded reference sample
    # and score a bounded deterministic comparison sample to keep local compute usable.
    lof_status: dict[str, Any] = {"status": "disabled"}
    if bool(cfg.get("local_outlier_factor_enabled", True)):
        rng = np.random.default_rng(seed)
        ref_positions = np.flatnonzero(reference_mask.to_numpy())
        fit_n = min(4000, len(ref_positions))
        fit_positions = np.sort(rng.choice(ref_positions, size=fit_n, replace=False))
        future_positions = np.flatnonzero((~reference_mask & frame.invoice_date.notna()).to_numpy())
        score_n = min(10000, len(future_positions))
        score_positions = np.sort(rng.choice(future_positions, size=score_n, replace=False)) if score_n else np.array([], dtype=int)
        frame["anomaly_score_lof_sample"] = np.nan
        if fit_n > 50 and score_n:
            try:
                lof_scaler = StandardScaler()
                lof_scaled_ref = lof_scaler.fit_transform(x_all[fit_positions])
                lof_scaled_later = lof_scaler.transform(x_all[score_positions])
                lof = LocalOutlierFactor(n_neighbors=min(35, fit_n - 1), novelty=True, contamination=contamination, n_jobs=-1)
                lof.fit(lof_scaled_ref)
                frame.loc[frame.index[score_positions], "anomaly_score_lof_sample"] = -lof.score_samples(lof_scaled_later)
                lof_status = {"status": "PASSED", "fit_reference_rows": int(fit_n), "scored_later_rows": int(score_n), "preprocessor": "StandardScaler fitted only on sampled reference rows", "note": "LOF scores are populated only for the bounded evaluation sample."}
            except Exception as exc:
                lof_status = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}

    # Top fraction is a review-capacity proxy, not a precision/recall estimate.
    score = frame["anomaly_score_isolation_forest"]
    later_scored_mask = (~reference_mask) & frame["invoice_date"].notna()
    later_scores = score.loc[later_scored_mask]
    if later_scores.empty:
        raise ValueError("No timestamped later-period transactions are available for alert ranking")
    n_alerts = max(1, int(np.ceil(len(later_scores) * float(cfg["alert_fraction"]))))
    threshold = float(later_scores.nlargest(n_alerts).min())
    frame["isolation_forest_alert"] = False
    later_rank = later_scores.rank(method="first", ascending=False)
    frame.loc[later_scored_mask, "isolation_forest_alert"] = later_rank.le(n_alerts).to_numpy()
    frame["anomaly_reference_role"] = np.where(reference_mask, "reference_fit", "later_scored")
    scores_path = project_path("data/processed/anomaly_scores.csv")
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(scores_path, index=False)

    review_n = min(int(cfg["review_queue_rows"]), int(frame["isolation_forest_alert"].sum()))
    review_queue = frame.loc[frame["isolation_forest_alert"]].nlargest(review_n, "anomaly_score_isolation_forest").copy()
    keep_cols = [c for c in [
        "invoice_no", "customer_id", "invoice_date", "stock_code", "description",
        "quantity", "unit_price", "line_value", "invoice_line_count", "invoice_net_value_gbp",
        "is_cancellation", "is_negative_quantity", "anomaly_score_isolation_forest",
        "anomaly_score_pca_reconstruction", "anomaly_score_lof_sample",
    ] if c in review_queue.columns]
    review_queue = review_queue[keep_cols].copy()
    review_queue["review_status"] = "pending"
    review_queue["investigator_confirmed"] = pd.NA
    review_queue["confirmed_event_value_gbp"] = pd.NA
    review_queue["review_notes"] = pd.NA
    review_path = project_path("data/processed/anomaly_review_queue.csv")
    review_queue.to_csv(review_path, index=False)

    later = frame.loc[~reference_mask & frame.invoice_date.notna()].copy()
    daily = frame.loc[frame["isolation_forest_alert"] & frame.invoice_date.notna()].groupby(frame.loc[frame["isolation_forest_alert"] & frame.invoice_date.notna(), "invoice_date"].dt.date).size()
    mean_alerts_per_observed_day = float(daily.mean()) if len(daily) else 0.0
    score_corr = None
    if "anomaly_score_pca_reconstruction" in frame.columns:
        corr = frame[["anomaly_score_isolation_forest", "anomaly_score_pca_reconstruction"]].corr().iloc[0, 1]
        score_corr = float(corr) if pd.notna(corr) else None

    report = {
        "task": "anomaly_detection", "input_path": str(input_path.relative_to(ROOT)),
        "input_sha256": sha256_file(input_path), "transaction_rows": int(len(frame)),
        "feature_columns": feature_cols,
        "reference_fraction": float(cfg["reference_fraction"]),
        "reference_cutoff_exclusive": cutoff_date.isoformat(),
        "reference_rows": int(reference_mask.sum()), "later_scored_rows": int((~reference_mask & frame.invoice_date.notna()).sum()),
        "detector": "IsolationForest", "contamination_parameter": contamination,
        "alert_fraction": float(cfg["alert_fraction"]), "alert_threshold_score": threshold,
        "alert_fraction_applies_to": "timestamped later-scored population only",
        "alert_count": int(frame["isolation_forest_alert"].sum()),
        "alert_count_per_observed_day_mean": mean_alerts_per_observed_day,
        "pca_reconstruction": {"status": pca_status, "score_correlation_with_isolation_forest": score_corr},
        "lof_comparison": lof_status,
        "operational_evaluation": {
            "precision_among_top_alerts": None,
            "investigator_confirmed_rate": None,
            "false_alerts_per_day": None,
            "event_value_recovered_gbp": None,
            "detection_delay": None,
            "reason": "The source dataset has no verified anomaly/fraud investigation labels or event-resolution timestamps. These metrics require reviewed alerts and outcome data.",
        },
        "review_queue": str(review_path.relative_to(ROOT)),
        "outputs": {"scores": str(scores_path.relative_to(ROOT)), "review_queue": str(review_path.relative_to(ROOT))},
        "warning": "Anomaly scores mean unusual relative to the chosen reference population; they do not prove fraud or misconduct.",
    }
    json_dump(report_dir / "anomaly_manifest.json", report)
    print(f"Anomaly detection complete. Rows scored: {len(frame):,}; alerts: {report['alert_count']:,}")
    print(f"Review queue: {review_path}")
    print("Precision, investigator-confirmed rate, false alerts and recovered value remain unknown until investigators review alerts.")


if __name__ == "__main__":
    main()
