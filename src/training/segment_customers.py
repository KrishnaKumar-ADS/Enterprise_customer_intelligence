"""PCA + clustering, segment profiles, stability and optional comparisons."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.training.common import ROOT, load_config, project_path, json_dump, sha256_file

BASE_FEATURES = [
    "recency_days", "observed_tenure_days", "orders_365d", "units_365d",
    "spend_365d_gbp", "products_365d", "orders_all_time",
    "spend_all_time_gbp", "unique_products_all_time",
]
TIME_FEATURES = [
    "recency_days", "customer_tenure_days", "orders_30d", "spend_30d_gbp",
    "orders_90d", "spend_90d_gbp", "orders_180d", "spend_180d_gbp",
    "products_90d", "return_cancel_line_count_180d",
    "negative_quantity_line_count_180d", "negative_value_magnitude_180d_gbp",
]


def _numeric_transform(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    values = frame[cols].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    x = imputer.fit_transform(values)
    # RFM-style measures are typically strongly right-skewed. These columns
    # are counts or nonnegative values by construction in the chosen feature list.
    x = np.log1p(np.clip(x, 0, None))
    return StandardScaler().fit_transform(x)


def _best_k_metrics(x_pca: np.ndarray, cfg: dict[str, Any]) -> tuple[int, list[dict[str, Any]], np.ndarray]:
    min_k = int(cfg["candidate_k_min"])
    max_k = int(cfg["candidate_k_max"])
    max_k = min(max_k, len(x_pca) - 1)
    if max_k < min_k:
        raise ValueError("Not enough customer rows to evaluate the configured k range")
    rows = []
    fitted: dict[int, KMeans] = {}
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(x_pca)
        score = float(silhouette_score(x_pca, labels, sample_size=min(5000, len(x_pca)), random_state=42))
        sizes = pd.Series(labels).value_counts().sort_index().to_dict()
        rows.append({
            "model": "kmeans", "k": k, "silhouette": score,
            "inertia": float(model.inertia_),
            "minimum_cluster_size": int(min(sizes.values())),
            "maximum_cluster_size": int(max(sizes.values())),
            "cluster_sizes": {str(key): int(value) for key, value in sizes.items()},
        })
        fitted[k] = model
    # Silhouette is supporting evidence only; profiles and business usefulness still need review.
    best_k = max(rows, key=lambda item: item["silhouette"])["k"]
    return int(best_k), rows, fitted[best_k].labels_


def _time_stability(snapshot_path: Path, k: int, seed: int, max_dates: int) -> list[dict[str, Any]]:
    if not snapshot_path.exists():
        return [{"status": "not_available", "reason": "customer_snapshots.csv missing"}]
    frame = pd.read_csv(snapshot_path, low_memory=False)
    required = {"customer_id", "snapshot_date"}
    if not required.issubset(frame.columns):
        return [{"status": "not_available", "reason": "customer snapshot keys missing"}]
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    cols = [c for c in TIME_FEATURES if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    dates = sorted(frame["snapshot_date"].dropna().unique())
    dates = dates[-max_dates:]
    assignments: list[pd.DataFrame] = []
    for date in dates:
        month = frame.loc[frame["snapshot_date"].eq(date)].copy()
        if len(month) <= k or not cols:
            continue
        x = _numeric_transform(month, cols)
        n_components = min(x.shape[1], max(2, min(5, x.shape[1])))
        x = PCA(n_components=n_components, random_state=seed).fit_transform(x)
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        assignments.append(pd.DataFrame({
            "customer_id": month["customer_id"].astype(str).to_numpy(),
            "snapshot_date": pd.Timestamp(date), "cluster": labels,
        }))
    result = []
    for prior, current in zip(assignments, assignments[1:]):
        joined = prior.merge(current, on="customer_id", suffixes=("_prior", "_current"))
        score = float(adjusted_rand_score(joined["cluster_prior"], joined["cluster_current"])) if len(joined) >= 2 else None
        result.append({
            "from_snapshot": prior["snapshot_date"].iloc[0].date().isoformat(),
            "to_snapshot": current["snapshot_date"].iloc[0].date().isoformat(),
            "shared_customers": int(len(joined)),
            "adjusted_rand_index": score,
            "note": "Clusters are independently refit each month; ARI handles label permutation. Feature mix shifts may still affect interpretation.",
        })
    if not result:
        result.append({"status": "not_available", "reason": "Not enough populated snapshots for month-to-month stability"})
    return result


def main() -> None:
    config = load_config()
    cfg = config["segmentation"]
    segmentation_path = project_path(config["paths"]["customer_segmentation"])
    snapshot_path = project_path(config["paths"]["customer_snapshots"])
    output_dir = project_path(config["paths"]["reports_dir"])
    model_dir = project_path(config["paths"]["models_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not segmentation_path.exists():
        raise FileNotFoundError(f"Missing {segmentation_path}. Run Phase 2 first.")
    frame = pd.read_csv(segmentation_path, low_memory=False)
    cols = [c for c in BASE_FEATURES if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    if len(cols) < 3:
        raise ValueError(f"Not enough numeric segmentation features. Found: {cols}")
    if "customer_id" not in frame.columns:
        raise ValueError("customer_id missing from segmentation dataset")
    if frame["customer_id"].duplicated().any():
        raise ValueError("Segmentation input must contain one row per customer")

    x_scaled = _numeric_transform(frame, cols)
    pca = PCA(n_components=float(cfg["variance_to_retain"]), random_state=int(config["project"]["random_seed"]))
    x_reduced = pca.fit_transform(x_scaled)
    best_k, k_rows, _ = _best_k_metrics(x_reduced, cfg)
    seed = int(config["project"]["random_seed"])
    chosen = KMeans(n_clusters=best_k, n_init=20, random_state=seed)
    labels = chosen.fit_predict(x_reduced)

    assignments = frame[["customer_id"]].copy()
    assignments["segment_id"] = labels.astype(int)
    assignments["segmentation_model"] = "PCA+KMeans"
    assignments["k"] = best_k
    assignments.to_csv(project_path("data/processed/customer_segments.csv"), index=False)

    # Segment size and behavioural profiles in original, interpretable units.
    profile_source = frame.copy()
    profile_source["segment_id"] = labels
    profile_numeric = [c for c in cols if c in profile_source.columns]
    profile_rows = []
    for segment_id, group in profile_source.groupby("segment_id"):
        row: dict[str, Any] = {
            "segment_id": int(segment_id), "customer_count": int(len(group)),
            "customer_share": float(len(group) / len(frame)),
        }
        for col in profile_numeric:
            row[f"median_{col}"] = float(pd.to_numeric(group[col], errors="coerce").median())
            row[f"mean_{col}"] = float(pd.to_numeric(group[col], errors="coerce").mean())
        if "country" in group.columns:
            country = group["country"].dropna().mode()
            row["most_common_country"] = str(country.iloc[0]) if not country.empty else None
        row["proposed_action"] = ""
        row["intended_use_case"] = ""
        row["business_owner"] = ""
        row["business_review_status"] = "pending_human_review"
        profile_rows.append(row)
    pd.DataFrame(profile_rows).to_csv(output_dir / "segment_profiles.csv", index=False)
    pd.DataFrame(k_rows).to_csv(output_dir / "segmentation_k_comparison.csv", index=False)

    # Initialization stability on the same population and PCA representation.
    stability_rows = []
    for run_seed in range(seed, seed + int(cfg["seed_repeats"])):
        alternate = KMeans(n_clusters=best_k, n_init=20, random_state=run_seed).fit_predict(x_reduced)
        stability_rows.append({
            "seed": run_seed,
            "adjusted_rand_index_vs_primary": float(adjusted_rand_score(labels, alternate)),
            "cluster_sizes": {str(key): int(value) for key, value in pd.Series(alternate).value_counts().sort_index().items()},
        })
    time_stability = _time_stability(snapshot_path, best_k, seed, int(cfg["time_stability_max_dates"]))
    pd.DataFrame(stability_rows).to_csv(output_dir / "segmentation_seed_stability.csv", index=False)
    pd.DataFrame(time_stability).to_csv(output_dir / "segmentation_time_stability.csv", index=False)

    # Optional model comparisons. Hierarchical is evaluated on a deterministic sample
    # due its quadratic memory/time cost; GMM and DBSCAN are exploratory diagnostics.
    comparison = [{**row, "status": "PASSED"} for row in k_rows]
    sample_n = min(3000, len(x_reduced))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x_reduced), size=sample_n, replace=False)) if sample_n < len(x_reduced) else np.arange(len(x_reduced))
    x_sample = x_reduced[idx]
    for k in range(int(cfg["candidate_k_min"]), min(int(cfg["candidate_k_max"]), len(x_sample) - 1) + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed, n_init=2)
            gmm_labels = gmm.fit_predict(x_sample)
            sil = float(silhouette_score(x_sample, gmm_labels, sample_size=min(2000, len(x_sample)), random_state=seed))
            comparison.append({"model": "gmm_diag", "k": k, "silhouette": sil, "bic": float(gmm.bic(x_sample)), "aic": float(gmm.aic(x_sample)), "status": "PASSED", "sample_rows": int(len(x_sample))})
        except Exception as exc:
            comparison.append({"model": "gmm_diag", "k": k, "status": "FAILED", "error": str(exc)})
    for linkage in ("ward", "average"):
        try:
            ag = AgglomerativeClustering(n_clusters=best_k, linkage=linkage)
            ag_labels = ag.fit_predict(x_sample)
            comparison.append({"model": "agglomerative", "linkage": linkage, "k": best_k, "silhouette": float(silhouette_score(x_sample, ag_labels, sample_size=min(2000, len(x_sample)), random_state=seed)), "status": "PASSED", "sample_rows": int(len(x_sample))})
        except Exception as exc:
            comparison.append({"model": "agglomerative", "linkage": linkage, "k": best_k, "status": "FAILED", "error": str(exc)})
    for eps in (0.5, 1.0, 1.5, 2.0):
        try:
            db = DBSCAN(eps=eps, min_samples=10, n_jobs=-1)
            db_labels = db.fit_predict(x_sample)
            non_noise = db_labels != -1
            n_clusters = len(set(db_labels[non_noise]))
            sil = float(silhouette_score(x_sample[non_noise], db_labels[non_noise], sample_size=min(2000, int(non_noise.sum())), random_state=seed)) if n_clusters >= 2 and non_noise.sum() > n_clusters else None
            comparison.append({"model": "dbscan", "eps": eps, "min_samples": 10, "n_clusters_excluding_noise": int(n_clusters), "noise_fraction": float(np.mean(db_labels == -1)), "silhouette_excluding_noise": sil, "status": "PASSED", "sample_rows": int(len(x_sample))})
        except Exception as exc:
            comparison.append({"model": "dbscan", "eps": eps, "status": "FAILED", "error": str(exc)})
    pd.DataFrame(comparison).to_csv(output_dir / "segmentation_model_comparison.csv", index=False)

    pca_df = pd.DataFrame({f"PC{i+1}": x_reduced[:, i] for i in range(min(2, x_reduced.shape[1]))})
    pca_df["customer_id"] = frame["customer_id"].astype(str).to_numpy()
    pca_df["segment_id"] = labels
    pca_df.to_csv(project_path("data/processed/customer_pca_projection.csv"), index=False)
    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(output_dir / "pca_variance.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if x_reduced.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            points = ax.scatter(x_reduced[:, 0], x_reduced[:, 1], c=labels, s=10, alpha=0.65)
            ax.set_xlabel("Principal component 1")
            ax.set_ylabel("Principal component 2")
            ax.set_title(f"Customer segments in PCA space (k={best_k})")
            fig.colorbar(points, ax=ax, label="Segment ID")
            fig.tight_layout()
            fig.savefig(output_dir / "customer_segments_pca.png", dpi=160)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker="o")
        ax.axhline(float(cfg["variance_to_retain"]), linestyle="--")
        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title("PCA explained variance")
        fig.tight_layout()
        fig.savefig(output_dir / "pca_explained_variance.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        warnings.warn(f"Could not write segmentation plots: {exc}")

    report = {
        "task": "customer_segmentation", "input_path": str(segmentation_path.relative_to(ROOT)),
        "input_sha256": sha256_file(segmentation_path), "customers": int(len(frame)),
        "features": cols, "pca_components": int(x_reduced.shape[1]),
        "pca_explained_variance_total": float(pca.explained_variance_ratio_.sum()),
        "pca_explained_variance_by_component": pca.explained_variance_ratio_.tolist(),
        "selected_k_by_silhouette_supporting_metric": int(best_k),
        "primary_silhouette": next(row["silhouette"] for row in k_rows if row["k"] == best_k),
        "cluster_size_counts": {str(key): int(value) for key, value in pd.Series(labels).value_counts().sort_index().items()},
        "seed_stability": stability_rows,
        "month_to_month_time_stability": time_stability,
        "comparisons": ["PCA+KMeans primary", "Gaussian Mixture diagonal covariance", "Agglomerative hierarchical (sampled)", "DBSCAN density clustering (sampled)"],
        "actionability_note": "Silhouette is supporting evidence. Review segment profiles, sizes, time stability and operational actionability before adopting segments.",
        "outputs": {
            "assignments": "data/processed/customer_segments.csv",
            "pca_projection": "data/processed/customer_pca_projection.csv",
            "segment_profiles": "reports/phase3/segment_profiles.csv",
            "model_comparison": "reports/phase3/segmentation_model_comparison.csv",
        },
    }
    json_dump(output_dir / "segmentation_manifest.json", report)
    print(f"Segmentation complete: {len(frame):,} customers, k={best_k}, silhouette={report['primary_silhouette']:.4f}")
    print("Review segment_profiles.csv and month_to_month_time_stability before treating clusters as business segments.")


if __name__ == "__main__":
    main()
