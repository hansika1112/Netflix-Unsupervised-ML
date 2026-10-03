"""Reproducible end-to-end model run for the supplied catalog."""

from __future__ import annotations
import json
import platform
import sklearn
import matplotlib
import seaborn
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from clustering import reduce_tfidf, fit_kmeans, hierarchical_linkage
from evaluation import evaluate_kmeans_range, clustering_metrics
from feature_engineering import make_text_features, add_metadata_features
from preprocessing import vectorize_text
from visualization import (plot_cluster_map, plot_cluster_sizes,
                           plot_dendrogram, plot_k_selection,
                           plot_top_terms, cluster_profiles)


def runtime_versions():
    """Return actual interpreter and core package versions for this run."""
    return {"python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "seaborn": seaborn.__version__,
            "joblib": joblib.__version__}


def run_models(clean_df: pd.DataFrame, project_dir: str | Path,
               k_values=range(2, 11), agglomerative_sample_size=1500,
               random_state=42):
    """Run TF-IDF/SVD, K-Means selection, and sampled hierarchical clustering."""
    root = Path(project_dir)
    figures, tables, models_dir = (root / "outputs/figures", root / "outputs/tables",
                                   root / "outputs/models")
    for path in (figures, tables, models_dir):
        path.mkdir(parents=True, exist_ok=True)

    featured = add_metadata_features(make_text_features(clean_df)).reset_index(drop=True)
    tfidf, tfidf_vectorizer, bow, bow_vectorizer = vectorize_text(featured["normalized_text"])
    Z, svd = reduce_tfidf(tfidf, n_components=100, random_state=random_state)
    selection, candidate_models = evaluate_kmeans_range(
        Z, k_values=k_values, sample_size=1000, random_state=random_state)
    selection.to_csv(tables / "kmeans_k_selection.csv", index=False)
    plot_k_selection(selection, figures)

    # When candidate scores are within 1% of the maximum, prefer the smaller K.
    best_score = selection["silhouette_sample"].max()
    eligible = selection.loc[selection["silhouette_sample"] >= best_score * 0.99, "k"]
    selected_k = int(eligible.min())
    kmeans, km_labels = fit_kmeans(Z, selected_k, random_state=random_state)

    # Compare both algorithms on exactly the same reproducible sample.
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(len(featured), min(agglomerative_sample_size, len(featured)), replace=False)
    Z_sample = Z[sample_idx]
    km_sample_labels = km_labels[sample_idx]
    linkage_candidates = []
    candidate_hierarchy = {}
    for linkage_name, metric_name in [("ward", "euclidean"), ("average", "cosine")]:
        for clusters in sorted(set([max(2, selected_k - 2), max(2, selected_k - 1),
                                    selected_k, selected_k + 1])):
            if linkage_name == "ward":
                model = AgglomerativeClustering(n_clusters=clusters, linkage="ward")
            else:
                model = AgglomerativeClustering(n_clusters=clusters, linkage="average", metric="cosine")
            labels = model.fit_predict(Z_sample)
            score = silhouette_score(Z_sample, labels, metric="cosine")
            candidate_hierarchy[(linkage_name, clusters)] = (model, labels)
            linkage_candidates.append({"linkage": linkage_name, "metric": metric_name,
                                       "n_clusters": clusters, "silhouette_cosine": float(score)})
    hierarchy_table = pd.DataFrame(linkage_candidates).sort_values(
        ["silhouette_cosine", "n_clusters"], ascending=[False, True])
    hierarchy_table.to_csv(tables / "agglomerative_selection.csv", index=False)
    best_row = hierarchy_table.iloc[0]
    selected_linkage, hierarchy_k = str(best_row["linkage"]), int(best_row["n_clusters"])
    ag_model, ag_sample_labels = candidate_hierarchy[(selected_linkage, hierarchy_k)]

    # Matched-K comparison is deliberately distinct from each method's selected
    # configuration. Both labelings cover every row of the exact same sample.
    matched_ag_model, matched_ag_labels = candidate_hierarchy[("ward", selected_k)]
    same_k_rows = [
        clustering_metrics(Z_sample, km_sample_labels, f"K-Means (K={selected_k})",
                           len(Z_sample), random_state),
        clustering_metrics(Z_sample, matched_ag_labels,
                           f"Agglomerative Ward (K={selected_k})",
                           len(Z_sample), random_state),
    ]
    same_k_table = pd.DataFrame(same_k_rows)
    same_k_table.insert(1, "n_clusters_requested", selected_k)
    same_k_table.to_csv(tables / "same_k_model_comparison.csv", index=False)

    km_metrics = clustering_metrics(Z_sample, km_sample_labels, "K-Means", len(Z_sample), random_state)
    ag_metrics = clustering_metrics(Z_sample, ag_sample_labels,
                                    f"Agglomerative ({selected_linkage})", len(Z_sample), random_state)
    metric_rows = [km_metrics, ag_metrics]
    metric_table = pd.DataFrame(metric_rows)
    metric_table.insert(1, "n_clusters_requested", [selected_k, hierarchy_k])
    metric_table["inertia"] = [float(kmeans.inertia_), np.nan]
    metric_table.to_csv(tables / "model_comparison.csv", index=False)

    featured["cluster"] = km_labels
    featured[[c for c in ["show_id", "title", "type", "country", "rating", "listed_in", "cluster"]
              if c in featured]].to_csv(tables / "kmeans_cluster_assignments.csv", index=False)
    pd.DataFrame({"row_index": sample_idx, "show_id": featured.iloc[sample_idx]["show_id"].to_numpy(),
                  "title": featured.iloc[sample_idx]["title"].to_numpy(),
                  "cluster": ag_sample_labels}).to_csv(tables / "agglomerative_sample_assignments.csv", index=False)

    profiles = cluster_profiles(featured, km_labels, tfidf, tfidf_vectorizer)
    # Map the reduced-space K-Means centroids back to an approximate TF-IDF
    # vocabulary space so the model's own centers can be explained in words.
    approximate_centers = svd.inverse_transform(kmeans.cluster_centers_)
    vocabulary = np.asarray(tfidf_vectorizer.get_feature_names_out())
    for profile in profiles:
        center = approximate_centers[profile["cluster"]]
        best = np.argsort(center)[::-1][:12]
        profile["kmeans_centroid_terms_approx"] = [vocabulary[i] for i in best if center[i] > 0]
    (tables / "cluster_profiles.json").write_text(json.dumps(profiles, ensure_ascii=False, indent=2))
    pd.DataFrame([{key: value for key, value in profile.items() if key != "description_samples"}
                  for profile in profiles]).to_csv(tables / "cluster_profiles.csv", index=False)
    plot_cluster_sizes(km_labels, figures)
    plot_top_terms(profiles, figures)
    plot_cluster_map(Z[:, :2], km_labels, figures)
    plot_cluster_map(Z_sample[:, :2], ag_sample_labels, figures,
                     filename="agglomerative_cluster_visualization.png",
                     title=f"Agglomerative Clusters ({selected_linkage}, k={hierarchy_k})")

    dendro_rng = np.random.default_rng(random_state)
    dendro_idx = dendro_rng.choice(len(Z_sample), min(300, len(Z_sample)), replace=False)
    linkage_matrix = hierarchical_linkage(Z_sample[dendro_idx], linkage="ward")
    plot_dendrogram(linkage_matrix, figures, len(dendro_idx))

    versions = runtime_versions()
    (tables / "runtime_versions.json").write_text(json.dumps(versions, indent=2))
    joblib.dump({"kmeans": kmeans, "tfidf_vectorizer": tfidf_vectorizer,
                 "svd": svd, "selected_k": selected_k,
                 "runtime_versions": versions,
                 "feature_columns": ["title", "director", "cast", "country", "listed_in", "description"]},
                models_dir / "netflix_kmeans.joblib", compress=3)
    joblib.dump({"agglomerative_model": ag_model, "linkage": selected_linkage,
                 "n_clusters": hierarchy_k, "training_sample_indices": sample_idx,
                 "runtime_versions": versions},
                models_dir / "netflix_agglomerative_sample.joblib", compress=3)

    return {"featured": featured, "tfidf": tfidf, "tfidf_vectorizer": tfidf_vectorizer,
            "bow": bow, "bow_vectorizer": bow_vectorizer, "svd_features": Z,
            "svd": svd, "kmeans": kmeans, "kmeans_labels": km_labels,
            "agglomerative": ag_model, "agglomerative_labels": ag_sample_labels,
            "agglomerative_sample_indices": sample_idx,
            "k_selection": selection, "agglomerative_selection": hierarchy_table,
            "metrics": metric_table, "same_k_metrics": same_k_table,
            "same_k_agglomerative": matched_ag_model,
            "same_k_agglomerative_labels": matched_ag_labels,
            "runtime_versions": versions, "profiles": profiles,
            "selected_k": selected_k, "selected_linkage": selected_linkage,
            "hierarchy_k": hierarchy_k, "svd_variance_explained": float(svd.explained_variance_ratio_.sum())}
