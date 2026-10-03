"""Unsupervised model selection and comparative evaluation."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)


def evaluate_kmeans_range(X, k_values=range(2, 11), sample_size=1000, random_state=42):
    """Fit candidate KMeans models and calculate inertia plus sampled silhouette."""
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], min(sample_size, X.shape[0]), replace=False)
    records, models = [], {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state,
                       max_iter=300, algorithm="lloyd")
        labels = model.fit_predict(X)
        sample_labels = labels[indices]
        score = silhouette_score(X[indices], sample_labels, metric="cosine")
        records.append({"k": k, "inertia": float(model.inertia_), "silhouette_sample": float(score)})
        models[k] = model
    return pd.DataFrame(records), models


def clustering_metrics(X, labels, algorithm, metric_sample_size=2500, random_state=42):
    """Compute complementary metrics on a reproducible common-size sample."""
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], min(metric_sample_size, X.shape[0]), replace=False)
    xs = X[idx]
    ys = np.asarray(labels)[idx]
    values = {"algorithm": algorithm, "n_evaluated": len(idx),
              "n_clusters_observed": int(np.unique(ys).size)}
    if np.unique(ys).size < 2 or np.unique(ys).size >= len(ys):
        values.update({"silhouette_cosine": np.nan, "davies_bouldin_euclidean": np.nan,
                       "calinski_harabasz_euclidean": np.nan})
    else:
        values["silhouette_cosine"] = float(silhouette_score(xs, ys, metric="cosine"))
        # DB and CH use Euclidean geometry; convert the sparse sample only when
        # the feature count is modest, otherwise use the already-reduced matrix.
        dense = xs.toarray() if hasattr(xs, "toarray") else np.asarray(xs)
        values["davies_bouldin_euclidean"] = float(davies_bouldin_score(dense, ys))
        values["calinski_harabasz_euclidean"] = float(calinski_harabasz_score(dense, ys))
    return values
