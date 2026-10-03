"""Model selection and final K-Means/Agglomerative clustering."""

from __future__ import annotations
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def reduce_tfidf(tfidf, n_components=100, random_state=42):
    """Project sparse TF-IDF into a compact, normalized semantic space."""
    n_components = min(n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if n_components < 2:
        raise ValueError("TF-IDF matrix is too small for dimensionality reduction")
    svd = TruncatedSVD(n_components=n_components, random_state=random_state,
                       n_iter=7, algorithm="randomized")
    reduced = svd.fit_transform(tfidf)
    return normalize(reduced), svd


def fit_kmeans(X, k, random_state=42):
    """Fit deterministic K-Means and return model plus labels."""
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state,
                   max_iter=300, algorithm="lloyd")
    return model, model.fit_predict(X)


def fit_agglomerative(X, k, linkage="ward"):
    """Fit hierarchical clustering on the explicit sample supplied by caller."""
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage,
                                    compute_full_tree=True)
    return model, model.fit_predict(X)


def hierarchical_linkage(X, linkage="ward"):
    """Return scipy linkage matrix for a representative, caller-sized sample."""
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    return scipy_linkage(X, method=linkage, metric="euclidean",
                         optimal_ordering=False)
