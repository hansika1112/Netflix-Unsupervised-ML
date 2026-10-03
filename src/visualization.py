"""Plots and content-oriented cluster summaries."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def save_fig(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_k_selection(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(results["k"], results["inertia"], marker="o", color="#d92b36")
    axes[0].set(title="K-Means Elbow Curve", xlabel="Number of clusters (K)", ylabel="Inertia")
    axes[1].plot(results["k"], results["silhouette_sample"], marker="o", color="#333333")
    axes[1].set(title="Sampled Silhouette by K", xlabel="Number of clusters (K)", ylabel="Silhouette score (cosine)")
    save_fig(fig, output, "elbow_and_silhouette.png")


def plot_cluster_map(X_2d, labels, output, filename="cluster_visualization.png", title="Netflix Title Clusters"):
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=11, alpha=.65)
    ax.set(title=title, xlabel="SVD component 1", ylabel="SVD component 2")
    ax.legend(*scatter.legend_elements(num=None), title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, output, filename)


def plot_cluster_sizes(labels, output):
    counts = pd.Series(labels).value_counts().sort_index()
    share = counts / counts.sum() * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#d92b36")
    ax.set(title="K-Means Cluster Sizes", xlabel="Cluster ID", ylabel="Titles")
    for bar, pct in zip(bars, share.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{pct:.1f}%", ha="center", va="bottom")
    save_fig(fig, output, "cluster_sizes.png")


def cluster_profiles(df, labels, tfidf, vectorizer, top_n_terms=12, top_n_titles=5):
    """Summarize actual catalog metadata and TF-IDF evidence for each cluster."""
    work = df.reset_index(drop=True).copy()
    work["cluster"] = np.asarray(labels)
    terms = np.asarray(vectorizer.get_feature_names_out())
    profiles = []
    for cluster_id in sorted(work["cluster"].unique()):
        idx = np.flatnonzero(work["cluster"].to_numpy() == cluster_id)
        subset = work.iloc[idx]
        mean_vector = np.asarray(tfidf[idx].mean(axis=0)).ravel()
        top_idx = np.argsort(mean_vector)[::-1][:top_n_terms]
        terms_out = [terms[i] for i in top_idx if mean_vector[i] > 0]
        centroid = csr_matrix(tfidf[idx].mean(axis=0))
        similarities = cosine_similarity(tfidf[idx], centroid).ravel()
        reps = subset.iloc[np.argsort(similarities)[::-1][:top_n_titles]]["title"].tolist()

        def top_split(field, n=5):
            if field not in subset:
                return []
            values = subset[field].dropna().astype(str).str.split(", ").explode().str.strip()
            return values[values.ne("")].value_counts().head(n).index.tolist()

        type_counts = subset["type"].value_counts(normalize=True).mul(100).round(1).to_dict()
        profiles.append({
            "cluster": int(cluster_id), "size": int(len(subset)),
            "percentage": round(len(subset) / len(work) * 100, 2),
            "content_type_share_pct": type_counts,
            "common_countries": top_split("country"), "common_ratings": top_split("rating"),
            "common_genres": top_split("listed_in"), "important_tfidf_terms": terms_out,
            "representative_titles": reps,
            "description_samples": subset["description"].dropna().head(3).tolist() if "description" in subset else [],
        })
    return profiles


def plot_top_terms(profiles, output):
    for profile in profiles:
        terms = profile["important_tfidf_terms"][:10]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(terms[::-1], np.arange(len(terms), 0, -1), color="#d92b36")
        ax.set(title=f"Cluster {profile['cluster']}: Leading TF-IDF Terms", xlabel="Relative rank", ylabel="Term")
        save_fig(fig, output, f"cluster_{profile['cluster']}_top_terms.png")


def plot_dendrogram(linkage_matrix, output, sample_size):
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=28, show_leaf_counts=True,
               show_contracted=True, ax=ax, no_labels=True)
    ax.set(title=f"Ward Dendrogram (Representative Sample, n={sample_size})",
           xlabel="Sample groups shown at leaves", ylabel="Ward linkage distance")
    save_fig(fig, output, "dendrogram.png")
