"""Unit 3: mixtures, latent-variable methods, and clustering."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

FEATURES = ["Return_1D", "Return_5D", "Volatility_5D", "Volatility_20D", "RSI", "MACD"]
OUT = "results/unit3_clustering.png"


def heading(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def prepare(data):
    d = data.dropna(subset=FEATURES + ["Target_Direction", "Date"]).sort_values("Date")
    train = d[d["Date"] <= "2021-12-31"]
    validation = d[(d["Date"] > "2021-12-31") & (d["Date"] <= "2023-12-31")]
    scaler = StandardScaler().fit(train[FEATURES])
    x_train = scaler.transform(train[FEATURES])
    x_validation = scaler.transform(validation[FEATURES])
    print(f"Unsupervised observations: {len(x_train)} train, {len(x_validation)} validation.")
    return train, x_train, x_validation


def latent_dimensions(x_train, x_validation):
    heading("Latent variables: factor analysis, PCA, and dimension choice")
    pca = PCA().fit(x_train)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for threshold in (0.90, 0.95):
        count = int(np.searchsorted(cumulative, threshold) + 1)
        print(f"PCA components for {threshold:.0%} explained variance: {count} of {x_train.shape[1]}")

    max_dimensions = min(5, x_train.shape[1] - 1, len(x_validation) - 1)
    scores = []
    for dimensions in range(1, max_dimensions + 1):
        model = FactorAnalysis(n_components=dimensions, random_state=42)
        model.fit(x_train)
        scores.append(model.score(x_validation))
    best_dimensions = int(np.argmax(scores) + 1)
    print("Factor analysis validation log-likelihood by latent dimension:")
    print("  " + ", ".join(f"{i + 1}:{score:.3f}" for i, score in enumerate(scores)))
    print(f"Selected factor dimension by validation likelihood: {best_dimensions}")
    factors = FactorAnalysis(n_components=best_dimensions, random_state=42).fit(x_train)
    loadings = pd.DataFrame(factors.components_.T, index=FEATURES,
                            columns=[f"Factor {i + 1}" for i in range(best_dimensions)])
    print("Factor loadings (feature-to-latent-factor relationships):")
    print(loadings.round(2).to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, "o-")
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=1)
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    ax.set(xlabel="number of PCA components", ylabel="cumulative explained variance",
           title="Choosing latent dimensions with PCA")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/unit3_dimensions.png", dpi=140)
    plt.close(fig)


def compare_clustering_methods(train, x_train):
    heading("Clustering: K-means, Gaussian mixtures, dissimilarity, and evaluation")
    print("Euclidean and cosine dissimilarity for a sample pair:")
    a, b = x_train[0:1], x_train[1:2]
    euclidean = np.linalg.norm(a - b)
    cosine = 1 - ((a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)).item()
    print(f"  Euclidean={euclidean:.3f}; cosine distance={cosine:.3f}")

    sample_size = min(2000, len(x_train))
    rng = np.random.default_rng(42)
    rows = np.sort(rng.choice(len(x_train), sample_size, replace=False))
    x = x_train[rows]
    direction = train["Target_Direction"].to_numpy()[rows]

    # K-means uses hard assignments and minimizes within-cluster squared distance.
    candidates = []
    print("K-means: silhouette (higher is more separated), Davies-Bouldin (lower is better):")
    for k in range(2, 6):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(x)
        labels = model.labels_
        silhouette = silhouette_score(x, labels, sample_size=min(1000, len(x)), random_state=42)
        db = davies_bouldin_score(x, labels)
        candidates.append((silhouette, k, model, labels))
        print(f"  k={k}: silhouette={silhouette:.3f}, Davies-Bouldin={db:.3f}")
    _, best_k, kmeans, kmeans_labels = max(candidates, key=lambda item: item[0])
    print(f"Selected K-means cluster count by silhouette: {best_k}")
    print(f"Adjusted Rand score vs. next-day direction (exploratory only): {adjusted_rand_score(direction, kmeans_labels):.3f}")

    # A Gaussian mixture assigns soft probabilities; EM alternates responsibilities and parameter updates.
    bic_scores = []
    models = []
    for components in range(1, 6):
        model = GaussianMixture(n_components=components, covariance_type="full",
                                n_init=2, max_iter=100, random_state=42).fit(x)
        bic_scores.append(model.bic(x))
        models.append(model)
    best_gmm = models[int(np.argmin(bic_scores))]
    gmm_labels = best_gmm.predict(x)
    responsibilities = best_gmm.predict_proba(x)
    print("Gaussian mixture (soft clusters; fitted by EM):")
    print("  BIC by component count: " + ", ".join(f"{i + 1}:{score:.1f}" for i, score in enumerate(bic_scores)))
    print(f"  selected components={best_gmm.n_components}; EM iterations={best_gmm.n_iter_}; lower bound={best_gmm.lower_bound_:.3f}")
    print(f"  mean maximum responsibility={responsibilities.max(axis=1).mean():.3f}")
    print("  Alternative EM view: each observation contributes fractionally to each cluster, weighted by its responsibility.")
    if len(np.unique(gmm_labels)) > 1:
        gmm_silhouette = silhouette_score(x, gmm_labels, sample_size=min(1000, len(x)), random_state=42)
        print(f"  GMM silhouette={gmm_silhouette:.3f}")
    else:
        print("  GMM silhouette is undefined for a single selected component.")

    # Hierarchical clustering is quadratic in sample size; use a fixed sample for a readable dendrogram.
    hierarchical_size = min(600, len(x))
    hx = x[:hierarchical_size]
    tree = linkage(hx, method="ward", metric="euclidean")
    hierarchy = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(hx)
    hlabels = hierarchy.labels_
    print("Hierarchical clustering (Ward linkage):")
    print(f"  clusters={best_k}; silhouette={silhouette_score(hx, hlabels):.3f}; "
          f"direction ARI={adjusted_rand_score(direction[:hierarchical_size], hlabels):.3f}")

    # PCA gives a 2D view of the K-means hard assignments.
    coordinates = PCA(n_components=2).fit_transform(x)
    fig, ax = plt.subplots(figsize=(6, 5))
    points = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=kmeans_labels,
                        s=9, alpha=0.55, cmap="tab10")
    ax.set(title=f"K-means clusters shown in PCA space (k={best_k})",
           xlabel="principal component 1", ylabel="principal component 2")
    fig.colorbar(points, ax=ax, label="cluster")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(tree, truncate_mode="lastp", p=25, ax=ax, no_labels=True)
    ax.set(title="Hierarchical clustering dendrogram (Ward linkage)", xlabel="merged observations/clusters",
           ylabel="merge distance")
    fig.tight_layout()
    fig.savefig("results/unit3_hierarchy.png", dpi=140)
    plt.close(fig)


def run(data):
    print("\nUNIT 3 - MIXTURE MODELS, LATENT DIMENSIONS, AND CLUSTERING")
    train, x_train, x_validation = prepare(data)
    latent_dimensions(x_train, x_validation)
    compare_clustering_methods(train, x_train)
    print("Unit 3 complete. Plots: results/unit3_dimensions.png, results/unit3_clustering.png, results/unit3_hierarchy.png")
