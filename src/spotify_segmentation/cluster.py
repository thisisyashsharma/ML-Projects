# src/spotify_segmentation/cluster.py
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score

def make_clustering(method="minibatchkmeans", n_clusters=20, batch_size=4096, max_iter=100, random_state=42, **kwargs):
    """
    Create a clustering estimator based on 'method'. Accepts extra kwargs to be future-proof.
    method: "minibatchkmeans" (default) or "kmeans"
    """
    method = (method or "minibatchkmeans").lower()
    if method == "minibatchkmeans":
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
            n_init="auto",
            **kwargs
        )
    elif method == "kmeans":
        return KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_init="auto",
            **kwargs
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")

def score_clusters(X, labels, sample_size=10000):
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.RandomState(0)
        idx = rng.choice(n, sample_size, replace=False)
        Xs = X[idx]; labs = labels[idx]
    else:
        Xs, labs = X, labels
    return silhouette_score(Xs, labs)
