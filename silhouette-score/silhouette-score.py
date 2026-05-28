import numpy as np
def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    X = np.array(X, dtype=float)
    labels = np.array(labels)
    n = len(X)
    dist = np.sqrt(((X[:, np.newaxis] - X[np.newaxis, :]) ** 2).sum(axis=2))
    unique_labels = np.unique(labels)
    scores = []
    for i in range(n):
        same_cluster = labels == labels[i]
        same_cluster[i] = False
        
        if np.sum(same_cluster) > 0:
            a = np.mean(dist[i][same_cluster])
        else:
            a = 0.0
        b = np.inf
        
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = labels == label
            if np.sum(other_cluster) > 0:
                b = min(b, np.mean(dist[i][other_cluster]))
        s = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
        scores.append(s)
    return float(np.mean(scores))