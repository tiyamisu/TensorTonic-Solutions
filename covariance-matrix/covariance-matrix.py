import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """

    X = np.array(X, dtype=float)

    if X.ndim != 2 or X.shape[0] < 2:
        return None

    mean = np.mean(X, axis=0)

    X_centered = X - mean

    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

    return cov