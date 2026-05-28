import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """

    X = np.array(X, dtype=float)

    mean = np.mean(X, axis=0)
    X_centered = X - mean

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    indices = np.argsort(eigenvalues)[::-1]

    top_vectors = eigenvectors[:, indices[:k]]

    projected = np.dot(X_centered, top_vectors)

    return projected.tolist()