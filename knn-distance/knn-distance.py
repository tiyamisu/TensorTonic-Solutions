import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    distances = np.sqrt(
        np.sum(
            (X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2,
            axis=2
        )
    )

    indices = np.argsort(distances, axis=1)

    n_test = X_test.shape[0]

    result = np.full((n_test, k), -1, dtype=int)

    valid_k = min(k, X_train.shape[0])

    result[:, :valid_k] = indices[:, :valid_k]

    return result