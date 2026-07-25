import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.asarray(X, dtype=float).copy()

    # Handle 1D input
    if X.ndim == 1:
        if strategy == 'mean':
            value = np.nanmean(X)
        else:
            value = np.nanmedian(X)

        if np.isnan(value):
            value = 0.0

        X[np.isnan(X)] = value
        return X

    # Handle 2D input
    for j in range(X.shape[1]):
        col = X[:, j]

        if strategy == 'mean':
            value = np.nanmean(col)
        else:
            value = np.nanmedian(col)

        # Entire column is NaN
        if np.isnan(value):
            value = 0.0

        col[np.isnan(col)] = value
        X[:, j] = col

    return X
    