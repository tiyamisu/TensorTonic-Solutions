import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    try:
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            return None

        n = X.shape[0]

        # Need at least 2 samples
        if n < 2:
            return None

        # Center data
        X_centered = X - np.mean(X, axis=0)

        # Covariance matrix
        cov = (X_centered.T @ X_centered) / (n - 1)

        # Standard deviations
        std = np.sqrt(np.diag(cov))

        # Compute correlation (NaNs appear automatically for zero variance)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = cov / np.outer(std, std)

        return corr

    except:
        return None