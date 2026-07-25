import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = np.asarray(y, dtype=int)

    if num_classes is None:
        num_classes = np.max(y) + 1

    one_hot_matrix = np.zeros((len(y), num_classes), dtype=int)
    one_hot_matrix[np.arange(len(y)), y] = 1

    return one_hot_matrix