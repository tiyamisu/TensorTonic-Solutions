import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)
    left = y[split_mask]
    right = y[~split_mask]
    if len(left) == 0 or len(right) == 0:
        return 0.0
    parent_entropy = _entropy(y)
    left_entropy = _entropy(left)
    right_entropy = _entropy(right)
    n = len(y)
    weighted_entropy = (
        (len(left) / n) * left_entropy +
        (len(right) / n) * right_entropy
    )
    ig = parent_entropy - weighted_entropy
    return float(ig)