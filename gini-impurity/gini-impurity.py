import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """

    def gini(y):
        y = np.array(y)
        
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)
        
    n_left = len(y_left)
    n_right = len(y_right)
    total = n_left + n_right

    if total == 0:
        return 0.0
    weighted_gini = (
        (n_left / total) * gini(y_left) +
        (n_right / total) * gini(y_right)
    )
    
    return float(weighted_gini)