import numpy as np

def expected_value_discrete(x, p):
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    if len(x) != len(p):
        raise ValueError("x and p must have the same length")

    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative")

    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must sum to 1")

    return float(np.dot(x, p))