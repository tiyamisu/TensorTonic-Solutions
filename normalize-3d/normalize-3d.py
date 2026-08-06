import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """

    v = np.asarray(v, dtype=float)

    # Single vector
    if v.ndim == 1:
        norm = np.sqrt(np.sum(v * v))
        if norm == 0:
            return np.zeros_like(v)
        return v / norm

    # Batch of vectors
    norms = np.sqrt(np.sum(v * v, axis=1))

    result = np.zeros_like(v)
    nonzero = norms != 0
    result[nonzero] = v[nonzero] / norms[nonzero][:, None]

    return result