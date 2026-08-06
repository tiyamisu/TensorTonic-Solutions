import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """

    v = np.asarray(v)

    # Single vector: shape (3,)
    if v.ndim == 1:
        return np.sqrt(np.sum(v * v))

    # Batch of vectors: shape (N, 3)
    return np.sqrt(np.sum(v * v, axis=1))